"""Kairos replica implementation."""
from __future__ import annotations

import json
import math
import os
import socket
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


MAX_MESSAGE = 2 * 1024 * 1024
MODES = ("observe", "duplicate-lru", "lease", "prefix-lease", "coordinated", "frontier", "loan",
         "ledger", "span", "history-tail", "history-value", "history-cover", "history-cover-wide", "history-horizon", "history-active")


def is_qwen3_8b_cache_model(model):
    """Validate the supported layout without rewriting LMCache's identity.

    vLLM offline resolution can replace the HF ID with its snapshot path.
    Hash lookup must keep that exact path; normalizing it breaks native keys.
    """
    if model == "Qwen/Qwen3-8B":
        return True
    parts = Path(model).parts
    return (len(parts) >= 3 and parts[-3:-1] ==
            ("models--Qwen--Qwen3-8B", "snapshots") and bool(parts[-1]))


def replica_model_supported(model):
    """Only the checkpoint layout included in the released experiment."""
    return is_qwen3_8b_cache_model(model)


def replica_bytes_per_token():
    if os.environ.get('CONTINUUM_GAPPED_REUSE') == '1':
        model = os.environ.get('CONTINUUM_GAPPED_MODEL_ID', 'Qwen/Qwen3-8B')
        if not is_qwen3_8b_cache_model(model):
            raise ValueError('Unvalidated full-attention cache layout')
    return 147456


@dataclass(frozen=True)
class ReplicaKey:
    model: str
    salt: str
    chunk_hash: str
    rank: int = 0
    group: int = 0
    # Position in the contiguous token prefix certified by a program.  The
    # native LMCache ObjectKey does not carry this field, so matching code
    # uses the remaining identity fields and treats this as control-plane
    # metadata only.
    chunk_index: int = field(default=-1, compare=False)

    def base(self):
        return (self.model, self.salt, self.chunk_hash, self.rank, self.group)


@dataclass(frozen=True)
class Grant:
    deadline: float
    keys: tuple[ReplicaKey, ...]


class LeaseRegistry:
    def __init__(self, mode="observe"):
        if mode not in MODES:
            raise ValueError("Unsupported replica mode")
        self.mode = mode
        self.lock = threading.RLock()
        self.grants: dict[str, Grant] = {}
        self.borrowed: dict[str, set[ReplicaKey]] = {}
        from kairos._runtime.compat.history_policy import HistoryPolicy
        self.history = HistoryPolicy()

    def grant(self, program, keys, deadline):
        if not program or not math.isfinite(deadline):
            raise ValueError("Invalid grant")
        with self.lock:
            self.grants[program] = Grant(deadline, tuple(keys))
            self.borrowed[program] = self.borrowed.get(program, set()).intersection(keys)

    def revoke(self, program, preserve_borrowed=False):
        with self.lock:
            self.grants.pop(program, None)
            if not preserve_borrowed:
                self.borrowed.pop(program, None)

    def record_deleted(self, keys):
        """Account only successful deletions while a GPU certificate exists."""
        with self.lock:
            for key in keys:
                for program, _, certified in self._matches(key):
                    self.borrowed.setdefault(program, set()).add(certified)

    def borrowed_missing(self, program, resident_keys):
        with self.lock:
            resident = {key.base() for key in resident_keys}
            return {key for key in self.borrowed.get(program, set())
                    if key.base() not in resident}

    def settle_committed(self, program, committed_keys):
        with self.lock:
            ready = {key.base() for key in committed_keys}
            if program in self.borrowed:
                self.borrowed[program] = {key for key in self.borrowed[program]
                                          if key.base() not in ready}

    def deadlines(self, now=None):
        with self.lock:
            result = {}
            # A grant represents an actual GPU reference, not merely a future
            # TTL promise. Continuum may retain that reference past its nominal
            # deadline while a return request is queued. Only an acknowledged
            # revoke means the CPU object is no longer a known duplicate.
            for grant in self.grants.values():
                for key in grant.keys:
                    result[key] = max(result.get(key, 0), grant.deadline)
            return result

    def _matches(self, key):
        base = key.base()
        return [(program, grant, item) for program, grant in self.grants.items()
                for item in grant.keys if item.base() == base]

    def protected_deadline(self, key):
        matches = self._matches(key)
        return max((grant.deadline for _, grant, _ in matches), default=None)

    def protected(self, key):
        return self.protected_deadline(key) is not None

    def _select_prefix_safe(self, keys, target, now, resident_keys=None):
        """Select victims while preserving a small contiguous CPU prefix.

        LMCache's normal lookup is a leading-prefix lookup.  Removing an
        interior chunk can therefore make every later chunk unusable even if
        it remains in L1.  This mode only removes a program's highest
        certified chunks and keeps a configurable floor at the beginning.
        It is deliberately conservative: when the floor prevents satisfying
        the native request, the remaining native candidates are considered.
        """
        remaining = list(keys)
        selected = []
        # Keep four 256-token chunks by default (configurable by the caller).
        floor = (0 if self.mode in ("coordinated", "loan", "ledger", "span") else
                 max(0, int(getattr(self, "prefix_floor", 4))))
        resident = {key.base() for key in (resident_keys or keys)}
        debt = {program: sum(key.base() not in resident for key in borrowed)
                for program, borrowed in self.borrowed.items()}
        limit = max(0, int(getattr(self, "repair_chunk_limit", 8)))
        tails = {}
        positions = {}
        for program, grant in self.grants.items():
            items = [item for item in grant.keys
                     if item.chunk_index >= 0 and item.base() in resident]
            if items:
                tails[program] = max(item.chunk_index for item in items)
                for item in items:
                    positions.setdefault(item.base(), []).append((program, item.chunk_index))

        while remaining and len(selected) < max(0, target):
            safe = []
            for key in remaining:
                owners = positions.get(key.base(), [])
                # Uncertified/native-only objects remain eligible.  Certified
                # objects are eligible only while their deletion is a suffix
                # and leaves the configured prefix floor.
                if not owners:
                    safe.append(key)
                    continue
                if all(index == tails.get(program, index)
                       and index >= floor
                       and (self.mode not in ("loan", "ledger", "span") or debt.get(program, 0) < limit)
                       for program, index in owners):
                    safe.append(key)
            if not safe:
                break
            deadlines = {key: self.protected_deadline(key) for key in safe}
            current = time.monotonic() if now is None else now
            if self.mode in ("prefix-lease", "coordinated", "loan", "ledger", "span"):
                safe.sort(key=lambda key: (
                    deadlines[key] is None,
                    -max(0.0, deadlines[key] - current)
                    if deadlines[key] is not None else 0.0))
            elif self.mode == "duplicate-lru":
                safe.sort(key=lambda key: deadlines[key] is None)
            # observe/unknown preserves the native order in `remaining`.
            chosen = safe[0]
            selected.append(chosen)
            remaining.remove(chosen)
            for program, index in positions.get(chosen.base(), []):
                debt[program] = debt.get(program, 0) + 1
                if tails.get(program) == index:
                    tails[program] = index - 1
        return selected

    def select(self, ordered_keys, target, eligible=lambda key: True, now=None,
               resident_keys=None):
        """Preserve native LRU among equal priorities and unprotected objects."""
        with self.lock:
            keys = [key for key in ordered_keys if eligible(key)]
            if self.mode in ("history-tail", "history-value", "history-cover", "history-cover-wide", "history-horizon", "history-active"):
                return self.history.select(keys, target,
                    ordered_keys if resident_keys is None else resident_keys,
                    self.grants, self.mode, now)
            if self.mode == "frontier":
                return self._select_restore_frontier(
                    keys, target, now, ordered_keys if resident_keys is None else resident_keys)
            if self.mode in ("prefix-lease", "coordinated", "loan", "ledger", "span"):
                return self._select_prefix_safe(
                    keys, target, now,
                    ordered_keys if resident_keys is None else resident_keys)
            if self.mode == "observe":
                return keys[:max(0, target)]
            deadlines = self.deadlines(now)
            if self.mode == "duplicate-lru":
                keys.sort(key=lambda key: key not in deadlines)
            else:
                now = time.monotonic() if now is None else now
                keys.sort(key=lambda key: (
                    key not in deadlines,
                    -max(0.0, deadlines.get(key, 0)-now)))
            return keys[:max(0, target)]

    def _select_restore_frontier(self, keys, target, now, resident_keys):
        """Borrow a bounded leading CPU duplicate, preserving its suffix.

        GPU APC evicts unpinned tail blocks before its leading prefix in the
        measured allocator. Range-aware restore can combine that surviving
        prefix with a CPU suffix; deleting the same suffix on both tiers loses
        recoverability. Never create more backup debt than a bounded repair
        plan can cover, even across several pressure passes of one grant.
        """
        resident = {key.base() for key in resident_keys}
        positions, heads, missing = {}, {}, {}
        limit = max(0, int(getattr(self, "repair_chunk_limit", 8)))
        for program, grant in self.grants.items():
            items = [item for item in grant.keys if item.chunk_index >= 0]
            present = [item for item in items if item.base() in resident]
            missing[program] = len(items)-len(present)
            if present:
                heads[program] = min(item.chunk_index for item in present)
            for item in items:
                positions.setdefault(item.base(), []).append((program, item.chunk_index))
        remaining, selected = list(keys), []
        current = time.monotonic() if now is None else now
        while remaining and len(selected) < max(0, target):
            duplicates = [key for key in remaining
                          if positions.get(key.base()) and all(
                              index == heads.get(program) and missing[program] < limit
                              for program, index in positions[key.base()])]
            if duplicates:
                duplicates.sort(key=lambda key: -max(0, (self.protected_deadline(key) or 0)-current))
                chosen = duplicates[0]
            else:
                # Fall back to native LRU of unprotected objects; a protected
                # non-leading chunk is not a safe complementary victim.
                unique = [key for key in remaining if not self.protected(key)]
                if not unique:
                    break
                chosen = unique[0]
            selected.append(chosen)
            remaining.remove(chosen)
            resident.discard(chosen.base())
            for program, index in positions.get(chosen.base(), []):
                missing[program] += 1
                heads[program] = min((item.chunk_index for item in self.grants[program].keys
                                      if item.base() in resident and item.chunk_index >= 0), default=-1)
        return selected


def _request_bytes(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_MESSAGE:
        raise ValueError("Replica control message is too large")
    return raw


def _response_result(response):
    if not response.endswith(b"\n") or len(response) > MAX_MESSAGE:
        raise RuntimeError("Invalid replica control response")
    result = json.loads(response)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Replica control failure"))
    return result


def rpc(path, payload, timeout=2.0):
    raw = _request_bytes(payload)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(raw)
        with sock.makefile("rb") as stream:
            response = stream.readline(MAX_MESSAGE + 1)
    return _response_result(response)


class ControlClient:
    """One ordered, acknowledged control stream; never retry a mutation.

    The process and path are fixed for a method. Any protocol/dispatch error
    closes the stream and poisons this client: reconnecting and repeating an
    operation could replay a deletion whose acknowledgement was lost.
    """
    def __init__(self, path, timeout=2.0):
        self.path = str(path)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock = None
        self._stream = None
        self._failed = False

    def _close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def close(self):
        with self._lock:
            self._close()

    def call(self, payload):
        raw = _request_bytes(payload)
        with self._lock:
            if self._failed:
                raise RuntimeError('Failed control stream; mutation must not be replayed')
            try:
                if self._sock is None:
                    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self._sock.settimeout(self.timeout)
                    self._sock.connect(self.path)
                    self._stream = self._sock.makefile('rb')
                self._sock.sendall(raw)
                return _response_result(self._stream.readline(MAX_MESSAGE + 1))
            except Exception:
                self._failed = True
                self._close()
                raise


class ControlServer:
    """Private filesystem socket. Refuse to overwrite any existing path."""
    def __init__(self, path, dispatch):
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                # Only the explicit stream arm stays connected. One-shot
                # clients close after their first acknowledged operation.
                # Shutdown closes registered connections to wake idle reads.
                with connection_lock:
                    connections.add(self.connection)
                try:
                    while True:
                        raw = self.rfile.readline(MAX_MESSAGE + 1)
                        if not raw:
                            return
                        try:
                            if not raw.endswith(b"\n") or len(raw) > MAX_MESSAGE:
                                raise ValueError("Invalid control message size")
                            result = dict(dispatch(json.loads(raw)), ok=True)
                        except Exception as exc:
                            result = dict(ok=False, error=str(exc))
                        self.wfile.write(json.dumps(result).encode() + b"\n")
                        self.wfile.flush()
                        if not result['ok']:
                            return
                except (OSError, ValueError):
                    return
                finally:
                    with connection_lock:
                        connections.discard(self.connection)

        connection_lock = self.connection_lock = threading.Lock()
        connections = self.connections = set()
        self.server = socketserver.ThreadingUnixStreamServer(str(self.path), Handler)
        self.server.daemon_threads = True
        os.chmod(self.path, 0o600)
        self.inode = self.path.stat().st_ino
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="replica-control", daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        with self.connection_lock:
            for connection in self.connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.path.exists() and self.path.stat().st_ino == self.inode:
            self.path.unlink()
