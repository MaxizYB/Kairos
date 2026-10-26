"""Kairos hicache policy implementation."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import heapq
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA = "3dc91366ac04b314f856839cb1a76e9a66aac9e7"
CORE_BLOB = "d1acd88b1e0e15cb0359da8b0977b3445616ccd0"
COMPONENT_BLOB = "3ec838c382a6407b7be6434252c0b05682833310"


def upstream_methods(filename, blob, names):
    path = Path(__file__).resolve().parents[1] / "vendor/hicache-3dc9136" / filename
    data = path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != blob:
        raise RuntimeError("HiCache policy source differs from verified Git blob")
    tree = ast.parse(data)
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != set(names):
        raise RuntimeError("Missing official HiCache policy method")
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *methods], type_ignores=[]))
    scope = dict(BASE_COMPONENT_TYPE=0, heapq=heapq)
    exec(compile(module, str(path), "exec"), scope)
    return {name: scope[name] for name in names}


@dataclass(eq=False)
class HostNode:
    id: int
    key: object = None
    parent: object = None
    children: dict = field(default_factory=dict)
    component_data: list = field(default_factory=list)
    write_through_pending_id: object = None
    load_back_pending_id: object = None
    native_rank: int = 0

    @property
    def evicted(self):
        return self.component_data[0].value is None

    @property
    def backuped(self):
        return self.component_data[0].host_value is not None

    def __lt__(self, other):
        return self.id < other.id


class HostRulePort:
    def __init__(self):
        core = upstream_methods("unified_tree_core.py", CORE_BLOB, (
            "_reclaim_full_host_duplicates", "_can_reclaim_full_host_duplicate",
            "_is_device_leaf", "_is_host_leaf"))
        comp = upstream_methods("full_component.py", COMPONENT_BLOB, ("drive_host_eviction",))
        for name, method in core.items():
            setattr(self, name, method.__get__(self))
        self._official_host_walk = comp["drive_host_eviction"].__get__(self)
        self.root_node = HostNode(0, component_data=[self.data(False, False, False)])
        self.full_host_duplicates = {}
        self._node_ids = {}
        self._serial = 0
        self.component_type = 0
        self.tree_core = self
        self.evictable_device_leaves = set()
        self.evictable_host_leaves = set()
        self._deleted = []
        self._duplicate_deleted = []
        self._events = []

    @staticmethod
    def data(gpu, host, locked, gpu_locked=False):
        return SimpleNamespace(value=True if gpu else None,
            host_value=True if host else None, host_lock_ref=int(locked), lock_ref=int(gpu_locked))

    def _ensure_eviction_strategy(self):
        pass

    def session_ref_eviction_strategy(self, node):
        # Session cache disabled; preserve the native LRU priority among leaves.
        return node.native_rank

    def _release_full_host_duplicate(self, node, tracker, device_frees, host_frees):
        self._delete(node, tracker, duplicate=True)

    def _evict_host_leaf(self, node, tracker, device_frees, host_frees):
        if self._delete(node, tracker, duplicate=False):
            self.evictable_host_leaves.discard(node)
            parent = node.parent
            parent.children.pop(node.id, None)
            # Prune payload-free metadata after removal, as tree leaf cleanup.
            while parent is not self.root_node and parent.evicted and not parent.backuped and not parent.children:
                old = parent
                parent = old.parent
                parent.children.pop(old.id, None)
            if self._is_host_leaf(parent):
                self.evictable_host_leaves.add(parent)

    def _delete(self, node, tracker, duplicate):
        success = self._delete_native(node.key)
        self._events.append(dict(key=node.key, duplicate=duplicate, success=success,
            imminent=node in self.evictable_device_leaves))
        if not success:
            # The native deletion may lose a lock race. Do not count its bytes.
            node.component_data[0].host_lock_ref = 1
            return False
        node.component_data[0].host_value = None
        tracker[0] += self._chunk_tokens
        self._deleted.append(node.key)
        if duplicate:
            self._duplicate_deleted.append(node.key)
        return True

    def reclaim(self, rows, gpu, target_chunks, delete_native, chunk_tokens=256):
        """rows: (native key, base identity, Program, index, eligible, rank).

        gpu maps base identity to whether all source pages are currently
        present and to whether any page is referenced. Locks/wanted/inflight
        exclusions enter eligibility before the official walks execute.
        """
        self._delete_native = delete_native
        self._chunk_tokens = chunk_tokens
        self._deleted, self._duplicate_deleted, self._events = [], [], []
        nodes = {}
        chains = {}
        for key, base, program, index, eligible, rank in rows:
            ident = (program, index) if program is not None else ("unknown", base)
            if ident not in self._node_ids:
                self._serial += 1
                self._node_ids[ident] = self._serial
            covered, referenced = gpu.get(base, (False, False))[:2]
            node = HostNode(self._node_ids[ident], key=key,
                component_data=[self.data(covered, True, not eligible, referenced)], native_rank=rank)
            nodes[base] = node
            if program is not None:
                chains.setdefault(program, {})[index] = node
            else:
                node.parent = self.root_node
        # GPU-only nodes must remain in the path: they can make a duplicate
        # an ancestor rather than a device leaf even without a host payload.
        for base, info in gpu.items():
            if base in nodes or len(info) < 4:
                continue
            covered, referenced, program, index = info
            if not covered:
                continue
            ident = (program, index)
            if ident not in self._node_ids:
                self._serial += 1
                self._node_ids[ident] = self._serial
            node = HostNode(self._node_ids[ident],
                component_data=[self.data(True, False, False, referenced)])
            nodes[base] = node
            chains.setdefault(program, {})[index] = node
        self.root_node.children.clear()
        for program, chain in chains.items():
            parent = self.root_node
            for index in sorted(chain):
                node = chain[index]
                node.parent = parent
                parent.children[node.id] = node
                parent = node
        for node in nodes.values():
            if node.parent is self.root_node:
                self.root_node.children[node.id] = node
        self.evictable_device_leaves = {node for node in nodes.values() if self._is_device_leaf(node)}
        self.evictable_host_leaves = {node for node in nodes.values() if self._is_host_leaf(node)}
        # Keep insertion order for duplicate nodes that survived; add new
        # duplicates at the end. This pass is not the LRU leaf fallback.
        duplicates = {node.id: node for node in nodes.values() if not node.evicted and node.backuped}
        self.full_host_duplicates = {ident: duplicates[ident] for ident in self.full_host_duplicates if ident in duplicates}
        self.full_host_duplicates.update({ident: node for ident, node in duplicates.items()
                                         if ident not in self.full_host_duplicates})
        tracker, frees = {0: 0}, {0: []}
        self._reclaim_full_host_duplicates(target_chunks * chunk_tokens, tracker, frees, frees)
        duplicate_count = len(self._duplicate_deleted)
        self._official_host_walk(target_chunks * chunk_tokens, tracker, frees, frees)
        return dict(deleted=list(self._deleted), duplicate_deleted=duplicate_count,
            host_leaf_deleted=len(self._deleted)-duplicate_count, decisions=list(self._events))


class DeviceRulePort:
    """Original HiCache device-leaf walk over native 16-token APC pages.

    The backend retains its real page granularity. All known program paths
    are unshared (cache salts); referenced pages cannot enter the free set.
    Selection only simulates demotion; the caller copies and acknowledges
    complete host chunks before allowing native reuse of selected pages.
    """
    def __init__(self, reuse_nodes=False):
        core = upstream_methods("unified_tree_core.py", CORE_BLOB, ("_is_device_leaf",))
        comp = upstream_methods("full_component.py", COMPONENT_BLOB, (
            "_evict_device_start", "_evict_device_next_node", "_evict_device_end"))
        for name, method in {**core, **comp}.items():
            setattr(self, name, method.__get__(self))
        self.tree_core = self
        self.component_type = 0
        self.root_node = HostNode(0, component_data=[HostRulePort.data(False, False, False)])
        self.reuse_nodes = reuse_nodes
        self._device_nodes = {}

    def _ensure_eviction_strategy(self):
        pass

    def session_ref_eviction_strategy(self, node):
        return node.native_rank

    def choose(self, free_ids, layouts, count):
        self.root_node.children.clear()
        ranks = {bid: i for i, bid in enumerate(free_ids)}
        nodes = {}
        serial = 0
        for row in layouts:
            parent = self.root_node
            for bid in row["blocks"]:
                if bid is None:
                    continue
                if bid in nodes:
                    raise ValueError("HiCache mechanism port requires unshared program paths")
                serial += 1
                node = self._device_nodes.get(bid) if self.reuse_nodes else None
                if node is None:
                    node = HostNode(serial, key=bid,
                        component_data=[HostRulePort.data(True, False, False, bid not in ranks)])
                else:
                    node.children.clear()
                    node.component_data[0].value = True
                    node.component_data[0].lock_ref = int(bid not in ranks)
                node.id = serial
                node.parent = parent
                node.native_rank = ranks.get(bid, len(ranks))
                parent.children[node.id] = node
                parent = node
                nodes[bid] = node
        if self.reuse_nodes:
            # Only reuse metadata objects; recompute the same ordered paths,
            # serial tie-breakers and native locks before the official walk.
            self._device_nodes = nodes
        known = set(nodes)
        # Uncached or untracked pages have no recoverable path dependency.
        selected = [bid for bid in free_ids if bid not in known][:count]
        self.evictable_device_leaves = {n for n in nodes.values() if self._is_device_leaf(n)}
        by_id = {n.id: n for n in nodes.values()}
        tracker, frees = {0: len(selected)}, {0: []}
        self._evict_device_start(count)
        while len(selected) < count:
            nid = self._evict_device_next_node(tracker, frees, frees)
            if nid is None:
                break
            node = by_id[nid]
            selected.append(node.key)
            tracker[0] += 1
            node.component_data[0].value = None
            self.evictable_device_leaves.discard(node)
            if self._is_device_leaf(node.parent):
                self.evictable_device_leaves.add(node.parent)
        self._evict_device_end()
        # Native free capacity stays authoritative for unusual sparse paths.
        if len(selected) < count:
            selected.extend(bid for bid in free_ids if bid not in set(selected))
        return selected[:count]
