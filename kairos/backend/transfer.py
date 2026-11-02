"""backend.transfer."""

from dataclasses import dataclass
import ctypes
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import threading
import time

REQUEST = ContextVar("forecast_restore_request", default={})


@dataclass(frozen=True)
class PrefixWork:
    first: int
    skip_pages: int
    suffix: int
    missing_tokens: int


def prefix_work(count, chunk, page, skip):
    if (
        any((type(x) is not int for x in (count, chunk, page, skip)))
        or count < 0
        or page < 1
        or (chunk < page)
        or chunk % page
        or (skip < 0)
        or skip % page
    ):
        raise ValueError(
            "Restore must use whole native pages inside complete CPU chunks"
        )
    start = min(count * chunk, skip)
    first, local = divmod(start, chunk)
    return PrefixWork(first, local // page, first + (local > 0), count * chunk - start)


class Unavailable(RuntimeError):
    pass


class MappedHost:
    def __init__(self):
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcudart" in line
            and ".so" in line
            and line.split()[-1].startswith("/")
        }
        if len(paths) != 1:
            raise Unavailable("Expected one already-loaded CUDA runtime")
        try:
            self.lib = ctypes.CDLL(next(iter(paths)))
            self.fn = self.lib.cudaHostGetDevicePointer
        except (OSError, AttributeError) as exc:
            raise Unavailable("Loaded CUDA runtime has no host mapping API") from exc
        self.fn.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        )
        self.fn.restype = ctypes.c_int

    def __call__(self, pointer):
        mapped = ctypes.c_void_p()
        code = self.fn(ctypes.byref(mapped), ctypes.c_void_p(pointer), 0)
        if code or not mapped.value:
            raise Unavailable("cudaHostGetDevicePointer failed: " + str(code))
        return mapped.value


class TransferLog:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def __call__(self, event, **row):
        with self.lock, self.path.open("a") as f:
            f.write(
                json.dumps(dict(event=event, time=time.time(), **REQUEST.get(), **row))
                + "\n"
            )


class RangeTransfer:
    def __init__(self, original, module, lazy_type, mapper=None, emit=None):
        self.original, self.module, self.lazy_type = (original, module, lazy_type)
        self.mapper = mapper
        self.emit = emit or (lambda *a, **k: None)

    def _mapped_launches(self, context, ids, obj, group_id, first, skip_pages):
        m = self.module
        group = context.kv_layer_groups_manager.object_groups[group_id]
        if not context.kv_layer_groups_manager.get_attn_desc().is_full_attention(
            group_id
        ):
            raise Unavailable("sliding/standalone object group")
        if obj is None or not isinstance(obj.parent(), self.lazy_type):
            raise Unavailable("source is not mapped LazyMemoryAllocator storage")
        chunk = context.lmcache_tokens_per_chunk
        launches, payload = ([], 0)
        for local, kg in enumerate(group.kernel_group_indices):
            desc = context.get_shape_desc(kg)
            fmt = context.get_engine_kv_format(kg)
            classic = m.lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
            packed = getattr(m.lmcache_native.EngineKVFormat, "NL_X_NB_BS_NH_CS", None)
            if (
                fmt not in (classic, packed)
                or desc.bs != 16
                or desc.kv_size != (2 if fmt == classic else 1)
                or (desc.element_size != 2)
                or (context.get_slots_per_chunk_in_sw(kg) != chunk)
            ):
                raise Unavailable(
                    "not the validated full-attention BF16/FP16 16-token layout"
                )
            tensor = obj.get_tensor(local)
            expected = (desc.kv_size, desc.nl, chunk, desc.nh * desc.hs)
            if (
                tensor is None
                or str(tensor.device) != "cpu"
                or (not tensor.is_contiguous())
                or (tuple(tensor.shape) != expected)
                or (tensor.element_size() != 2)
            ):
                raise Unavailable(
                    "source layout does not match native plane/layer/token kernel"
                )
            bpc = context.calculate_num_blocks(chunk, kg)
            if bpc != chunk // 16 or len(ids[kg]) < (first + 1) * bpc:
                raise ValueError("Short or inconsistent native destination table")
            if self.mapper is None:
                self.mapper = MappedHost()
            pointer = self.mapper(tensor.data_ptr())
            last = tensor.numel() * tensor.element_size() - 1
            if self.mapper(tensor.data_ptr() + last) != pointer + last:
                raise Unavailable("source mapping is not a contiguous device range")
            launches.append(
                (
                    context.get_kernel_group_kv_pointers(kg),
                    [pointer],
                    ids[kg][first * bpc : (first + 1) * bpc],
                    context.device,
                    m.lmcache_native.TransferDirection.H2D,
                    desc,
                    chunk,
                    fmt,
                    skip_pages,
                )
            )
            payload += (
                tensor.numel()
                * tensor.element_size()
                * (chunk - skip_pages * 16)
                // chunk
            )
        if not launches or obj.get_size() != sum(
            (
                obj.get_tensor(i).numel() * obj.get_tensor(i).element_size()
                for i in range(len(group.kernel_group_indices))
            )
        ):
            raise Unavailable("unexpected object padding or group layout")
        return (launches, payload)

    def __call__(
        self,
        cache_context,
        block_ids_gpu,
        memory_objs,
        object_group_id,
        batch_size,
        skip_first_n_tokens,
        direction,
    ):
        c, m = (cache_context, self.module)
        args = (
            c,
            block_ids_gpu,
            memory_objs,
            object_group_id,
            batch_size,
            skip_first_n_tokens,
            direction,
        )
        if direction != m.lmcache_native.TransferDirection.H2D:
            return self.original(*args)
        chunk = c.lmcache_tokens_per_chunk
        if skip_first_n_tokens < 0 or skip_first_n_tokens % 16:
            raise ValueError("Unaligned restore skip cannot be trusted")
        work = prefix_work(len(memory_objs), chunk, 16, skip_first_n_tokens)
        if not work.missing_tokens:
            self.emit("forecast_restore_empty", objects=len(memory_objs))
            return
        if not work.skip_pages:
            offset_ids = [
                ids[work.first * c.calculate_num_blocks(chunk, kg) :]
                for kg, ids in enumerate(block_ids_gpu)
            ]
            return self.original(
                c,
                offset_ids,
                memory_objs[work.first :],
                object_group_id,
                batch_size,
                0,
                direction,
            )
        try:
            if not str(c.device).startswith("cuda"):
                raise Unavailable("not a CUDA cache context")
            if batch_size < 1:
                raise ValueError("Invalid native batch size")
            group = c.kv_layer_groups_manager.object_groups[object_group_id]
            for kg in group.kernel_group_indices:
                bpc = c.calculate_num_blocks(chunk, kg)
                if len(block_ids_gpu[kg]) < len(memory_objs) * bpc:
                    raise ValueError("Restore destination underflow")
            if any((obj is None for obj in memory_objs[work.first :])):
                raise ValueError("Missing CPU source inside restored prefix")
            launches, payload = self._mapped_launches(
                c,
                block_ids_gpu,
                memory_objs[work.first],
                object_group_id,
                work.first,
                work.skip_pages,
            )
        except Unavailable as exc:
            self.emit(
                "forecast_restore_fallback",
                reason=str(exc),
                objects=len(memory_objs),
                skip_tokens=skip_first_n_tokens,
            )
            return self.original(*args)
        suffix_ids = [
            ids[work.suffix * c.calculate_num_blocks(chunk, kg) :]
            for kg, ids in enumerate(block_ids_gpu)
        ]
        suffix_objs = memory_objs[work.suffix :]
        try:
            for launch in launches:
                m.device_ops.multi_layer_block_kv_transfer(*launch)
            if suffix_objs:
                self.original(
                    c,
                    suffix_ids,
                    suffix_objs,
                    object_group_id,
                    batch_size,
                    0,
                    direction,
                )
            logical = sum((obj.get_size() for obj in memory_objs[work.first :]))
            staged = sum((obj.get_size() for obj in suffix_objs))
            self.emit(
                "forecast_restore_enqueued",
                source_objects=len(memory_objs) - work.first,
                skipped_pages=work.skip_pages,
                direct_payload_bytes=payload,
                staged_payload_bytes=staged,
                selected_payload_bytes=payload + staged,
                native_object_bytes=logical,
                avoided_staging_bytes=logical - staged,
                avoided_source_payload_bytes=logical - payload - staged,
                completion="native IPC event and finish_read_prefetched callback",
                completed=False,
                wire_bytes_measured=False,
            )
        except BaseException:
            c.stream.synchronize()
            self.emit("forecast_restore_drained_failure", partial_objects=1)
            raise


def install():
    from importlib.metadata import version
    from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as module

    if getattr(module.transfer_kv_per_object_group, "_kairos_forecast", False):
        return
    if version("lmcache") != "0.5.5rc4":
        raise RuntimeError("Kairos transfer adapter requires audited LMCache 0.5.5rc4")
    expected = (
        "cache_context",
        "block_ids_gpu",
        "memory_objs",
        "object_group_id",
        "batch_size",
        "skip_first_n_tokens",
        "direction",
    )
    if (
        tuple(inspect.signature(module.transfer_kv_per_object_group).parameters)
        != expected
    ):
        raise RuntimeError("Unexpected LMCache transfer ABI")
    for symbol in ("multi_layer_block_kv_transfer",):
        if not hasattr(module.device_ops, symbol):
            raise RuntimeError("Missing native transfer operation: " + symbol)
    log = TransferLog(os.environ["CONTINUUM_REPLICA_LOG"] + ".forecast-transfer.jsonl")
    original_retrieve = module.LMCacheDrivenTransferModule.retrieve

    @wraps(original_retrieve)
    def retrieve(owner, key, instance_id, *args, **kwargs):
        token = REQUEST.set(
            dict(
                request=key.request_id,
                instance_id=instance_id,
                start=key.start,
                end=key.end,
            )
        )
        try:
            return original_retrieve(owner, key, instance_id, *args, **kwargs)
        finally:
            REQUEST.reset(token)

    module.LMCacheDrivenTransferModule.retrieve = retrieve
    original = module.transfer_kv_per_object_group
    adapter = RangeTransfer(original, module, LazyMemoryAllocator, emit=log)
    adapter._kairos_forecast = True
    module.transfer_kv_per_object_group = adapter
    log(
        "forecast_transfer_registry",
        lmcache=version("lmcache"),
        native_kernel_reused=True,
        route="mapped partial boundary; native batched suffix",
        direct_attention=False,
        completion_owner="unchanged native retrieve",
        gpu_validation_required=True,
    )
