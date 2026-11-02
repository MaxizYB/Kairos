"""CPU byte-copy and shared-owner coverage invariants."""

import math
import types
import unittest
from kairos._runtime.compat.replica import ReplicaKey
from kairos._runtime.compat.history_policy import HistoryPolicy
from kairos.backend.coverage import CoverageHistory
from kairos.backend.transfer import RangeTransfer, Unavailable, prefix_work


def key(name, salt="p"):
    return ReplicaKey("qwen", salt, name, 0)


class Lazy:
    pass


class Tensor:
    def __init__(self, data, shape):
        self.data, self.shape, self.device = (data, shape, "cpu")

    def is_contiguous(self):
        return True

    def element_size(self):
        return 2

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return id(self)


class Source:
    def __init__(self, serial, layers=2, hidden=8):
        n = 2 * layers * 256 * hidden * 2
        self.tensor = Tensor(
            bytearray(((i + serial * 17) % 251 for i in range(n))),
            (2, layers, 256, hidden),
        )
        self.locked = True

    def parent(self):
        return Lazy()

    def get_tensor(self, i):
        if i != 0:
            raise ValueError("bad local group")
        return self.tensor

    def get_size(self):
        return len(self.tensor.data)


class Stream:
    def __init__(self):
        self.pending, self.drains = ([], 0)

    def synchronize(self):
        self.drains += 1
        while self.pending:
            self.pending.pop(0)()


def transfer_fixture(skip=240, count=3):
    stream, records, final, pointers = (Stream(), [], {}, {})
    sources = [Source(i) for i in range(count)]
    for obj in sources:
        pointers[obj.tensor.data_ptr()] = obj
    ids = [[i + 10 for i in range(count * 16)]]
    for layer in range(2):
        for kv in range(2):
            for bid in ids[0]:
                final[layer, kv, bid] = bytearray([255] * (16 * 8 * 2))
    read_bytes, staged_bytes = ([0], [0])
    desc = types.SimpleNamespace(bs=16, kv_size=2, element_size=2, nl=2, nh=1, hs=8)
    context = types.SimpleNamespace(
        device="cuda:0",
        stream=stream,
        lmcache_tokens_per_chunk=256,
        kv_layer_groups_manager=types.SimpleNamespace(
            object_groups=[types.SimpleNamespace(kernel_group_indices=[0])],
            get_attn_desc=lambda: types.SimpleNamespace(
                is_full_attention=lambda _: True
            ),
        ),
        get_shape_desc=lambda _: desc,
        get_engine_kv_format=lambda _: "full-nhd",
        get_slots_per_chunk_in_sw=lambda _: 256,
        calculate_num_blocks=lambda n, kg: n // 16,
        get_kernel_group_kv_pointers=lambda _: "native-layer-pointers",
    )

    def kernel(ptrs, obj_ptrs, dest, device, direction, shape, chunk, fmt, skip_pages):
        obj = pointers[obj_ptrs[0]]

        def execute():
            if not obj.locked:
                raise AssertionError("source released before DMA fence")
            for l in range(2):
                for kv in range(2):
                    for page in range(skip_pages, 16):
                        offset = ((kv * 2 + l) * 256 + page * 16) * 8 * 2
                        value = obj.tensor.data[offset : offset + 16 * 8 * 2]
                        final[l, kv, dest[page]][:] = value
                        read_bytes[0] += len(value)

        stream.pending.append(execute)

    module = types.SimpleNamespace(
        device_ops=types.SimpleNamespace(multi_layer_block_kv_transfer=kernel),
        lmcache_native=types.SimpleNamespace(
            TransferDirection=types.SimpleNamespace(H2D="H2D", D2H="D2H"),
            EngineKVFormat=types.SimpleNamespace(NL_X_TWO_NB_BS_NH_HS="full-nhd"),
        ),
    )
    calls = []

    def original(c, dests, objects, group, batch, skipped, direction):
        calls.append((len(objects), skipped, dests))
        for i, obj in enumerate(objects):
            if (i + 1) * 256 <= skipped:
                continue
            staged_bytes[0] += obj.get_size()
            kernel(
                None,
                [obj.tensor.data_ptr()],
                dests[0][i * 16 : (i + 1) * 16],
                None,
                None,
                desc,
                256,
                None,
                max(0, (skipped - i * 256) // 16),
            )

    adapter = RangeTransfer(
        original,
        module,
        Lazy,
        mapper=lambda pointer: pointer,
        emit=lambda name, **row: records.append(dict(event=name, **row)),
    )
    return types.SimpleNamespace(
        adapter=adapter,
        c=context,
        ids=ids,
        sources=sources,
        stream=stream,
        records=records,
        final=final,
        calls=calls,
        reads=read_bytes,
        staged=staged_bytes,
        skip=skip,
    )


class PhysicalRestore(unittest.TestCase):
    def test_all_boundary_alignments_execute_exact_bytes_and_preserve_hit_pages(self):
        for skip in range(16, 768, 16):
            f = transfer_fixture(skip)
            f.adapter(f.c, f.ids, f.sources, 0, 4, skip, "H2D")
            self.assertTrue(
                all((v == bytearray([255] * 256) for v in f.final.values()))
            )
            f.stream.synchronize()
            for token_page, bid in enumerate(f.ids[0]):
                for l in range(2):
                    for kv in range(2):
                        if token_page * 16 < skip:
                            expected = bytearray([255] * 256)
                        else:
                            chunk, local = divmod(token_page, 16)
                            offset = ((kv * 2 + l) * 256 + local * 16) * 8 * 2
                            expected = f.sources[chunk].tensor.data[
                                offset : offset + 256
                            ]
                        self.assertEqual(f.final[l, kv, bid], expected)
            self.assertEqual(f.reads[0], (768 - skip) * 2 * 2 * 8 * 2)
            if skip % 256:
                record = f.records[-1]
                self.assertEqual(record["selected_payload_bytes"], f.reads[0])
                self.assertEqual(record["staged_payload_bytes"], f.staged[0])
                self.assertEqual(
                    record["avoided_source_payload_bytes"], skip % 256 * 2 * 2 * 8 * 2
                )
                self.assertFalse(record["completed"])

    def test_failure_drains_actual_pending_reads_before_native_unlock(self):
        f = transfer_fixture()

        def fail(*args):
            raise RuntimeError("native suffix enqueue failed")

        f.adapter.original = fail
        with self.assertRaisesRegex(RuntimeError, "suffix"):
            f.adapter(f.c, f.ids, f.sources, 0, 4, 240, "H2D")
        self.assertEqual(f.stream.drains, 1)
        self.assertEqual(f.stream.pending, [])
        self.assertGreater(f.reads[0], 0)
        self.assertEqual(f.records[-1]["event"], "forecast_restore_drained_failure")
        self.assertFalse(
            any((e["event"] == "forecast_restore_enqueued" for e in f.records))
        )

    def test_mapping_failure_falls_back_before_any_direct_enqueue(self):
        f = transfer_fixture()
        f.adapter.mapper = lambda _: (_ for _ in ()).throw(Unavailable("unmapped"))
        f.adapter(f.c, f.ids, f.sources, 0, 4, 240, "H2D")
        self.assertEqual(f.calls[0][:2], (3, 240))
        self.assertEqual(f.records[0]["event"], "forecast_restore_fallback")
        f.stream.synchronize()
        self.assertEqual(f.staged[0], sum((s.get_size() for s in f.sources)))

    def test_short_destination_missing_source_and_unaligned_skip_fail_before_enqueue(
        self,
    ):
        for kind in ("short", "missing", "unaligned"):
            f = transfer_fixture()
            if kind == "short":
                f.ids[0].pop()
            if kind == "missing":
                f.sources[-1] = None
            with self.assertRaises(ValueError):
                f.adapter(
                    f.c,
                    f.ids,
                    f.sources,
                    0,
                    4,
                    239 if kind == "unaligned" else 240,
                    "H2D",
                )
            self.assertFalse(f.stream.pending)
            self.assertFalse(f.records)

    def test_noncontiguous_mapped_address_falls_back_before_direct_launch(self):
        f = transfer_fixture()
        first = f.sources[0].tensor.data_ptr()
        f.adapter.mapper = lambda pointer: (
            pointer if pointer == first else pointer + 4096
        )
        f.adapter(f.c, f.ids, f.sources, 0, 4, 240, "H2D")
        self.assertEqual(f.records[0]["event"], "forecast_restore_fallback")
        self.assertIn("contiguous", f.records[0]["reason"])
        self.assertEqual(f.calls[0][:2], (3, 240))

    def test_complete_prefix_does_no_io_and_store_is_native(self):
        f = transfer_fixture()
        f.adapter(f.c, f.ids, f.sources, 0, 4, 768, "H2D")
        self.assertFalse(f.calls)
        self.assertFalse(f.stream.pending)
        self.assertEqual(prefix_work(3, 256, 16, 999 * 16).missing_tokens, 0)
        f.adapter.original = lambda *args: "native-store"
        self.assertEqual(
            f.adapter(f.c, f.ids, f.sources, 0, 4, 0, "D2H"), "native-store"
        )


class Residency(unittest.TestCase):
    def test_shared_key_protects_every_live_owner(self):
        h = CoverageHistory()
        shared = key("shared")
        tail = key("tail")
        h.update("a", "tool", (shared,), now=0)
        h.update("b", "tool", (shared, tail), now=0)
        h.observe_gpu({"a": 1, "b": 0}, now=1, event_tracked=True)
        self.assertEqual(
            h.select([shared], 1, [shared, tail], {}, "history-cover", 1, True), []
        )
        h.update("b", "finished", now=1)
        self.assertEqual(
            h.select([shared], 1, [shared, tail], {}, "history-cover", 1, True),
            [shared],
        )

    def test_stale_gpu_advice_and_locked_head_do_not_authorize_deletion(self):
        h = CoverageHistory()
        keys = (key("0"), key("1"), key("2"))
        h.update("p", "tool", keys, now=0)
        h.observe_gpu({"p": 2}, now=1)
        self.assertEqual(h.select([keys[0]], 1, keys, {}, "history-cover", 2, True), [])
        h.observe_gpu({"p": 2}, now=2, event_tracked=True)
        self.assertEqual(h.select([keys[1]], 1, keys, {}, "history-cover", 2, True), [])
        h.withdraw_gpu_pages({"p": [0]})
        self.assertEqual(h.select([keys[0]], 1, keys, {}, "history-cover", 2, True), [])

    def test_compatible_admission_keeps_boundary_backups(self):
        old, new = (HistoryPolicy(), CoverageHistory())
        keys = tuple((key(str(i)) for i in range(6)))
        for h in (old, new):
            h.update("p", "tool", keys, now=0)
            h.observe_gpu({"p": 3}, now=1, event_tracked=True)
        self.assertEqual(
            new.admission_order(list(keys), {}, now=1),
            old.admission_order(list(keys), {}, now=1),
        )
        self.assertEqual(
            new.admission_order(list(keys), {}, now=1),
            list(keys[3:]) + list(reversed(keys[:3])),
        )


class BackendIntegration(unittest.TestCase):
    def test_store_budget_shares_event_tracked_coverage_and_keeps_one_bridge(self):
        h = CoverageHistory()
        keys = tuple((key(str(i)) for i in range(6)))
        h.update("p", "tool", keys, now=0)
        h.observe_gpu({"p": 3}, now=1, event_tracked=True)
        kept, skipped, audit = h.write_budget(keys, [], 0, {}, now=100)
        self.assertEqual(kept, list(keys[2:]))
        self.assertEqual(skipped, list(keys[:2]))
        self.assertTrue(audit["all_owners_checked"])
        h.withdraw_gpu_pages({"p": [0]})
        kept, skipped, _ = h.write_budget(keys, [], 0, {}, now=100)
        self.assertEqual(kept, list(keys))
        self.assertEqual(skipped, [])

    def test_shared_store_key_essential_for_any_owner_is_kept(self):
        h = CoverageHistory()
        keys = tuple((key(str(i)) for i in range(3)))
        h.update("a", "tool", keys, now=0)
        h.update("b", "tool", keys, now=0)
        h.observe_gpu({"a": 3, "b": 0}, now=1, event_tracked=True)
        self.assertEqual(h.write_budget(keys, [], 0, {}, now=1)[0], list(keys))
