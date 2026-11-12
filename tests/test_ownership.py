"""Exercise reservation settlement through the actual public scheduler MRO."""

from types import SimpleNamespace
import unittest

from support import controller_type


class OwnershipTests(unittest.TestCase):
    def fixture(self):
        cls = controller_type(public=True)
        state = object.__new__(cls)
        state.online_reserved, state.online_insurance = {"p": 299}, {}
        state.online_active, state.online_granted = {"p"}, set()
        state.online_cleaning, state.pins = {}, {}
        state.online_requests = {"p": SimpleNamespace(request_id="next-request")}
        state.online_dirty = False
        state._occupied = lambda: set()
        state._native_pages = lambda request: 0
        state.events = []
        state._emit = lambda event, **fields: state.events.append(
            dict(event=event, **fields)
        )
        return state

    def test_dma_cleanup_keeps_capacity_until_actual_retirement(self):
        state = self.fixture()
        state.online_cleaning = {"p": {"previous-request"}}
        state._sync_idle_frontier("p")
        self.assertEqual(state.online_reserved["p"], 299)
        state.online_cleaning.clear()
        state._sync_idle_frontier("p")
        self.assertNotIn("p", state.online_reserved)
        self.assertTrue(state.online_dirty)
        self.assertEqual(state.events[-1]["physical_blocks_freed"], 0)

    def test_occupied_owner_and_pin_remain_accounted(self):
        state = self.fixture()
        state._occupied = lambda: {"p"}
        state._sync_idle_frontier("p")
        self.assertEqual(state.online_reserved["p"], 299)
        state._occupied = lambda: set()
        state.pins = {"p": SimpleNamespace(blocks=list(range(17)))}
        state._sync_idle_frontier("p")
        self.assertEqual(state.online_reserved["p"], 17)

    def test_real_native_pages_are_kept_and_orphan_capacity_is_settled(self):
        state = self.fixture()
        state._native_pages = lambda request: 23
        state._sync_idle_frontier("p")
        self.assertEqual(state.online_reserved["p"], 23)
        state._native_pages = lambda request: 0
        state._sync_frontiers()
        self.assertFalse(state.online_reserved)
        self.assertNotIn("p", state.online_active)
