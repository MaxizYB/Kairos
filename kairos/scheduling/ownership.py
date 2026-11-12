"""Settle logical KV capacity against live ownership at existing boundaries."""


class ReservationSettlement:
    def _ownership(self, program, occupied=None):
        if occupied is None:
            occupied = self._occupied()
        pin = self.pins.get(program)
        request = self.online_requests.get(program)
        return dict(
            occupied=program in occupied,
            cleaning=bool(self.online_cleaning.get(program)),
            pin_pages=len(pin.blocks) if pin is not None else 0,
            native_pages=self._native_pages(request) if request is not None else 0,
            pin_present=pin is not None,
            request=request.request_id if request is not None else None,
        )

    def _sync_idle_frontier(self, program):
        result = super()._sync_idle_frontier(program)
        proof = self._ownership(program)
        if proof["occupied"] or proof["cleaning"] or proof["pin_present"]:
            return result
        before = self.online_reserved.get(program)
        pages = proof["native_pages"]
        if pages:
            self.online_reserved[program] = pages
        else:
            self.online_reserved.pop(program, None)
            self.online_active.discard(program)
            self.online_insurance.pop(program, None)
        after = self.online_reserved.get(program)
        if before != after:
            self.online_dirty = True
            self._emit(
                "reservation_settlement",
                program=program,
                before=before,
                after=after,
                ownership=proof,
                physical_blocks_freed=0,
                future_information=False,
            )
        return result

    def _sync_frontiers(self):
        result = super()._sync_frontiers()
        occupied = self._occupied()
        # Existing admission/lifecycle boundaries only; no per-token replan.
        candidates = (
            set(self.online_reserved)
            - occupied
            - set(self.online_cleaning)
            - set(self.pins)
        )
        for program in sorted(candidates):
            self._sync_idle_frontier(program)
        return result

    def _emit(self, event, **fields):
        if event in ("program_service", "forecast_service"):
            occupied = self._occupied()
            proof = {p: self._ownership(p, occupied) for p in self.online_reserved}
            for p, pages in self.online_reserved.items():
                row = proof[p]
                if pages > 0 and not (
                    row["occupied"]
                    or row["cleaning"]
                    or row["pin_pages"]
                    or row["native_pages"]
                ):
                    raise RuntimeError("Unowned logical KV capacity: " + p)
            fields["reservation_ownership"] = proof
            fields["reservation_settlement_enabled"] = True
        return super()._emit(event, **fields)
