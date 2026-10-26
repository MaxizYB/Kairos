"""Kairos probation scheduler implementation."""
import time

from kairos._runtime.compat.online_path_scheduler import OnlinePathScheduler


class ProbationScheduler(OnlinePathScheduler):
    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        tags=set(name.split('~'))
        protocols=tags & {'probation','probationslack','probationparallel',
            'probationparallelcarry','probationparallelguard','probationparallelprecise',
            'probationparallelcohort','probationparallelcohortfree'}
        if len(protocols)>1:
            raise ValueError('Choose one explicit capacity protocol, not a combination of old variants')
        super()._switch_experiment(name)
        self.probation_carry = 'probationparallelcarry' in name.split('~')
        self.probation_guard = 'probationparallelguard' in name.split('~')
        self.probation_precise = 'probationparallelprecise' in name.split('~')
        self.probation_cohort = 'probationparallelcohort' in name.split('~')
        # A cohort credit preserves a returned Program's service eligibility
        # without consuming one of the finite resident slots.  The original
        # cohort arm intentionally coupled both states; this arm isolates the
        # state-separation hypothesis and never changes physical capacity.
        self.probation_cohort_free = ('probationparallelcohortfree' in name.split('~'))
        self.probation_atomic = 'atomicreclaim' in name.split('~')
        self.probation_parallel = ('probationparallel' in name.split('~') or
                                    self.probation_carry or self.probation_guard or
                                    self.probation_precise or self.probation_cohort or
                                    self.probation_cohort_free)
        self.probation_retain = ('probationslack' in name.split('~') or
                                 self.probation_parallel)
        self.probation_enabled = ('probation' in name.split('~') or
                                   self.probation_retain)
        self.probation_loan = None
        self.probation_cooldown = set()
        self.probation_credits = {}
        self.probation_carry_owner = None
        if self.probation_enabled:
            if self.online_mode != 'onlinefitresidual':
                raise ValueError('Probation requires the original r12 execution contract')
            self._emit('probation_registry', revision=('one-call-funded-return-v4-carry'
                       if self.probation_carry else 'one-call-funded-return-v6-cohort-free-slot'
                       if self.probation_cohort_free else 'one-call-funded-return-v5-cohort'
                       if self.probation_cohort else 'one-call-funded-return-v4-deficit'
                       if self.probation_precise else 'one-call-funded-return-v4-no-trim'
                       if self.probation_guard else 'one-call-funded-return-v3'
                       if self.probation_parallel else 'one-call-escrow-slack-v2'
                       if self.probation_retain else 'one-call-escrow-v1'),
                future_information=False, max_outstanding=1,
                borrower='unentered arrived first call only',
                repayment='native physical cleanup; reserve max(donor pages, actual call pages)',
                cooldown='both parties must complete a normally funded call before lending',
                retained_prefix=('largest prefix funded alongside complete borrower call; '
                    'funded optional refs held through one-call repayment fence'
                    if self.probation_retain else 'none'),
                prediction='unchanged r12; no new model or future labels',
                continuation_deposit=self.probation_carry,
                service_lease_survives_optional_kv_reclaim=(self.probation_cohort or self.probation_cohort_free),
                cohort_credit_consumes_resident_slot=not self.probation_cohort_free,
                reclaim_preflight=self.probation_atomic,
                donor_must_retain_full_prefix=self.probation_guard,
                observed_growth_reclaim=('actual deficit; preserve funded prefix and membership'
                    if self.probation_precise else 'original r12 whole idle commitment'),
                optimality='bounded information-acquisition experiment; no JCT guarantee')

    def _probation_blocked(self, request):
        loan = self.probation_loan
        if loan is None:
            return False
        p = self.program(request)
        return (p == loan['donor'] and not loan.get('early_funded')) or (p == loan['borrower'] and
                                     request.request_id != loan['request'])

    def _probation_early_fund(self, request, limit):
        """A returned donor need not wait if both real complete calls fit.

        The provisional cohort slot is distinct from a native execution slot.
        No pin is revoked and no third Program can borrow the same credit.
        """
        loan = self.probation_loan; p = self.program(request)
        if not (self.probation_parallel and loan and p == loan['donor'] and
                not loan.get('early_funded')):
            return None
        borrower = loan['borrower']; target = self._online_target(request)
        used = sum(self.online_reserved.values())
        prospective = (used-self.online_reserved.get(p,0)-self.online_reserved.get(borrower,0)
                       +target+loan['actual_call_pages'])
        if prospective > limit:
            return False
        self.online_reserved[borrower] = loan['actual_call_pages']
        loan['early_funded'] = True
        self._emit('probation_early_fund', donor=p, borrower=borrower,
            target_pages=target, prospective_pages=prospective,
            reserved_pages=sum(self.online_reserved.values()))
        return True

    def _online_has_slot(self, p):
        if p == self.probation_carry_owner:
            # One deposited prefix may re-enter only after paying the whole
            # actual next call. The deposit itself is not a resident slot.
            return True
        loan = getattr(self, 'probation_loan', None)
        if (getattr(self, 'probation_parallel', False) and loan and
                p == loan['donor'] and loan.get('early_funded')):
            return True
        return super()._online_has_slot(p)

    def _online_reclaim(self, request, limit):
        if self.probation_enabled:
            early = self._probation_early_fund(request, limit)
            if early is not None:return early
        if self.probation_enabled and self._probation_blocked(request):
            return False
        if self.probation_atomic and not self._probation_reclaim_feasible(request,limit):
            self._emit('probation_reclaim_rejected',program=self.program(request),
                target_pages=self._online_target(request),
                reserved_pages=sum(self.online_reserved.values()),
                reason='whole call and resident slot cannot be funded; no refs revoked')
            return False
        if self.probation_precise and self.program(request) in self.online_active:
            return self._probation_reclaim_deficit(request, limit)
        p = self.program(request)
        eligible_credit = self.probation_cohort_free and p in self.probation_credits
        if ((self.probation_cohort or self.probation_cohort_free) and
                (p in self.online_active or eligible_credit)):
            # Cache reclaim frees pages, not another Program's service lease.
            # Keep only the cohort identities already paid for by the original
            # finite tool lease; lifetime `entered` does not confer this right.
            before={q:pin.deadline for q,pin in self.pins.items()
                    if q in self.online_active}
            funded=super()._online_reclaim(request,limit)
            for q in before.keys()-self.online_active:
                if not self.probation_cohort_free:
                    self.online_active.add(q)
                self.probation_credits[q]=before[q]
                self._emit('probation_cohort_credit',program=q,deadline=before[q],
                    physical_pin=q in self.pins,pages=self.online_reserved.get(q,0),
                    reserved_pages=sum(self.online_reserved.values()))
            return funded
        return super()._online_reclaim(request, limit)

    def _probation_reclaim_feasible(self,request,limit):
        """Dry-run the existing victim order, with no new rank or prediction.

        Scheduler control is synchronous: there is no allocator mutation or
        await between this check and the original commit. Running/granted/DMA
        references are excluded by the identical existing eligibility rule.
        """
        p=self.program(request);target=self._online_target(request)
        used=sum(self.online_reserved.values())-self.online_reserved.get(p,0)+target
        slots=set(self.online_active)
        has_slot=self._online_has_slot(p)
        def fits():
            return used<=limit and (has_slot or len(slots)<self.max_num_running_reqs)
        if fits():return True
        running={self.program(r) for r in self.running}
        victims=[q for q in self.pins if q!=p and q not in running and
                 q not in self.online_cleaning and self._online_idle_reclaimable(q) and
                 not (q in self.online_requests and
                      self.online_requests[q].request_id in self.online_granted)]
        victims.sort(key=lambda q:self.online_rank.get(q,self.order[q]),reverse=True)
        for q in victims:
            used-=self.online_reserved.get(q,0);slots.discard(q)
            if fits():return True
        return False

    def _probation_reclaim_deficit(self, request, limit):
        """Pay observed growth with only the suffixes actually required.

        The original victim order and immutable references are unchanged.
        Build a complete feasible transaction before modifying any pin.
        A retained prefix keeps resident identity, so a partial revoke does
        not manufacture an all-or-nothing restoration wait on its next call.
        """
        p=self.program(request); target=self._online_target(request)
        used=sum(self.online_reserved.values())
        need=max(0,used-self.online_reserved.get(p,0)+target-limit)
        if not self._online_has_slot(p):return False
        if not need:return True
        running={self.program(r) for r in self.running}
        victims=[q for q in self.pins if q!=p and q not in running and
                 q not in self.online_cleaning and self._online_idle_reclaimable(q) and
                 not (q in self.online_requests and
                      self.online_requests[q].request_id in self.online_granted)]
        victims.sort(key=lambda q:self.online_rank.get(q,self.order[q]),reverse=True)
        actions=[]; missing=need
        for q in victims:
            old=self.online_reserved.get(q,0); count=len(self.pins[q].blocks)
            if not old or not count:continue
            release=min(count,missing)
            remaining=old-release if release<count else 0
            actions.append((q,release,remaining))
            missing-=old-remaining
            if missing<=0:break
        if missing>0:return False
        started=time.perf_counter()
        for q,count,remaining in actions:
            self._release(q,'online_observed_growth_partial',count=count)
            self.online_insurance.pop(q,None)
            if remaining:
                self.online_reserved[q]=remaining
            else:
                self.online_reserved.pop(q,None);self.online_active.discard(q)
            self._emit('online_path_recovery_partial',program=p,victim=q,
                target_pages=target,requested_deficit_pages=need,
                released_references=count,victim_reserved_pages=remaining,
                reserved_pages=sum(self.online_reserved.values()),
                seconds=time.perf_counter()-started)
        return True

    def _online_restore_fits(self, request, limit):
        if self.probation_enabled:
            early = self._probation_early_fund(request, limit)
            if early is not None:return early
        if self.probation_enabled and self._probation_blocked(request):
            return False
        if (self.probation_cohort_free and
                self.program(request) in self.probation_credits):
            # A finite service lease can request a resident-slot exchange;
            # it neither occupies a slot before return nor bypasses its
            # funding. Lifetime `entered` alone never confers this right.
            return self._online_reclaim(request, limit)
        return super()._online_restore_fits(request, limit)

    def _expire_idle_protection(self, now, queued):
        loan = self.probation_loan
        if (self.probation_enabled and self.probation_retain and loan is not None
                and not loan.get('early_funded')):
            # These refs are fully funded, but the donor temporarily lent its
            # cohort slot. Keep their accounting intact through repayment.
            queued = set(queued) | {loan['donor']}
        super()._expire_idle_protection(now, queued)
        if (self.probation_carry_owner is not None and
                self.probation_carry_owner not in self.pins and
                self.probation_carry_owner not in queued):
            self._emit('probation_carry_expired', program=self.probation_carry_owner)
            self.probation_carry_owner = None
        if not self.probation_enabled:
            return
        for p, deadline in list(self.probation_credits.items()):
            if now >= deadline and p not in queued and p not in self.online_cleaning:
                pages = self.online_reserved.pop(p, 0)
                self.online_active.discard(p)
                del self.probation_credits[p]
                self.online_dirty = True
                self._emit('probation_credit_expired', program=p, pages=pages,
                           reserved_pages=sum(self.online_reserved.values()))

    def _online_idle_reclaimable(self, program):
        loan = self.probation_loan
        return not (self.probation_enabled and loan and program == loan['donor']
                    and not loan.get('early_funded'))

    def _online_grant(self, request, pages, reason):
        p = self.program(request)
        if (self.probation_carry_owner == p and
                getattr(self, 'probation_carry_request', None) != request.request_id):
            self._emit('probation_carry_consumed', program=p, request=request.request_id)
            self.probation_carry_owner = None
        self.probation_credits.pop(self.program(request), None)
        return super()._online_grant(request, pages, reason)

    def _online_control(self, queued):
        dirty = self.online_dirty
        super()._online_control(queued)
        if (not self.probation_enabled or not dirty or self.probation_loan is not None
                or self.probation_carry_owner is not None):
            return
        started = time.perf_counter()
        pending = [r for r in queued if r.request_id not in self.online_granted]
        # Actual continuations retain first claim; probation never jumps an
        # already waiting return and cannot form an unbounded admission chain.
        if any(self.program(r) in self.online_entered for r in pending):
            return
        borrowers = sorted(pending, key=lambda r:self.online_rank.get(
            self.program(r), self.order[self.program(r)]))
        running = {self.program(r) for r in self.running}
        donors = [p for p in self.pins if p in self.online_active and
                  p not in self.online_requests and p not in running and
                  p not in self.online_cleaning and p not in self.probation_cooldown]
        donors.sort(key=lambda p:self.online_rank.get(p, self.order[p]), reverse=True)
        used = sum(self.online_reserved.values()); limit = self._capacity()-32
        for request in borrowers:
            p = self.program(request); target = self._online_target(request)
            for donor in donors:
                old = self.online_reserved.get(donor, 0)
                # Maximal affordable prefix has a closed form. Both the
                # complete call and retained optional KV are paid for; the
                # unused difference still escrows the full original credit.
                retained = (max(0, min(len(self.pins[donor].blocks), old-
                    max(0, used+target-limit))) if self.probation_retain else 0)
                if self.probation_guard and retained < len(self.pins[donor].blocks):
                    continue
                escrow = max(old-retained, target)
                if old <= 0 or used-old+retained+escrow > limit:
                    continue
                if len(self.online_active) > self.max_num_running_reqs:
                    continue
                pin = self.pins[donor]
                self.probation_loan = dict(donor=donor, borrower=p,
                    request=request.request_id, pages=old, escrow_pages=escrow,
                    retained_pages=retained,
                    actual_call_pages=target,
                    deadline=pin.deadline, started=time.monotonic(),
                    started_wall=time.time(), returned=False)
                self.probation_cooldown.update((donor, p))
                # Only an optional tool-period reference is revoked. Native
                # running and DMA references have been excluded above.
                if self.probation_retain:
                    count = len(pin.blocks)-retained
                    if count:self._release(donor, 'probation_lend_partial', count=count)
                else:
                    self._release(donor, 'probation_lend')
                self.online_active.discard(donor)
                if retained:self.online_reserved[donor] = retained
                else:self.online_reserved.pop(donor)
                self._emit('probation_layout', donor=donor, retained_pages=retained,
                           reserved_pages=sum(self.online_reserved.values()))
                self._online_grant(request, escrow, 'probation-first-call')
                self._emit('probation_lend', **self.probation_loan,
                    seconds=time.perf_counter()-started,
                    reserved_pages=sum(self.online_reserved.values()))
                return

    def add_request(self, request):
        super().add_request(request)
        loan = self.probation_loan
        if self.probation_enabled and loan and self.program(request) == loan['donor']:
            loan['returned'] = True
            self._emit('probation_donor_return', donor=loan['donor'],
                       borrower=loan['borrower'], request=request.request_id)

    def _continuum_complete_policy(self, request):
        super()._continuum_complete_policy(request)
        if not self.probation_enabled:
            return
        loan = self.probation_loan; p = self.program(request)
        if loan and request.request_id == loan['request']:
            # End this borrower's right at ONE call. Actual request/DMA refs
            # still live until native cleanup. A following call cannot use
            # this reservation, even if it arrives before cleanup.
            carry = (self.probation_carry and loan.get('early_funded') and
                     not self.metadata(request).get('cont_final', False))
            if carry:
                loan['borrower_carry_pending'] = True
                self._emit('probation_call_complete', borrower=p, donor=loan['donor'],
                           request=request.request_id, carry=True)
            else:
                self._release(p, 'probation_call_complete')
                self._emit('probation_call_complete', borrower=p, donor=loan['donor'],
                           request=request.request_id, carry=False)
        else:
            self.probation_cooldown.discard(p)

    def _free_request_blocks(self, request):
        result = super()._free_request_blocks(request)
        loan = self.probation_loan
        if not self.probation_enabled or loan is None or request.request_id != loan['request']:
            return result
        if not request.is_finished() or self.online_cleaning.get(loan['borrower']):
            return result
        borrower = loan['borrower']; donor = loan['donor']
        carry = bool(loan.get('borrower_carry_pending') and loan.get('early_funded') and
                     borrower not in self.online_ended and borrower in self.pins and
                     borrower in self.online_reserved)
        if carry:
            # The borrower keeps the already-funded one-call prefix as a
            # bounded continuation deposit. It is consumed by exactly the
            # next real request (or expires with its pin); no new capacity is
            # created and the donor's early-funded call remains untouched.
            self.online_active.discard(borrower)
            self.probation_carry_owner = borrower
            self.probation_carry_request = loan['request']
            self._emit('probation_repay', donor=donor, borrower=borrower,
                request=loan['request'], pages=0, restored=False,
                early_funded=True, borrower_carry=True,
                donor_returned=loan['returned'], elapsed_s=time.monotonic()-loan['started'],
                reserved_pages=sum(self.online_reserved.values()))
            self.probation_loan = None
            self.online_dirty = True
            return result
        # This is after the original native free, including its DMA fence.
        # A new borrower call might already be queued, but was never granted.
        self.online_reserved.pop(borrower, None)
        self.online_active.discard(borrower)
        early = loan.get('early_funded', False)
        if not early:self.online_reserved.pop(donor, None)
        restore = (donor not in self.online_ended and
                   (donor in self.online_requests or time.monotonic() < loan['deadline']))
        if restore and not early:
            self.online_reserved[donor] = loan['pages']
            self.online_active.add(donor)
            self.probation_credits[donor] = loan['deadline']
        elif not early and donor in self.pins:
            self._release(donor, 'probation_expired_repayment')
        self.probation_loan = None
        self.online_dirty = True
        self._emit('probation_repay', donor=donor, borrower=borrower,
            request=request.request_id, restored=restore, pages=loan['pages'] if restore else 0,
            early_funded=early,
            donor_returned=loan['returned'], elapsed_s=time.monotonic()-loan['started'],
            reserved_pages=sum(self.online_reserved.values()))
        if sum(self.online_reserved.values()) > self._capacity()-32:
            raise RuntimeError('Probation repayment spent escrowed capacity')
        return result
