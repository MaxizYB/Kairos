"""Kairos lmcache server implementation."""
import json
import os
import runpy
from pathlib import Path

from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBus


def main():
    # The wrapper is process-local.  It is intentionally installed before
    # LMCache constructs its L1 manager, so the native allocator and lock
    # protocol remain unchanged.
    if os.environ.get("CONTINUUM_REPLICA_SOCKET"):
        import kairos._runtime.compat.lmcache_replica as lmcache_replica
        if os.environ.get("CONTINUUM_GAPPED_REUSE") == "1":
            from kairos._runtime.layout.runtime import GappedRuntime
            lmcache_replica.ReplicaRuntime = GappedRuntime
            if os.environ.get("CONTINUUM_PCV_CONTROL") == "1":
                from kairos._runtime.layout.pcv_runtime import PCVRuntime
                lmcache_replica.ReplicaRuntime = PCVRuntime
        if os.environ.get("CONTINUUM_VALIDATION_BASELINES") == "1":
            from kairos._runtime.compat.baseline_runtime import BaselineRuntime
            lmcache_replica.ReplicaRuntime = BaselineRuntime
        if os.environ.get("CONTINUUM_RECOVERY_PROBE") == "1":
            from kairos._runtime.compat.recovery_probe import runtime_class
            lmcache_replica.ReplicaRuntime = runtime_class()
        lmcache_replica.install()
    path=Path(os.environ['CONTINUUM_TRANSFER_LOG'])
    original_init=EventBus.__init__

    def initialize(bus,*args,**kwargs):
        original_init(bus,*args,**kwargs)
        def record(event):
            from kairos._runtime.compat.lmcache_replica import serializable
            with path.open('a') as file:
                file.write(json.dumps(serializable(dict(event=event.event_type.value,
                    time=event.timestamp,request=event.session_id,
                    metadata=event.metadata)))+'\n')
        for event in (EventType.MP_STORE_START,EventType.MP_STORE_END,
                      EventType.MP_RETRIEVE_START,EventType.MP_RETRIEVE_END):
            bus.subscribe(event,record)

    # This process-local constructor wrapper leaves installed files intact.
    EventBus.__init__=initialize
    # HTTP and cache-control flags belong to the integrated HTTP frontend,
    # not the ZMQ-only server. It exposes /metrics on the same HTTP port.
    runpy.run_module('lmcache.v1.multiprocess.http_server',run_name='__main__')


if __name__=='__main__':main()
