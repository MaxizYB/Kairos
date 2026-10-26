"""Kairos copy wait schedule implementation."""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap


def install_copy_wait_guard(scheduler_cls):
    original = scheduler_cls.schedule
    if hasattr(original, "_copy_wait_provenance"):
        return original._copy_wait_provenance
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    candidates = []
    for node in ast.walk(function):
        if not isinstance(node, ast.While):
            continue
        allocates = any(isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Attribute)
                        and stmt.value.func.attr == "allocate_slots"
                        for stmt in node.body)
        preempts = any(isinstance(call, ast.Call)
                       and isinstance(call.func, ast.Attribute)
                       and call.func.attr == "_preempt_request"
                       for call in ast.walk(node))
        if allocates and preempts:
            candidates.append(node)
    if len(candidates) != 1:
        raise RuntimeError("Unknown native running-allocation/preemption loop")
    loop = candidates[0]
    matched = [i for i, node in enumerate(loop.body)
               if isinstance(node, ast.If)
               and ast.dump(node.test) == ast.dump(ast.parse(
                   "new_blocks is not None", mode="eval").body)
               and len(node.body) == 1 and isinstance(node.body[0], ast.Break)]
    if len(matched) != 1:
        raise RuntimeError("Unknown native successful-allocation branch")
    guard = ast.parse(
        "if getattr(self, '_hicache_copy_wait', False) or "
        "getattr(self, '_pipeline_wait_for_fence', False):\n    break\n").body[0]
    loop.body.insert(matched[0]+1, guard)
    function.decorator_list = []
    tree.body.insert(0, ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0))
    tree = ast.fix_missing_locations(tree)
    scope = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), scope)
    adapted = scope[original.__name__]
    provenance = dict(native_schedule_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      mode="copy_wait_before_preemption", inserted_guards=1,
                      package_file_modified=False)
    adapted._copy_wait_provenance = provenance
    # A generated function's line numbers no longer point at the original
    # installed source. Later private adapters must compose this exact body,
    # rather than inspect an unrelated class/import at the same file offset.
    adapted._copy_wait_source = ast.unparse(function)
    scheduler_cls.schedule = adapted
    return provenance
