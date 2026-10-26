"""Kairos writeback plan implementation."""


def plan_writebacks(free_ids, layouts, count, attempted, limit=8):
    victims = set(free_ids[:count])
    plans = []
    remaining = limit
    for row in layouts:
        if row.get("state") == "finished":
            continue
        indices = []
        for index, group in enumerate(row["chunks"]):
            if remaining == 0:
                break
            key = (row["salt"], index, tuple(group))
            if (None in group or len(group) != 16 or index in row["cpu_chunks"]
                    or not victims.intersection(group) or key in attempted):
                continue
            indices.append(index)
            remaining -= 1
        if indices:
            plans.append((row, indices))
    return plans
