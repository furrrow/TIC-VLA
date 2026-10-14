"""Character commands for benchmark episodes with fixed pedestrian routes."""


def character_name(index):
    return "Character" if index == 0 else f"Character_{index:02d}"


def generate_scripted_commands(humans, route_cycles=20):
    """Repeat routes so characters keep walking throughout a benchmark episode."""
    lines = []
    for index, human in enumerate(humans):
        name = character_name(index)
        route = human.get("walk_route", [])
        if not route:
            lines.append(f"{name} Idle 10000")
            continue
        lines.append(f"{name} Idle {human['initial_idle']}")
        for _ in range(route_cycles):
            for x, y, z in route:
                lines.append(f"{name} GoTo {x} {y} {z} _")
    return "\n".join(lines) + "\n"
