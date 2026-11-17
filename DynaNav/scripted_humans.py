"""Character commands for benchmark episodes with fixed pedestrian routes."""


def character_name(index):
    return "Character" if index == 0 else f"Character_{index:02d}"


def generate_scripted_commands(humans, route_cycles=20):
    """Generate commands for fixed pedestrian routes.

    A single-point route is a one-way walk: the character goes to that endpoint
    once and then stops.  Routes with multiple points are patrols and repeat for
    ``route_cycles`` as before.
    """
    lines = []
    for index, human in enumerate(humans):
        name = character_name(index)
        route = human.get("walk_route", [])
        if not route:
            lines.append(f"{name} Idle 10000")
            continue
        lines.append(f"{name} Idle {human['initial_idle']}")
        cycles = 1 if len(route) == 1 else route_cycles
        for _ in range(cycles):
            for x, y, z in route:
                lines.append(f"{name} GoTo {x} {y} {z} _")
        if len(route) == 1:
            lines.append(f"{name} Idle 10000")
    return "\n".join(lines) + "\n"
