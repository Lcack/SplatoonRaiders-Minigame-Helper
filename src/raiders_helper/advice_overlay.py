"""Normalized display geometry; never participates in policy decisions."""


def advice_geometry(observation, decision):
    diagnostics = decision.diagnostics
    route = diagnostics.get("route", [])
    lines = list(zip(route, route[1:]))
    markers = []
    player = next(iter(observation.of_kind("player")), None)
    target = diagnostics.get("target")
    if isinstance(target, dict) and player:
        markers.append((target["x"], player.y))
    elif isinstance(target, (list, tuple)) and len(target) == 2:
        markers.append(target)
    for side, target in diagnostics.get("targets", {}).items():
        actors = observation.of_kind("player" if side == "left" else "partner")
        if target and actors:
            markers.append((target["target_x"], actors[0].y))
    return lines, markers
