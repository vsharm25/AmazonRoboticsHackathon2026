from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

from ar_hackathon.models.graph_state import GraphState


_adjacency_cache: Dict[
    Tuple[Tuple[int, int, float, Optional[int], bool], ...],
    Dict[int, List[Tuple[int, float]]],
] = {}

_distance_cache: Dict[
    Tuple[Tuple[Tuple[int, int, float, Optional[int], bool], ...], int],
    Dict[int, float],
] = {}

# unit_id -> pod_id this unit has been assigned to fetch.
# Recomputed once per time step (not once per unit call) and shared
# by every unit asked that step.
_assignment_cache: Dict[str, object] = {
    "graph_key": None,
    "time_step": None,
    "unit_pickup": {},
}

# Cap on how many unclaimed pods the exact assignment solver will run on.
# It's exponential in this count; beyond the cap we fall back to a cheap
# greedy match instead of risking the per-call time limit.
_EXACT_MATCH_POD_LIMIT = 16


def _graph_key(
    state: GraphState,
) -> Tuple[Tuple[int, int, float, Optional[int], bool], ...]:
    return tuple(
        sorted(
            (
                edge.from_node,
                edge.to_node,
                edge.weight,
                edge.capacity,
                edge.bidirectional,
            )
            for edge in state.edges
        )
    )


def _adjacency(state: GraphState) -> Dict[int, List[Tuple[int, float]]]:
    key = _graph_key(state)

    if key not in _adjacency_cache:
        graph: Dict[int, List[Tuple[int, float]]] = {
            node.id: [] for node in state.nodes
        }

        for edge in state.edges:
            graph.setdefault(edge.from_node, []).append(
                (edge.to_node, edge.weight)
            )

            if edge.bidirectional:
                graph.setdefault(edge.to_node, []).append(
                    (edge.from_node, edge.weight)
                )

        _adjacency_cache[key] = graph

    return _adjacency_cache[key]


def _distances_from(state: GraphState, start: int) -> Dict[int, float]:
    key = (_graph_key(state), start)

    if key in _distance_cache:
        return _distance_cache[key]

    distances = {start: 0.0}
    queue = [(0.0, start)]
    graph = _adjacency(state)

    while queue:
        cost, node = heappop(queue)

        if cost != distances.get(node):
            continue

        for neighbor, weight in graph.get(node, []):
            new_cost = cost + weight

            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                heappush(queue, (new_cost, neighbor))

    _distance_cache[key] = distances
    return distances


def _distance(state: GraphState, start: int, goal: int) -> float:
    return _distances_from(state, start).get(goal, float("inf"))


def _is_available_step(
    state: GraphState,
    current: int,
    next_node: int,
) -> bool:
    edge = state.get_edge(current, next_node)

    if edge is None:
        return False

    if (
        edge.capacity is not None
        and state.edge_occupancy(current, next_node) >= edge.capacity
    ):
        return False

    node = state.get_node(next_node)

    if (
        node is not None
        and node.capacity is not None
        and state.node_occupancy(next_node) >= node.capacity
    ):
        return False

    return True


def _nearest_target(
    state: GraphState,
    start: int,
    targets: Set[int],
) -> Tuple[Optional[float], Optional[int]]:
    """
    Dijkstra outward from `start` until it hits the closest node in
    `targets`.

    Returns (cost_to_nearest_target, first_step_towards_it),
    or (None, None) if no target is reachable (or none was given).

    Only the very first step out of `start` is capacity-checked.
    """

    if not targets or start in targets:
        return None, None

    graph = _adjacency(state)
    distances = {start: 0.0}
    first_steps: Dict[int, int] = {}
    queue = [(0.0, start)]

    while queue:
        cost, node = heappop(queue)

        if cost != distances.get(node):
            continue

        if node in targets:
            return cost, first_steps.get(node)

        for neighbor, weight in graph.get(node, []):
            if node == start and not _is_available_step(
                state, node, neighbor
            ):
                continue

            new_cost = cost + weight

            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                first_steps[neighbor] = (
                    neighbor if node == start else first_steps[node]
                )
                heappush(queue, (new_cost, neighbor))

    return None, None


def _step_off_full_dock(state: GraphState, unit) -> Optional[int]:
    """
    Nothing to fetch and nothing to deliver.

    If we're idling on a full, capacity-limited node, step aside
    instead of blocking the next arrival.
    """

    node = state.get_node(unit.current_node)

    if node is None or node.capacity is None:
        return None

    if state.node_occupancy(unit.current_node) < node.capacity:
        return None

    graph = _adjacency(state)
    best_step = None
    best_weight = float("inf")

    for neighbor, weight in graph.get(unit.current_node, []):
        if not _is_available_step(
            state,
            unit.current_node,
            neighbor,
        ):
            continue

        if weight < best_weight:
            best_weight = weight
            best_step = neighbor

    return best_step


# ---------------------------------------------------------------------------
# Pod <-> unit assignment
#
# Solved exactly (maximize pods claimed, then minimize total travel)
# when the number of unclaimed pods is small enough.
#
# Otherwise use a cheap greedy fallback.
#
# Computed once per time step and shared by every unit asked that step,
# rather than recomputed per unit.
# ---------------------------------------------------------------------------

def _waiting_pods(state: GraphState):
    return [
        pod
        for pod in state.active_pods
        if pod.carried_by is None and pod.current_node is not None
    ]


def _pod_fetch_cost(state: GraphState, start: int, pod) -> float:
    """Cost to fetch this pod and carry it home."""

    return (
        _distance(state, start, pod.current_node)
        + _distance(
            state,
            pod.current_node,
            pod.destination_station,
        )
    )


def _exact_assignment(units, pods, costs) -> Dict[int, int]:
    """
    Bitmask dynamic programming:
    maximize pods assigned, then minimize total cost.
    """

    memo: Dict[
        Tuple[int, int],
        Tuple[int, float, Tuple[Optional[int], ...]],
    ] = {}

    def solve(
        unit_index: int,
        used_mask: int,
    ) -> Tuple[int, float, Tuple[Optional[int], ...]]:

        if unit_index == len(units):
            return 0, 0.0, ()

        key = (unit_index, used_mask)

        if key in memo:
            return memo[key]

        best_count, best_cost, best_assignment = solve(
            unit_index + 1,
            used_mask,
        )

        best_assignment = (None,) + best_assignment

        for pod_index in range(len(pods)):
            if used_mask & (1 << pod_index):
                continue

            cost = costs.get((unit_index, pod_index))

            if cost is None:
                continue

            rem_count, rem_cost, rem_assignment = solve(
                unit_index + 1,
                used_mask | (1 << pod_index),
            )

            candidate_count = rem_count + 1
            candidate_cost = cost + rem_cost

            if (
                candidate_count > best_count
                or (
                    candidate_count == best_count
                    and candidate_cost < best_cost
                )
            ):
                best_count = candidate_count
                best_cost = candidate_cost
                best_assignment = (
                    pod_index,
                ) + rem_assignment

        memo[key] = (
            best_count,
            best_cost,
            best_assignment,
        )

        return memo[key]

    _, _, assignment = solve(0, 0)

    return {
        units[i].id: pod_index
        for i, pod_index in enumerate(assignment)
        if pod_index is not None
    }


def _greedy_assignment(units, pods, costs) -> Dict[int, int]:
    """
    Cheapest-pair-first fallback for when there are too many pods
    for the exact solver to run safely within the per-call time budget.
    """

    candidates = sorted(
        costs.items(),
        key=lambda item: item[1],
    )

    used_units = set()
    used_pods = set()
    result: Dict[int, int] = {}

    for (unit_index, pod_index), _cost in candidates:
        if unit_index in used_units or pod_index in used_pods:
            continue

        result[units[unit_index].id] = pod_index

        used_units.add(unit_index)
        used_pods.add(pod_index)

    return result


def _update_assignment(state: GraphState) -> None:
    graph_key = _graph_key(state)

    if (
        _assignment_cache["graph_key"] == graph_key
        and _assignment_cache["time_step"] == state.current_time_step
    ):
        return

    _assignment_cache["graph_key"] = graph_key
    _assignment_cache["time_step"] = state.current_time_step

    unit_pickup: Dict[int, str] = {}
    _assignment_cache["unit_pickup"] = unit_pickup

    pods = _waiting_pods(state)

    units = [
        unit
        for unit in sorted(
            state.drive_units,
            key=lambda drive_unit: drive_unit.id,
        )
        if not unit.in_transit and unit.has_capacity
    ]

    if not pods or not units:
        return

    costs: Dict[Tuple[int, int], float] = {}

    for unit_index, unit in enumerate(units):
        for pod_index, pod in enumerate(pods):
            cost = _pod_fetch_cost(
                state,
                unit.current_node,
                pod,
            )

            if cost != float("inf"):
                costs[(unit_index, pod_index)] = cost

    if len(pods) <= _EXACT_MATCH_POD_LIMIT:
        assignment = _exact_assignment(
            units,
            pods,
            costs,
        )
    else:
        assignment = _greedy_assignment(
            units,
            pods,
            costs,
        )

    for unit_id, pod_index in assignment.items():
        unit_pickup[unit_id] = pods[pod_index].id


def drive_unit_next_move(
    drive_unit_id: int,
    state: GraphState,
) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Level 3 strategy:

    - Once per time step, work out the best pod-to-unit matching among
      every idle-capable unit and every unclaimed waiting pod.

    - Use an exact solution when there aren't too many pods and a
      greedy fallback otherwise, preventing multiple units from
      converging on the same pod.

    - A unit heads for whichever is closer right now: its assigned
      pickup or the nearest destination among everything it is
      already carrying.

    - A unit blocked from delivering waits rather than wandering off.

    - An idle unit sitting on a full, capacity-limited dock steps
      aside instead of blocking the next arrival.
    """

    unit = state.get_drive_unit(drive_unit_id)

    if unit is None or unit.in_transit:
        return None

    _update_assignment(state)

    # ---------------------------------------------------------------
    # Potential pickup
    # ---------------------------------------------------------------

    pickup_cost = None
    pickup_step = None

    pickup_pod_id = _assignment_cache["unit_pickup"].get(
        unit.id
    )

    if pickup_pod_id is not None and unit.has_capacity:
        pod = state.get_pod(pickup_pod_id)

        if (
            pod is not None
            and pod.carried_by is None
            and pod.current_node is not None
        ):
            pickup_cost, pickup_step = _nearest_target(
                state,
                unit.current_node,
                {pod.current_node},
            )

    # ---------------------------------------------------------------
    # Potential delivery
    # ---------------------------------------------------------------

    deliver_cost = None
    deliver_step = None

    if unit.carrying:
        destinations = set()

        for pod_id in unit.carrying:
            pod = state.get_pod(pod_id)

            if pod is not None:
                destinations.add(
                    pod.destination_station
                )

        deliver_cost, deliver_step = _nearest_target(
            state,
            unit.current_node,
            destinations,
        )

    # ---------------------------------------------------------------
    # Choose pickup or delivery
    # ---------------------------------------------------------------

    if pickup_step is not None and deliver_step is not None:
        return (
            pickup_step
            if pickup_cost <= deliver_cost
            else deliver_step
        )

    if pickup_step is not None:
        return pickup_step

    if deliver_step is not None:
        return deliver_step

    # Carrying something but currently blocked from every destination
    # (usually a full dock/aisle). Wait instead of wandering away.
    if unit.carrying:
        return None

    # If we're blocking a full capacity-limited node, move aside.
    reposition = _step_off_full_dock(
        state,
        unit,
    )

    if reposition is not None:
        return reposition

    # Otherwise head toward the nearest storage node.
    storage_nodes = {
        node.id
        for node in state.nodes
        if node.node_type == "storage"
    }

    _, storage_step = _nearest_target(
        state,
        unit.current_node,
        storage_nodes,
    )

    return storage_step
