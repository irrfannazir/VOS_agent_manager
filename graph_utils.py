from collections import defaultdict, deque

from models import Graph, Node


class GraphValidationError(Exception):
    pass


def validate_graph(graph: Graph) -> None:
    ids = [n.id for n in graph.nodes]

    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise GraphValidationError(f"Duplicate node ids: {sorted(dupes)}")

    id_set = set(ids)
    for node in graph.nodes:
        for dep in node.depends_on:
            if dep not in id_set:
                raise GraphValidationError(
                    f"Node '{node.id}' depends on '{dep}' which does not exist"
                )

    roots = [n for n in graph.nodes if not n.depends_on]
    if not roots:
        raise GraphValidationError("Graph has no root nodes (no starting point)")

    adj = defaultdict(list)
    in_degree = defaultdict(int)
    for node in graph.nodes:
        in_degree[node.id]  # ensure entry exists
        for dep in node.depends_on:
            adj[dep].append(node.id)
            in_degree[node.id] += 1

    queue = deque(n.id for n in graph.nodes if in_degree[n.id] == 0)
    visited = 0
    while queue:
        nid = queue.popleft()
        visited += 1
        for child in adj[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(ids):
        raise GraphValidationError("Graph contains a cycle")


def build_waves(graph: Graph) -> list[list[Node]]:
    validate_graph(graph)

    by_id = {n.id: n for n in graph.nodes}
    adj = defaultdict(list)
    in_degree = {n.id: 0 for n in graph.nodes}
    for node in graph.nodes:
        for dep in node.depends_on:
            adj[dep].append(node.id)
            in_degree[node.id] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    waves: list[list[Node]] = []
    while queue:
        wave_ids: list[str] = []
        next_queue: deque[str] = deque()
        while queue:
            nid = queue.popleft()
            wave_ids.append(nid)
            for child in adj[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        waves.append([by_id[i] for i in wave_ids])
        queue = next_queue
    return waves


def get_sink_nodes(graph: Graph) -> list[Node]:
    referenced = set()
    for node in graph.nodes:
        referenced.update(node.depends_on)
    return [n for n in graph.nodes if n.id not in referenced]
