from graph_utils import get_sink_nodes
from models import Graph


class IntegratorAgent:
    def integrate(self, graph: Graph) -> str:
        sinks = get_sink_nodes(graph)

        # The kernel appends a synthesis node wired to every leaf, so when one
        # is present it already contains the complete answer. Concatenating the
        # other sinks after it would append the raw research the synthesis node
        # was built to organise.
        synthesis = [n for n in sinks if n.capability == "synthesis" and n.output]
        if synthesis:
            print(
                f"[integrator-agent] returning synthesis node "
                f"'{synthesis[-1].id}' output"
            )
            return synthesis[-1].output

        print(f"[integrator-agent] combining outputs from {len(sinks)} sink node(s)")

        if len(sinks) == 1:
            result = sinks[0].output
        else:
            parts = []
            for node in sinks:
                parts.append(
                    f"From node '{node.id}' ({node.description}):\n"
                    f"{node.output}"
                )
            result = "\n\n".join(parts)

        print("[integrator-agent] done")
        return result
