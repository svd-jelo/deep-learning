"""
Helper functions for drawing computation-graph diagrams (forward + backprop)
for deep learning notes, starting with logistic regression.

Usage:
    from comp_graph import CompGraph
    g = CompGraph()
    g.add_node("x", (0, 0))
    g.add_node("w", (0, 2))
    g.add_node("b", (0, 4))
    g.add_node("z", (2, 1), op="z = w*x + b")
    g.add_node("a", (4, 1), op="a = sigma(z)")
    g.add_node("L", (6, 1), op="L(a,y)")
    g.add_edge("x", "z")
    g.add_edge("w", "z")
    g.add_edge("b", "z")
    g.add_edge("z", "a")
    g.add_edge("a", "L")
    g.add_backward_edge("L", "a", label="dL/da")
    g.add_backward_edge("a", "z", label="dL/dz")
    g.add_backward_edge("z", "w", label="dL/dw")
    g.add_backward_edge("z", "b", label="dL/db")
    g.draw("logistic_regression_graph.png")
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patheffects as pe


class CompGraph:
    def __init__(self, node_radius=0.45):
        self.nodes = {}       # name -> (x, y, op_label)
        self.fwd_edges = []   # (src, dst)
        self.bwd_edges = []   # (src, dst, label)
        self.node_radius = node_radius

    def add_node(self, name, pos, op=None, enclose=True):
        """pos: (x, y) tuple. op: optional formula string shown under the node."""
        self.nodes[name] = (pos[0], pos[1], op, enclose)

    def add_edge(self, src, dst):
        self.fwd_edges.append((src, dst))

    def add_backward_edge(self, src, dst, label=None):
        self.bwd_edges.append((src, dst, label))

    def draw(self, save_path=None, **kwargs):
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10,6)))
        r = self.node_radius

        # forward edges (drawn slightly above center, arrow pointing right/down)
        for src, dst in self.fwd_edges:
            self._draw_edge(ax, src, dst,
                            color=kwargs.get("fwd_color","#2c6fbb"),
                            offset=kwargs.get("fwd_offset", 0.2),
                            arrowstyle=kwargs.get("edge_arrowstyle", "-|>"),
                            lw=kwargs.get("edge_lw", 2),)

        # backward edges (drawn slightly below, dashed, opposite direction, labeled)
        for src, dst, label in self.bwd_edges:
            self._draw_edge(ax, src, dst,
                            color=kwargs.get("bwd_color","#d1495b"),
                            offset=kwargs.get("bwd_offset", 0.2),
                            arrowstyle=kwargs.get("edge_arrowstyle", "-|>"),
                            lw=kwargs.get("edge_lw", 2),
                            linestyle="dashed",
                            label=label,
                            label_color=kwargs.get("bwd_color", "#d1495b"),)

        # nodes on top
        for name, (x, y, op, enclose) in self.nodes.items():
            if enclose:
                circ = Circle((x, y), r,
                              facecolor=kwargs.get("node_color", "#eef3fb"),
                              edgecolor="#333333",linewidth=1.6, zorder=3)
                ax.add_patch(circ)
            ax.text(x, y, name, ha="center", va="center",
                    fontsize=kwargs.get("fontsize", 13),
                    fontweight="bold",
                    zorder=4)
            if op:
                ax.text(x, y - r - 0.28, op, ha="center", va="top",
                        fontsize=kwargs.get("fontsize",13) - 3,
                        color="#555555",
                        zorder=4)

        # legend
        ax.plot([], [],
                color=kwargs.get("fwd_color", "#2c6fbb"),
                lw=2,
                label="forward pass")
        ax.plot([], [],
                color=kwargs.get("bwd_color", "#d1495b"),
                lw=1.8,
                linestyle="dashed",
                label="backward pass (gradient)")
        ax.legend(loc="upper left",
                  frameon=False,
                  fontsize=kwargs.get("fontsize", 13) - 3)

        xs = [p[0] for p in self.nodes.values()]
        ys = [p[1] for p in self.nodes.values()]
        pad = 1.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig, ax

    def _draw_edge(self, ax, src, dst, color, offset, arrowstyle, lw,
                    linestyle="solid", label=None, label_color=None):
        x1, y1, _, _ = self.nodes[src]
        x2, y2, _, _ = self.nodes[dst]
        r = self.node_radius

        # shrink to node boundaries + perpendicular offset for fwd/bwd separation
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / dist, dy / dist
        px, py = -uy * offset, ux * offset  # perpendicular offset

        start = (x1 + ux * r + px, y1 + uy * r + py)
        end = (x2 - ux * r + px, y2 - uy * r + py)

        arrow = FancyArrowPatch(start, end, arrowstyle=arrowstyle,
                                 color=color, lw=lw, linestyle=linestyle,
                                 mutation_scale=18, zorder=2)
        ax.add_patch(arrow)

        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my, label, color=label_color or color, fontsize=10,
                     ha="center", va="center", zorder=5,
                     path_effects=[pe.withStroke(linewidth=3, foreground="white")])