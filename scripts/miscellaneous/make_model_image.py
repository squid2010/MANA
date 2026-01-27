#!/usr/bin/env python3
"""
Simple, clean Graphviz diagram generator for the MANA model architecture.

Purpose:
- Produce a compact, poster-friendly architecture diagram (PNG / PDF / SVG).
- Intentionally *not* an autograd graph — this is a conceptual block diagram
  showing the main components and data flow for presentation (ISEF poster).

Usage:
    python make_model_image.py --out mana_arch.png --format png --nodes 6 --layers 4

Output:
- A rendered graph file (default: `mana_arch.png`) in the chosen format.
- A separate legend file (default: `mana_arch_legend.png`).
- If Graphviz Python bindings are not available, the DOT source will be written
  to `mana_arch.dot` instead so you can render it manually with Graphviz tools.

Notes:
- Designed to be compact and readable on a poster: left-to-right flow, few boxes,
  clear labels for Embedding, RBF, PaiNN backbone (stacked), Pooling and Heads.
"""

import argparse
import textwrap
from pathlib import Path

try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False


def build_arch_graph(num_layers: int = 4, num_nodes: int = 6) -> "Digraph":
    """
    Construct a simplified Graphviz Digraph describing the MANA architecture.

    - num_layers is used only for labeling (displayed as 'PaiNN x N').
    - num_nodes is only used to annotate pooling (e.g., 'avg over N atoms').
    """
    dot = Digraph(name="MANA", comment="MANA model architecture", format="png")
    # Presentation/style attributes - LR rankdir for horizontal layout
    dot.attr(rankdir="LR", splines="ortho", ranksep="0.5", nodesep="0.3")
    dot.attr(
        "node",
        shape="rect",
        style="rounded,filled",
        fontname="Gill Sans MT",
        fontsize="11",
        width="1.2",
        height="0.6",
    )
    dot.attr("edge", fontname="Gill Sans MT", fontsize="10")

    # Solute Input nodes
    dot.node(
        "inp_atoms_solute",
        "Solute\nAtom types",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000",
    )
    dot.node(
        "inp_pos_solute",
        "Solute\nPositions",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000",
    )

    # Solvent Input nodes
    dot.node(
        "inp_atoms_solvent",
        "Solvent\nAtom types",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000",
    )
    dot.node(
        "inp_pos_solvent",
        "Solvent\nPositions",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000",
    )

    # Solute processing path
    dot.edge("inp_pos_solute", "rbf_solute")
    dot.edge("inp_atoms_solute", "embedding_solute")

    dot.node("embedding_solute", "Embedding\n(atom type → vector)", fillcolor="#F6A21C")
    dot.node("rbf_solute", "RBF\n(distance → features)", fillcolor="#F6A21C")

    dot.edge("embedding_solute", "backbone_solute")
    dot.edge("rbf_solute", "backbone_solute")

    backbone_label = f"PaiNN Backbone\n({num_layers} layers)"
    dot.node(
        "backbone_solute",
        backbone_label,
        shape="rect",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )

    pool_label = "Pooling\n(mean over atoms)"
    dot.node(
        "pool_solute",
        pool_label,
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )
    dot.edge("backbone_solute", "pool_solute")

    dot.node(
        "norm_solute",
        "Layer Norm",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )
    dot.edge("pool_solute", "norm_solute")

    # Solvent processing path (parallel)
    dot.edge("inp_pos_solvent", "rbf_solvent")
    dot.edge("inp_atoms_solvent", "embedding_solvent")

    dot.node(
        "embedding_solvent", "Embedding\n(atom type → vector)", fillcolor="#F6A21C"
    )
    dot.node("rbf_solvent", "RBF\n(distance → features)", fillcolor="#F6A21C")

    dot.edge("embedding_solvent", "backbone_solvent")
    dot.edge("rbf_solvent", "backbone_solvent")

    dot.node(
        "backbone_solvent",
        backbone_label,
        shape="rect",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )

    dot.node(
        "pool_solvent",
        pool_label,
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )
    dot.edge("backbone_solvent", "pool_solvent")

    dot.node(
        "norm_solvent",
        "Layer Norm",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )
    dot.edge("pool_solvent", "norm_solvent")

    # Interaction and concatenation
    dot.node(
        "interaction",
        "Element-wise\nProduct",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
        fontsize="11",
    )
    dot.edge("norm_solute", "interaction")
    dot.edge("norm_solvent", "interaction")

    dot.node(
        "concat",
        "Concat\n[h_mol, h_solv, h_mol * h_solv]",
        fillcolor="#565AA2",
        fontcolor="#ffffff",
    )
    dot.edge("norm_solute", "concat")
    dot.edge("norm_solvent", "concat")
    dot.edge("interaction", "concat")

    # Output heads
    dot.node("lambda", "λmax Head\n(MLP → λ_max)", fillcolor="#F6A21C")
    dot.node("phi", "φ Head\n(MLP → φ)", fillcolor="#F6A21C")

    dot.edge("concat", "lambda")
    dot.edge("concat", "phi")

    # Add dashed clusters with numbered section labels for poster-style separation
    # Clusters group nodes; labels are short (no parenthetical text)
    with dot.subgraph(name="cluster_1") as c:
        c.attr(
            style="dashed",
            color="gray",
            label="1) Graph Construction",
            fontsize="16",
            fontname="Gill Sans MT",
            labelloc="t",
        )
        c.node("inp_atoms_solute")
        c.node("inp_pos_solute")
        c.node("inp_atoms_solvent")
        c.node("inp_pos_solvent")
        c.node("embedding_solute")
        c.node("rbf_solute")
        c.node("embedding_solvent")
        c.node("rbf_solvent")

    with dot.subgraph(name="cluster_2") as c:
        c.attr(
            style="dashed",
            color="gray",
            label="2) Equivariant Message Passing",
            fontsize="16",
            fontname="Gill Sans MT",
            labelloc="t",
        )
        c.node("backbone_solute")
        c.node("backbone_solvent")
        c.node("pool_solute")
        c.node("pool_solvent")
        c.node("norm_solute")
        c.node("norm_solvent")

    with dot.subgraph(name="cluster_3") as c:
        c.attr(
            style="dashed",
            color="gray",
            label="3) Context-aware Mechanism",
            fontsize="16",
            fontname="Gill Sans MT",
            labelloc="t",
            margin="15",
        )
        c.node("interaction")
        c.node("concat")

    # Add invisible edges from the Layer Norms to the interaction node to push
    # the context-aware nodes down (these are invisible so they only affect layout)
    # Increased minlen to move the cluster border and its label further away from the wires.
    dot.edge("norm_solute", "interaction", style="invis", minlen="3")
    dot.edge("norm_solvent", "interaction", style="invis", minlen="3")

    with dot.subgraph(name="cluster_4") as c:
        c.attr(
            style="dashed",
            color="gray",
            label="4) Prediction Heads",
            fontsize="16",
            fontname="Gill Sans MT",
            labelloc="t",
        )
        c.node("lambda")
        c.node("phi")

    # Note: removed explicit rank constraints so clusters can contain nodes cleanly.
    # Layout/ranks can be adjusted separately if desired.

    return dot


def build_legend_graph(num_layers: int = 4, num_nodes: int = 6) -> "Digraph":
    """
    Build a separate legend/info diagram.
    """
    dot = Digraph(name="MANA_Legend", comment="MANA model legend", format="png")
    dot.attr(rankdir="TB")
    dot.attr(
        "node",
        shape="plaintext",
        fontname="Gill Sans MT",
        fontsize="11",
    )

    legend_label = textwrap.dedent(f"""
        <b>MANA Architecture Legend</b>

        <b>Dual-Graph Processing:</b>
        • Separate PaiNN backbones for solute and solvent
        • Each processes atom types and positions independently
        • Shared embedding and RBF layers

        <b>Inputs (per graph):</b>
        • Atom types (z indices)
        • Atomic positions (r coordinates)

        <b>Backbone:</b>
        • {num_layers} × PaiNN layers per graph
        • E(3)-equivariant message passing
        • Mean pooling over atoms

        <b>Combination:</b>
        • Layer normalization of embeddings
        • Concatenate: [h_mol, h_solv, h_mol * h_solv]
        • Combined dimension: 3 × hidden_dim

        <b>Output Heads:</b>
        • λ_max: Absorption maximum wavelength (Huber loss)
        • φ: Singlet oxygen quantum yield (Huber loss, Sigmoid activation)
    """).strip()

    dot.node("legend", f"<{legend_label}>", shape="plaintext")

    return dot


def write_dot_and_render(dot: "Digraph", out_path: Path, fmt: str):
    """
    Try to render using Graphviz Python bindings. If not available, write DOT source
    and instruct the user to run Graphviz manually.
    """
    out_path = out_path.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if GRAPHVIZ_AVAILABLE:
        # Set format and render. graphviz will create files like <n>.<fmt>
        dot.format = fmt
        # `render()` writes files and returns path to source; we control filename below.
        # Use out_path.stem as filename, and out_dir as directory.
        filename = out_path.stem
        try:
            rendered = dot.render(
                filename=filename, directory=str(out_dir), cleanup=True
            )
            # graphviz.render returns the path to the source file; append extension for image
            final_path = out_dir / f"{filename}.{fmt}"
            if final_path.exists():
                print(f"Saved architecture diagram: {final_path}")
            else:
                # Fallback message
                print(f"Rendered, but expected output not found at: {final_path}")
                print(f"Graphviz render returned: {rendered}")
        except Exception as e:
            print("Failed to render graph with graphviz Python bindings.")
            print("Falling back to writing DOT source (you can render manually).")
            dot_path = out_dir / f"{out_path.stem}.dot"
            dot.save(str(dot_path))
            print(f"Wrote DOT source to: {dot_path}")
            print("To render manually, run e.g.:")
            print(
                f"  dot -T{fmt} {dot_path} -o {out_dir / (out_path.stem + '.' + fmt)}"
            )
            print(f"Error details: {e}")
    else:
        # Write DOT source for manual rendering
        dot_path = out_dir / f"{out_path.stem}.dot"
        dot.save(str(dot_path))
        print("Graphviz python package not available in this environment.")
        print(f"Wrote DOT source to: {dot_path}")
        print("To render manually, run e.g.:")
        print(f"  dot -T{fmt} {dot_path} -o {out_dir / (out_path.stem + '.' + fmt)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a simplified MANA architecture diagram (poster-friendly)."
    )
    p.add_argument(
        "--out",
        "-o",
        default="mana_arch.png",
        help="Output filename (default: mana_arch.png). Extension is ignored; use --format to set.",
    )
    p.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (png/pdf/svg).",
    )
    p.add_argument(
        "--layers",
        "-l",
        type=int,
        default=4,
        help="Number of PaiNN layers (display only).",
    )
    p.add_argument(
        "--nodes",
        "-n",
        type=int,
        default=6,
        help="Illustrative atom count used in pooling label (display only).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    fmt = args.format

    # Generate main architecture diagram
    dot = build_arch_graph(num_layers=args.layers, num_nodes=args.nodes)
    write_dot_and_render(dot, out_path, fmt)

    # Generate legend diagram
    legend_path = out_path.parent / f"{out_path.stem}_legend{out_path.suffix}"
    legend_dot = build_legend_graph(num_layers=args.layers, num_nodes=args.nodes)
    write_dot_and_render(legend_dot, legend_path, fmt)


if __name__ == "__main__":
    main()
