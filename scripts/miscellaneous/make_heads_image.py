#!/usr/bin/env python3
"""
Graphviz diagram generator specifically for the MANA output heads.

Purpose:
- Visualize the internal architecture of the LambdaMaxHead and PhiDeltaHead.
- Highlights differences between the heads (e.g., LayerNorm and Sigmoid in PhiHead).
- Produces a poster-friendly block diagram (PNG / PDF / SVG).

Usage:
    python visualize_heads.py --out mana_heads.png --format png

Output:
- A rendered graph file (default: `mana_heads.png`).
- A separate legend file (default: `mana_heads_legend.png`).
"""

import argparse
import sys
import textwrap
from pathlib import Path

try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False


def build_heads_graph() -> "Digraph":
    """
    Construct a Graphviz Digraph describing the two head architectures in MANA.
    """
    dot = Digraph(name="MANA_Heads", comment="MANA Output Heads Architecture")
    
    # Global attributes for a clean, vertical flow
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.4")
    dot.attr(compound="true") # Allow edges between clusters

    # Default node styling
    dot.attr(
        "node",
        shape="rect",
        style="rounded,filled",
        fontname="Gill Sans MT",
        fontsize="12",
        penwidth="1.5",
        width="2.5",
        height="0.6",
    )
    dot.attr("edge", fontname="Gill Sans MT", fontsize="10", arrowsize="0.8")

    # Colors
    COLOR_INPUT = "#B5BCBE"   # Gray for inputs
    COLOR_LINEAR = "#565AA2"  # Purple for Linear layers
    COLOR_ACT = "#F6A21C"     # Orange for Activations/Norms
    COLOR_OUT = "#E0E0E0"     # Light gray for final output

    # ---------------------------------------------------------
    # SUBGRAPH: LambdaMaxHead
    # ---------------------------------------------------------
    with dot.subgraph(name="cluster_lambda") as c:
        c.attr(label="LambdaMaxHead\n(Predicts Absorption λ)", fontname="Gill Sans MT-Bold", fontsize="14", style="dashed", color="#888888")
        
        # Nodes
        c.node("l_in", "Input Embedding\n(Dimension: D)", shape="oval", fillcolor=COLOR_INPUT)
        
        c.node("l_lin1", "Linear\n(D ➝ D)", fillcolor=COLOR_LINEAR, fontcolor="white")
        c.node("l_silu1", "SiLU", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        c.node("l_drop", "Dropout (0.1)", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        
        c.node("l_lin2", "Linear\n(D ➝ D/2)", fillcolor=COLOR_LINEAR, fontcolor="white")
        c.node("l_silu2", "SiLU", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        
        c.node("l_lin3", "Linear\n(D/2 ➝ 1)", fillcolor=COLOR_LINEAR, fontcolor="white")
        
        c.node("l_out", "Output: λ_max\n(Scalar)", shape="doubleoctagon", fillcolor=COLOR_OUT)

        # Edges
        c.edge("l_in", "l_lin1")
        c.edge("l_lin1", "l_silu1")
        c.edge("l_silu1", "l_drop")
        c.edge("l_drop", "l_lin2")
        c.edge("l_lin2", "l_silu2")
        c.edge("l_silu2", "l_lin3")
        c.edge("l_lin3", "l_out")

    # ---------------------------------------------------------
    # SUBGRAPH: PhiDeltaHead
    # ---------------------------------------------------------
    with dot.subgraph(name="cluster_phi") as c:
        c.attr(label="PhiDeltaHead\n(Predicts Singlet Oxygen φ)", fontname="Gill Sans MT-Bold", fontsize="14", style="dashed", color="#888888")
        
        # Nodes
        c.node("p_in", "Input Embedding\n(Dimension: D)", shape="oval", fillcolor=COLOR_INPUT)
        
        c.node("p_lin1", "Linear\n(D ➝ D)", fillcolor=COLOR_LINEAR, fontcolor="white")
        c.node("p_silu1", "SiLU", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        
        # Distinctive feature: LayerNorm
        c.node("p_ln", "Layer Norm", shape="box", style="filled,dashed", fillcolor=COLOR_ACT, width="2.0")
        
        c.node("p_drop", "Dropout (0.1)", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        
        c.node("p_lin2", "Linear\n(D ➝ D/2)", fillcolor=COLOR_LINEAR, fontcolor="white")
        c.node("p_silu2", "SiLU", shape="oval", fillcolor=COLOR_ACT, width="1.5", height="0.4")
        
        c.node("p_lin3", "Linear\n(D/2 ➝ 1)", fillcolor=COLOR_LINEAR, fontcolor="white")
        
        # Distinctive feature: Sigmoid
        c.node("p_sig", "Sigmoid\n(Bound 0-1)", shape="component", fillcolor=COLOR_ACT)
        
        c.node("p_out", "Output: φ_Δ\n(Scalar 0.0 - 1.0)", shape="doubleoctagon", fillcolor=COLOR_OUT)

        # Edges
        c.edge("p_in", "p_lin1")
        c.edge("p_lin1", "p_silu1")
        c.edge("p_silu1", "p_ln")
        c.edge("p_ln", "p_drop")
        c.edge("p_drop", "p_lin2")
        c.edge("p_lin2", "p_silu2")
        c.edge("p_silu2", "p_lin3")
        c.edge("p_lin3", "p_sig")
        c.edge("p_sig", "p_out")

    return dot


def build_legend_graph() -> "Digraph":
    """
    Build a text-based legend explaining the operations in the heads.
    """
    dot = Digraph(name="MANA_Heads_Legend", comment="MANA Heads Legend", format="png")
    dot.attr(rankdir="TB")
    dot.attr(
        "node",
        shape="plaintext",
        fontname="Gill Sans MT",
        fontsize="11",
    )
    
    legend_label = textwrap.dedent("""
        <b>MANA Heads Legend</b>
        
        <b>Input (Dimension D):</b>
        • Concatenated vector from backbone
        • D = 3 × hidden_dim (solute + solvent + interaction)
        
        <b>Common Components:</b>
        • <b>Linear:</b> Fully connected transformation
        • <b>SiLU:</b> Sigmoid Linear Unit activation (x * σ(x))
        • <b>Dropout:</b> Regularization to prevent overfitting
        
        <b>PhiDeltaHead Specifics:</b>
        • <b>Layer Norm:</b> Normalizes features (added for stability in φ prediction)
        • <b>Sigmoid:</b> Forces output between 0 and 1 (probability/yield)
        • <b>Bias Init:</b> Final layer bias init to -2.0 (starts predictions low)
    """).strip()
    
    dot.node("legend", f"<{legend_label}>", shape="plaintext")
    
    return dot


def write_dot_and_render(dot: "Digraph", out_path: Path, fmt: str):
    """
    Render utilizing Graphviz or fallback to DOT source file.
    """
    out_path = out_path.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if GRAPHVIZ_AVAILABLE:
        dot.format = fmt
        filename = out_path.stem
        try:
            rendered = dot.render(
                filename=filename, directory=str(out_dir), cleanup=True
            )
            final_path = out_dir / f"{filename}.{fmt}"
            if final_path.exists():
                print(f"Saved diagram: {final_path}")
            else:
                print(f"Rendered, but expected output not found at: {final_path}")
        except Exception as e:
            print("Failed to render graph with graphviz Python bindings.")
            dot_path = out_dir / f"{out_path.stem}.dot"
            dot.save(str(dot_path))
            print(f"Wrote DOT source to: {dot_path}")
    else:
        dot_path = out_dir / f"{out_path.stem}.dot"
        dot.save(str(dot_path))
        print("Graphviz not available. Wrote DOT source for manual rendering.")
        print(f"  dot -T{fmt} {dot_path} -o {out_dir / (out_path.stem + '.' + fmt)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MANA Head architecture diagrams."
    )
    p.add_argument(
        "--out",
        "-o",
        default="img/mana_heads.png",
        help="Output filename (default: mana_heads.png).",
    )
    p.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (png/pdf/svg).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    fmt = args.format

    # Generate main architecture diagram
    dot = build_heads_graph()
    write_dot_and_render(dot, out_path, fmt)
    
    # Generate legend diagram
    legend_path = out_path.parent / f"{out_path.stem}_legend{out_path.suffix}"
    legend_dot = build_legend_graph()
    write_dot_and_render(legend_dot, legend_path, fmt)


if __name__ == "__main__":
    main()