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
import sys
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
        fontname="Helvetica",
        fontsize="11",
        width="1.2",
        height="0.6",
    )
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    # Input nodes (will be placed at left)
    dot.node("inp_atoms", "Atom types\n(z indices)", shape="oval", fillcolor="#B5BCBE", color="#000")
    dot.node(
        "inp_pos",
        "Atomic positions\n(r coordinates)",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000"
    )
    dot.node(
        "inp_solv",
        "Dielectric constant\n(ε)",
        shape="oval",
        fillcolor="#B5BCBE",
        color="#000"
    )
    
    # Connect inputs to their respective processing nodes
    dot.edge("inp_pos", "rbf")
    dot.edge("inp_atoms", "embedding")
    dot.edge("inp_solv", "solv")

    # Embedding + RBF
    dot.node("embedding", "Embedding\n(atom type → vector)", fillcolor="#F6A21C")
    dot.node("rbf", "Radial Basis Functions\n(distance → features)", fillcolor="#F6A21C")
    
    dot.edge("embedding", "backbone")
    dot.edge("rbf", "backbone")

    # Backbone: PaiNN stack (represent as single box with repeat count)
    backbone_label = f"PaiNN Backbone\n({num_layers} layers)"
    dot.node(
        "backbone", backbone_label, shape="rect", fillcolor="#565AA2", fontcolor="#ffffff", fontsize="11"
    )

    # Pooling / Mol Embedding
    pool_label = f"Pooling\n(mean over atoms)\n→ h_mol"
    dot.node("pool", pool_label, fillcolor="#565AA2", fontcolor="#ffffff", fontsize="11")
    dot.edge("backbone", "pool")

    # Lambda head (simple MLP)
    dot.node("lambda", "λmax Head\n(MLP → λ_max)", fillcolor="#F6A21C")
    dot.edge("pool", "lambda")

    # Solvent encoder + Phi head
    dot.node("solv", "Solvent Encoder\n(dielectric → vector)", fillcolor="#F6A21C")
    dot.node("concat", "Concat\n[h_mol, solvent]", fillcolor="#565AA2", fontcolor="#ffffff")
    dot.node("phi", "φ Head\n(MLP → φ)", fillcolor="#F6A21C")

    dot.edge("pool", "concat")
    dot.edge("solv", "concat")
    dot.edge("concat", "phi")

    # Force inputs to be at left (same rank)
    dot.body.append("{ rank = same; inp_atoms; inp_pos; inp_solv; }")
    # Force Lambda and Phi heads to be vertically aligned (same rank)
    dot.body.append("{ rank = same; lambda; phi; }")

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
        fontname="Helvetica",
        fontsize="11",
    )
    
    legend_label = textwrap.dedent(f"""
        <b>MANA Architecture Legend</b>
        
        <b>Inputs:</b>
        • Atom types (z indices)
        • Atomic positions (r coordinates)
        • Dielectric constant (ε)
        
        <b>Backbone:</b>
        • {num_layers} × PaiNN layers
        • Equivariant message passing
        
        <b>Output Heads:</b>
        • λ_max: Absorption maximum wavelength
        • φ: Singlet oxygen quantum yield
        
        <b>Pooling:</b>
        • Sum or mean over {num_nodes} atoms
        • Produces molecular embedding (h_mol)
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