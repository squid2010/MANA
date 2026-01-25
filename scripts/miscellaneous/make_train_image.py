#!/usr/bin/env python3
"""
Graphviz diagram generator for the MANA Transfer Learning Protocol (Vertical Layout).

Purpose:
- Visualizes the Two-Phase training strategy in a vertical flow.
- Highlights data scarcity vs abundance.
- Shows "Training" vs "Frozen" states.
- Uses 'Gill Sans MT' font.
- Inputs are Ovals (instead of Cylinders).

Usage:
    python visualize_training_vertical.py --out mana_training_vertical.png --format png
"""

import argparse
import textwrap
from pathlib import Path

try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False


# --- Visual Style Configuration ---
COLOR_TRAIN = "#F6A21C"      # Orange/Gold for active training
COLOR_FROZEN = "#E0E0E0"     # Light Gray for frozen
COLOR_INPUT = "#B5BCBE"      # Grey for Data Inputs
COLOR_BG = "#FFFFFF"         # White background
FONT_NAME = "Gill Sans MT"   # Font

def build_training_graph() -> "Digraph":
    dot = Digraph(name="MANA_Training", comment="Two-Phase Transfer Learning")
    
    # Top-to-Bottom flow
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.8")
    dot.attr(compound="true")

    # Global Node Style
    dot.attr(
        "node",
        shape="rect",
        style="rounded,filled",
        fontname=FONT_NAME,
        fontsize="12",
        penwidth="1.5",
        width="2.5",
        height="0.6"
    )
    dot.attr("edge", fontname=FONT_NAME, fontsize="10")

    # =================================================================
    # PHASE 1: Pre-training (Top)
    # =================================================================
    with dot.subgraph(name="cluster_phase1") as c:
        c.attr(
            label="PHASE 1: Representation Learning", 
            fontname=f"{FONT_NAME} Bold", 
            fontsize="14", 
            style="dashed", 
            color="#666666",
            margin="20"
        )
        
        # Data Node (Changed from cylinder to oval)
        c.node(
            "data1", 
            "Absorption Data (λ_max)\n~17,000 Points", 
            shape="oval", 
            fillcolor=COLOR_INPUT, 
            fontcolor="black",
            height="0.8"
        )
        
        # Backbone
        c.node(
            "bb1", 
            "Backbone\n(Training)", 
            fillcolor=COLOR_TRAIN,
            fontcolor="black"
        )
        
        # Head
        c.node(
            "head1", 
            "λ Head\n(Training)", 
            fillcolor=COLOR_TRAIN,
            fontcolor="black"
        )

        # Edges
        c.edge("data1", "bb1")
        c.edge("bb1", "head1")

    # =================================================================
    # PHASE 2: Fine-tuning (Bottom)
    # =================================================================
    with dot.subgraph(name="cluster_phase2") as c:
        c.attr(
            label="PHASE 2: Transfer & Fine-tuning", 
            fontname=f"{FONT_NAME} Bold", 
            fontsize="14", 
            style="dashed", 
            color="#666666",
            margin="20"
        )
        
        # Data Node (Changed from cylinder to oval)
        c.node(
            "data2", 
            "Singlet Oxygen Data (Φ_Δ)\n~1,100 Points", 
            shape="oval", 
            fillcolor=COLOR_INPUT, 
            fontcolor="black",
            height="0.8"
        )
        
        # Backbone (Frozen)
        c.node(
            "bb2", 
            "Backbone\n(FROZEN)", 
            fillcolor=COLOR_FROZEN,
            fontcolor="#555555",
            style="rounded,filled,dashed"
        )
        
        # Head
        c.node(
            "head2", 
            "Φ Head\n(Training)", 
            fillcolor=COLOR_TRAIN,
            fontcolor="black"
        )

        # Edges
        c.edge("data2", "bb2")
        c.edge("bb2", "head2")

    # =================================================================
    # TRANSFER ARROW
    # =================================================================
    # Connects Phase 1 Backbone to Phase 2 Backbone
    dot.edge(
        "bb1", "bb2", 
        xlabel=" Transfer Weights ", 
        style="bold", 
        color="#333333", 
        arrowhead="vee",
    )

    return dot

def build_legend_graph() -> "Digraph":
    dot = Digraph(name="MANA_Training_Legend", comment="Legend")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="plaintext", fontname=FONT_NAME, fontsize="11")
    
    legend_text = textwrap.dedent(f"""
        <b>Training Protocol Legend</b>
        
        <b>Strategy:</b>
        • Phase 1: Learn molecular grammar from abundant λ data.
        • Phase 2: Transfer grammar to scarce Φ data (Backbone Frozen).
        
        <b>Color Code:</b>
        • <font color="{COLOR_TRAIN}"><b>Orange:</b></font> Active Training (Gradients ON)
        • <font color="{COLOR_FROZEN}"><b>Light Gray:</b></font> Frozen Parameters (Gradients OFF)
        • <font color="{COLOR_INPUT}"><b>Grey:</b></font> Dataset Source
    """).strip()
    
    dot.node("legend", f"<{legend_text}>")
    return dot

def render(dot, filename, out_dir, fmt):
    if GRAPHVIZ_AVAILABLE:
        dot.format = fmt
        try:
            dot.render(filename=filename, directory=str(out_dir), cleanup=True)
            print(f"✓ Saved: {out_dir / filename}.{fmt}")
        except Exception as e:
            print(f"⚠ Failed to render (Graphviz Python error): {e}")
            dot.save(str(out_dir / f"{filename}.dot"))
    else:
        out_file = out_dir / f"{filename}.dot"
        dot.save(str(out_file))
        print(f"⚠ Graphviz not found. Saved DOT source: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="img/mana_training_vertical", help="Output filename base")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    out_dir = Path(".")
    
    # Main Diagram
    dot = build_training_graph()
    render(dot, args.out, out_dir, args.format)
    
    # Legend
    legend = build_legend_graph()
    render(legend, f"{args.out}_legend", out_dir, args.format)

if __name__ == "__main__":
    main()