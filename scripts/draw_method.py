"""Draw original system schematics as editable PPTX and vector PDF/SVG.

Every slide uses native shapes, connectors and text, not a pasted bitmap.
Coordinates are shared with the Matplotlib renderer to keep exports aligned.
The page layouts are conceptual examples, not measured performance claims.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itertools

from build_figures import BLUE, GREEN, INK, ORANGE, PALE, PURPLE, configure

OUT = Path(__file__).resolve().parents[1] / "figures/method"
W, H = 14, 8.0


def rgb(color):
    from matplotlib.colors import to_hex

    return RGBColor.from_string(to_hex(color).lstrip("#"))


class Scene:
    def __init__(self, title, subtitle):
        self.items = []
        self.text(0.4, 0.14, 13.2, 0.4, title, size=22, bold=True)
        self.text(0.4, 0.61, 13.2, 0.3, subtitle, size=11, color=INK)

    def box(self, x, y, w, h, fill="white", edge=BLUE, rounded=True, lw=0.9):
        self.items.append(
            {
                "kind": "box",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "fill": fill,
                "edge": edge,
                "rounded": rounded,
                "lw": lw,
            }
        )

    def text(self, x, y, w, h, text, size=13, bold=False, color=INK, align="left"):
        self.items.append(
            {
                "kind": "text",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "text": text,
                "size": size,
                "bold": bold,
                "color": color,
                "align": align,
            }
        )

    def line(self, points, color=INK, arrow=False, dashed=False, lw=1.2):
        self.items.append(
            {
                "kind": "line",
                "points": points,
                "color": color,
                "arrow": arrow,
                "dashed": dashed,
                "lw": lw,
            }
        )

    def module(self, x, y, w, h, title, color, number):
        self.box(x, y, w, h, PALE[color], color)
        self.box(x + 0.14, y + 0.16, 0.3, 0.3, color, color)
        self.text(
            x + 0.14,
            y + 0.16,
            0.3,
            0.3,
            number,
            size=11,
            bold=True,
            color="white",
            align="center",
        )
        self.text(
            x + 0.55, y + 0.12, w - 0.7, 0.4, title, size=15, bold=True, color=color
        )

    def tiles(self, x, y, n, w=0.3, h=0.32, colors=None, labels=False, gap=0.04):
        for i in range(n):
            color = colors[i] if colors else BLUE
            self.box(x + i * (w + gap), y, w, h, PALE[color], color, False, 0.7)
            if labels:
                self.text(x + i * (w + gap), y, w, h, str(i), size=8, align="center")

    def distribution(self, x, y, color, w=1.1, h=0.4):
        heights = [0.18, 0.43, 0.8, 1, 0.83, 0.5, 0.27, 0.12]
        for j, v in enumerate(heights):
            self.box(
                x + j * w / 8,
                y + h * (1 - v),
                w / 8 - 0.025,
                h * v,
                color,
                color,
                False,
                0.3,
            )

    def tree(self, x, y, color):
        self.line(
            [(x + 0.20, y), (x + 0.20, y + 0.12), (x + 0.05, y + 0.28)], color, lw=0.85
        )
        self.line([(x + 0.20, y + 0.12), (x + 0.35, y + 0.28)], color, lw=0.85)
        for a, b in [(x + 0.2, y), (x + 0.05, y + 0.28), (x + 0.35, y + 0.28)]:
            self.box(a - 0.04, b - 0.04, 0.08, 0.08, color, color, False, 0.4)

    def export(self, name, prs):
        OUT.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, W), ylim=(H, 0))
        ax.axis("off")
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        for p in self.items:
            if p["kind"] == "box":
                x, y, w, h = p["x"], p["y"], p["w"], p["h"]
                if p["rounded"]:
                    patch = FancyBboxPatch(
                        (x, y),
                        w,
                        h,
                        boxstyle="round,pad=0,rounding_size=.08",
                        fc=p["fill"],
                        ec=p["edge"],
                        lw=p["lw"],
                    )
                    shape = slide.shapes.add_shape(
                        MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(x),
                        Inches(y),
                        Inches(w),
                        Inches(h),
                    )
                    shape.adjustments[0] = 0.06
                else:
                    patch = Rectangle(
                        (x, y), w, h, fc=p["fill"], ec=p["edge"], lw=p["lw"]
                    )
                    shape = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
                    )
                ax.add_patch(patch)
                shape.fill.solid()
                shape.fill.fore_color.rgb = rgb(p["fill"])
                shape.line.color.rgb = rgb(p["edge"])
                shape.line.width = Pt(p["lw"])
            elif p["kind"] == "text":
                offset = (
                    0
                    if p["align"] == "left"
                    else p["w"] / 2
                    if p["align"] == "center"
                    else p["w"]
                )
                ax.text(
                    p["x"] + offset,
                    p["y"] + p["h"] / 2,
                    p["text"],
                    fontsize=p["size"],
                    weight="bold" if p["bold"] else "normal",
                    color=p["color"],
                    ha=p["align"],
                    va="center",
                    linespacing=1.3,
                )
                shape = slide.shapes.add_textbox(
                    Inches(p["x"]), Inches(p["y"]), Inches(p["w"]), Inches(p["h"])
                )
                tf = shape.text_frame
                tf.clear()
                tf.word_wrap = False
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = MSO_ANCHOR.MIDDLE
                for i, line in enumerate(p["text"].split("\n")):
                    para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                    para.alignment = {
                        "left": PP_ALIGN.LEFT,
                        "center": PP_ALIGN.CENTER,
                        "right": PP_ALIGN.RIGHT,
                    }[p["align"]]
                    para.space_before = Pt(0)
                    para.space_after = Pt(0)
                    run = para.add_run()
                    run.text = line
                    run.font.name = "Arial"
                    run.font.size = Pt(p["size"])
                    run.font.bold = p["bold"]
                    run.font.color.rgb = rgb(p["color"])
            else:
                pts = p["points"]
                ax.plot(
                    *zip(*pts),
                    color=p["color"],
                    lw=p["lw"],
                    ls="--" if p["dashed"] else "-",
                    solid_capstyle="round",
                )
                if p["arrow"]:
                    a, b = pts[-2:]
                    ax.add_patch(
                        FancyArrowPatch(
                            a,
                            b,
                            arrowstyle="-|>",
                            mutation_scale=10,
                            color=p["color"],
                            lw=p["lw"],
                        )
                    )
                # Explicit line segments and editable triangular arrowhead.
                from pptx.enum.dml import MSO_LINE_DASH_STYLE
                from pptx.enum.shapes import MSO_CONNECTOR

                for a, b in itertools.pairwise(pts):
                    conn = slide.shapes.add_connector(
                        MSO_CONNECTOR.STRAIGHT,
                        Inches(a[0]),
                        Inches(a[1]),
                        Inches(b[0]),
                        Inches(b[1]),
                    )
                    conn.line.color.rgb = rgb(p["color"])
                    conn.line.width = Pt(p["lw"])
                    if p["dashed"]:
                        conn.line.dash_style = MSO_LINE_DASH_STYLE.DASH
                if p["arrow"]:
                    a, b = pts[-2:]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    shape = slide.shapes.add_shape(
                        MSO_SHAPE.ISOSCELES_TRIANGLE,
                        Inches(b[0] - 0.045),
                        Inches(b[1] - 0.05),
                        Inches(0.09),
                        Inches(0.1),
                    )
                    shape.rotation = (
                        90 if dx > 0 else 270 if dx < 0 else 180 if dy > 0 else 0
                    )
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = rgb(p["color"])
                    shape.line.fill.background()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(
                OUT / f"{name}.{ext}", dpi=200, facecolor="white", bbox_inches=None
            )
            if ext == "svg":
                path = OUT / f"{name}.{ext}"
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
        plt.close(fig)
        slide.notes_slide.notes_text_frame.text = "Original Kairos schematic. Conceptual example, not a measured trace. All objects are native editable PowerPoint shapes. Visual grouping references: Continuum Fig. 7 and CacheBlend Fig. 1."


def overview():
    s = Scene(
        "Kairos: from observed history to executable progress",
        "A single ready order, a separate capacity transaction, and layout-aware state continuity",
    )
    s.box(0.4, 1.1, 13.2, 0.72, PALE[BLUE], BLUE)
    s.text(0.6, 1.14, 2, 0.25, "Causal state snapshot", 12, True)
    for x, label in [
        (3.05, "Arrived prompt"),
        (5.55, "Completed history"),
        (8.1, "Actual KV layout"),
        (10.85, "Ready / running / tool"),
    ]:
        s.text(x, 1.26, 2.4, 0.3, label, 11)
    s.module(0.4, 2.25, 3.65, 3.65, "Information", BLUE, "1")
    s.box(0.74, 3.03, 1.55, 0.42, "white", BLUE)
    s.text(0.74, 3.03, 1.55, 0.42, "Text + history", 11, align="center")
    s.line([(2.31, 3.24), (2.50, 3.24)], BLUE, True)
    s.box(2.52, 3.03, 1.25, 0.42, "white", BLUE)
    s.text(2.52, 3.03, 1.25, 0.42, "Encoder", 11, align="center")
    s.distribution(0.75, 3.73, BLUE, 1.1, 0.4)
    s.text(2.03, 3.68, 1.8, 0.5, "Current output\nP(O | history)", 11)
    s.distribution(0.75, 4.52, GREEN, 1.1, 0.4)
    s.text(2.03, 4.47, 1.8, 0.5, "Conditional service\nP(R+ | O, history)", 11)
    s.text(
        0.74,
        5.2,
        3,
        0.45,
        "Infer once per arrived call.\nCondition on observed output survival.",
        10.2,
    )

    s.module(4.5, 2.25, 4.7, 3.65, "Program scheduling", GREEN, "2")
    s.text(4.78, 2.94, 4.1, 0.35, "Geometric J = exp(E[log(t / g)])", 13, True)
    s.text(4.78, 3.39, 1.25, 0.4, "Ready order", 11)
    for i, label in enumerate(["A", "B", "C", "D"]):
        s.box(6.05 + i * 0.65, 3.39, 0.53, 0.4, "white", GREEN)
        s.text(6.05 + i * 0.65, 3.39, 0.53, 0.4, label, 11, True, align="center")
    s.line([(6.1, 3.95), (8.6, 3.95)], GREEN, True)
    s.text(4.78, 3.96, 3.9, 0.25, "Increasing J; cold and returning calls together", 10)
    s.box(4.78, 4.43, 4.12, 1.1, "white", GREEN)
    s.text(4.96, 4.49, 3.75, 0.3, "Independent capacity transaction", 12, True)
    s.text(
        4.96,
        4.89,
        3.75,
        0.46,
        "Native-frontier pages + execution slots\nEarlier claim + eligible optional victims",
        10.6,
    )

    s.module(9.65, 2.25, 3.95, 3.65, "Restore & execute", PURPLE, "4")
    s.text(9.94, 2.99, 1.2, 0.3, "GPU pages", 11, True)
    s.tiles(11.25, 2.99, 5, w=0.30, h=0.3, colors=[BLUE] * 5)
    s.text(9.94, 3.53, 1.2, 0.3, "CPU chunks", 11, True)
    for j in range(3):
        s.box(11.25 + j * 0.53, 3.53, 0.45, 0.34, PALE[GREEN], GREEN, False)
    s.text(
        9.94,
        4.08,
        3.3,
        0.5,
        "Use the joint recoverable prefix.\nCopy missing pages; recompute gaps.",
        11,
    )
    s.box(9.94, 4.95, 3.35, 0.55, "white", PURPLE)
    s.text(
        10.03,
        4.98,
        3.16,
        0.46,
        "Native allocator + vLLM / LMCache",
        11,
        True,
        align="center",
    )

    s.module(0.4, 6.32, 3.65, 1.13, "Program & tool", ORANGE, "•")
    s.text(0.7, 6.88, 3.04, 0.34, "LLM → tool wait → next arrived call", 10.5)
    s.module(4.5, 6.32, 9.1, 1.13, "KV continuity", ORANGE, "3")
    s.text(4.8, 6.92, 3.1, 0.3, "Keep benefit: actual recovery saved", 10.8)
    s.text(
        8.0,
        6.92,
        5.25,
        0.3,
        "Return probability × benefit − pressure × hold time",
        10.8,
    )
    s.line([(4.05, 4.05), (4.5, 4.05)], BLUE, True)
    s.line([(9.2, 4.05), (9.65, 4.05)], GREEN, True)
    s.line([(2.23, 1.82), (2.23, 2.25)], BLUE, True)
    s.line([(6.85, 1.82), (6.85, 2.25)], BLUE, True)
    s.line([(11.6, 1.82), (11.6, 2.25)], PURPLE, True)
    s.line([(6.85, 5.9), (6.85, 6.32)], GREEN, True)
    s.line([(11.6, 6.32), (11.6, 5.9)], ORANGE, True)
    s.line([(4.05, 6.88), (4.5, 6.88)], ORANGE, True)
    s.line(
        [(13.6, 5.35), (13.86, 5.35), (13.86, 1.46), (13.6, 1.46)],
        PURPLE,
        True,
        dashed=True,
    )
    s.text(
        0.4,
        7.63,
        13.2,
        0.25,
        "Feedback contains completed observations only.  Solid: decision / execution flow.  Dashed: completed-state feedback.",
        10,
        align="center",
    )
    return s


def admission():
    s = Scene(
        "Capacity admission without unnecessary state loss",
        "Conceptual example: 16 usable pages, one ready call A requiring 5 pages, and no earlier pending claim",
    )
    s.box(0.4, 1.14, 13.2, 0.62, PALE[GREEN], GREEN)
    s.text(
        0.65,
        1.24,
        12.7,
        0.38,
        "Ready A outranks optional owner C.  Running pages are locked; C is private, idle, and safe to trim.",
        12,
    )

    def row(y, groups, label):
        s.text(0.55, y, 2.4, 0.5, label, 13, True)
        x = 3.0
        for n, color, kind in groups:
            for i in range(n):
                s.box(
                    x + i * 0.59,
                    y,
                    0.53,
                    0.48,
                    PALE[color] if kind != "free" else "white",
                    color,
                    False,
                )
            s.text(
                x,
                y + 0.56,
                n * 0.59 - 0.06,
                0.28,
                f"{kind} ({n})",
                10.5,
                align="center",
            )
            x += n * 0.59

    row(
        2.2,
        [(6, BLUE, "running"), (8, ORANGE, "optional C"), (2, PURPLE, "free")],
        "Current snapshot",
    )
    row(
        3.65,
        [(6, BLUE, "running"), (5, GREEN, "grant A"), (5, PURPLE, "free")],
        "Reference plan",
    )
    s.text(
        3.0,
        3.21,
        9.4,
        0.28,
        "Revoke C's 8 pages, then grant A; the checked plan leaves 5 pages free.",
        11,
    )
    row(
        5.1,
        [(6, BLUE, "running"), (5, GREEN, "grant A"), (5, ORANGE, "retained C")],
        "Safe partial retention",
    )
    s.text(
        3.0,
        4.66,
        9.4,
        0.28,
        "Keep C's useful 5-page prefix; release only its final 3 pages before granting A.",
        11,
    )
    s.line([(1.75, 2.98), (1.75, 3.55)], GREEN, True)
    s.line([(1.75, 4.43), (1.75, 5.0)], GREEN, True)
    s.text(
        0.55,
        6.15,
        12.9,
        0.4,
        "Retention budget U = min(final free − claim credit, minimum commit slack) = min(5 − 0, 5) = 5 pages",
        12,
        True,
    )
    for x, title, body, color in [
        (0.55, "Same execution choice", "A receives the original grant.", GREEN),
        (
            4.95,
            "Preserve capacity obligations",
            "Earlier claims and commit limits hold.",
            BLUE,
        ),
        (
            9.35,
            "Release only safe state",
            "No running, shared, or DMA-held victims.",
            ORANGE,
        ),
    ]:
        s.box(x, 6.88, 4.1, 0.75, PALE[color], color)
        s.text(x + 0.13, 6.94, 3.84, 0.25, title, 11.5, True, color)
        s.text(x + 0.13, 7.27, 3.84, 0.22, body, 10.1)
    return s


def restoration():
    s = Scene(
        "Restore the missing history, preserve the resident pages",
        "Conceptual layout: 16 tokens per GPU page; 16 pages per CPU chunk (256 tokens)",
    )
    s.text(0.5, 1.14, 3.5, 0.32, "Logical history pages", 13, True)
    s.text(0.5, 1.65, 2, 0.34, "GPU at return", 12, True, BLUE)
    s.text(0.5, 2.85, 2, 0.34, "CPU resident", 12, True, GREEN)
    # Range [0, 48) at a readable 0.2 inches per page.
    x0, unit = 3.05, 0.205
    for boundary in [0, 16, 20, 32, 48]:
        s.text(
            x0 + boundary * unit - 0.25,
            1.10,
            0.5,
            0.3,
            str(boundary),
            10.5,
            align="center",
        )
    s.tiles(x0, 1.65, 20, w=0.18, h=0.42, gap=0.025, colors=[BLUE] * 20)
    s.box(x0 + 20 * unit, 1.65, 28 * unit - 0.025, 0.42, "white", PURPLE, False)
    s.text(
        x0 + 20 * unit,
        2.13,
        28 * unit,
        0.28,
        "Not resident on GPU",
        10.5,
        align="center",
    )
    for j in [1, 2]:
        s.box(
            x0 + 16 * j * unit, 2.8, 16 * unit - 0.025, 0.68, PALE[GREEN], GREEN, False
        )
        s.text(
            x0 + 16 * j * unit,
            2.87,
            16 * unit - 0.025,
            0.26,
            f"CPU chunk {j}  [{16 * j}, {16 * (j + 1)})",
            11,
            True,
            GREEN,
            align="center",
        )
        s.tiles(
            x0 + 16 * j * unit + 0.04,
            3.23,
            16,
            w=0.16,
            h=0.14,
            gap=0.04,
            colors=[GREEN] * 16,
        )
    s.box(x0, 2.8, 16 * unit - 0.025, 0.68, "white", BLUE, False)
    s.text(
        x0,
        2.91,
        16 * unit - 0.025,
        0.4,
        "GPU already provides [0, 16)",
        10.6,
        align="center",
    )
    s.text(0.5, 4.12, 2.1, 0.6, "H2D work\nfor this return", 12, True)
    s.box(x0, 4.1, 20 * unit - 0.025, 0.63, PALE[BLUE], BLUE, False)
    s.text(
        x0,
        4.21,
        20 * unit - 0.025,
        0.35,
        "Keep pages [0, 20)",
        11,
        True,
        BLUE,
        align="center",
    )
    s.box(x0 + 20 * unit, 4.1, 12 * unit - 0.025, 0.63, PALE[ORANGE], ORANGE, False)
    s.text(
        x0 + 20 * unit,
        4.15,
        12 * unit - 0.025,
        0.5,
        "Copy 12 missing pages\nfrom boundary chunk",
        10.8,
        True,
        ORANGE,
        align="center",
    )
    s.box(x0 + 32 * unit, 4.1, 16 * unit - 0.025, 0.63, PALE[GREEN], GREEN, False)
    s.text(
        x0 + 32 * unit,
        4.15,
        16 * unit - 0.025,
        0.5,
        "Copy next 16 pages\nwith native batch path",
        10.8,
        True,
        GREEN,
        align="center",
    )
    s.line([(x0 + 26 * unit, 3.48), (x0 + 26 * unit, 4.04)], ORANGE, True)
    s.line([(x0 + 40 * unit, 3.48), (x0 + 40 * unit, 4.04)], GREEN, True)
    s.text(
        x0,
        4.99,
        9.84,
        0.34,
        "Joint recoverable endpoint: page 48; H2D transfers 28 pages, with no copy over [16, 20).",
        11.2,
        True,
    )
    for x, title, body, color in [
        (
            0.5,
            "1  Join coverage",
            "Extend the GPU prefix through\nconsecutive, confirmed CPU chunks.",
            BLUE,
        ),
        (
            4.95,
            "2  Respect page boundaries",
            "Preserve resident GPU pages;\nmap only the missing chunk suffix.",
            ORANGE,
        ),
        (
            9.4,
            "3  Retire transfer references",
            "DMA completion precedes page reuse;\nrecompute beyond any coverage gap.",
            GREEN,
        ),
    ]:
        s.box(x, 5.88, 4.1, 1.34, PALE[color], color)
        s.text(x + 0.17, 6.02, 3.76, 0.35, title, 12, True, color)
        s.text(x + 0.17, 6.49, 3.76, 0.5, body, 10.8)
    s.text(
        0.5,
        7.52,
        12.9,
        0.25,
        "The page strip shows logical history order, not physically contiguous GPU allocation. No transfer overlap or speedup is assumed.",
        10,
        align="center",
    )
    return s


def main():
    configure()
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    for name, scene in [
        ("architecture", overview()),
        ("admission_retention", admission()),
        ("joint_restoration", restoration()),
    ]:
        scene.export(name, prs)
    prs.core_properties.title = "Kairos method figures"
    prs.core_properties.subject = (
        "Editable system architecture, capacity admission and joint-prefix restoration"
    )
    prs.save(OUT / "Kairos_Method_Figures.pptx")
    print(f"Saved 3 editable slides and PDF/PNG/SVG exports to {OUT}")


if __name__ == "__main__":
    main()
