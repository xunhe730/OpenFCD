"""Generate two custom figures for the OpenFCD group-meeting deck:
   1) FCD refraction-geometry schematic with η(x) wave + ray bending
   2) Three-stage pipeline architecture diagram (Preprocess→Compute→Postprocess
      + GUI/CLI parity)
Run from project root; outputs into /sessions/.../mnt/outputs/.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# CJK font setup
from matplotlib import font_manager as _fm
for _p in ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",):
    try:
        _fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon
from matplotlib.lines import Line2D

OUT = "/sessions/tender-focused-hamilton/mnt/outputs"
os.makedirs(OUT, exist_ok=True)

# ---------- Slide 2 figure : FCD refraction geometry ----------
def fig_refraction():
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 5.8)
    ax.set_aspect("equal"); ax.axis("off")

    # checkerboard at the bottom (pattern below window)
    n = 16
    for i in range(n):
        for j in range(2):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((i*0.6+0.4, 0.05+j*0.18), 0.6, 0.18,
                                        facecolor="#222", edgecolor="none"))
            else:
                ax.add_patch(Rectangle((i*0.6+0.4, 0.05+j*0.18), 0.6, 0.18,
                                        facecolor="#eee", edgecolor="#888", lw=0.3))
    ax.text(5.0, -0.05, "棋盘格图案 (pattern)", ha="center", va="top",
            fontsize=9.5, color="#444")

    # glass window (thin)
    ax.add_patch(Rectangle((0.4, 0.55), 9.6, 0.20,
                           facecolor="#cfe6f5", edgecolor="#5b8bb0", lw=0.6))
    ax.text(10.05, 0.65, "玻璃窗", fontsize=8, color="#345", va="center")

    # water layer
    ax.add_patch(Rectangle((0.4, 0.75), 9.6, 2.6,
                           facecolor="#dff1ff", edgecolor="none", alpha=0.7))
    ax.text(10.05, 2.05, "水层 (n=1.33)", fontsize=8, color="#256", va="center")

    # free surface η(x) — small wave
    x = np.linspace(0.4, 10.0, 400)
    eta = 0.32*np.sin(2*np.pi*(x-0.4)/3.2) * np.exp(-(x-5)**2/40)
    ax.plot(x, 3.35 + eta, color="#1f4ea0", lw=2.0, zorder=5)
    ax.fill_between(x, 0.75, 3.35+eta, color="#dff1ff", alpha=0.55, zorder=1)
    ax.text(0.45, 3.85, r"自由液面 $\eta(x,y,t)$", fontsize=10, color="#1f4ea0")

    # camera (top)
    ax.add_patch(FancyBboxPatch((4.3, 4.9), 1.4, 0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#f3f3f3", edgecolor="#333", lw=1.0))
    ax.add_patch(Polygon([[5.0, 4.9],[4.6, 4.55],[5.4, 4.55]],
                         closed=True, facecolor="#333", edgecolor="#333"))
    ax.text(5.0, 5.25, "相机", ha="center", va="center", fontsize=9, color="#222")

    # rays: a straight reference ray and a deflected ray
    cam = np.array([5.0, 4.55])
    # reference straight ray (no surface deflection)
    p_ref = np.array([3.4, 0.3])
    ax.plot([cam[0], p_ref[0]], [cam[1], p_ref[1]],
            color="#888", lw=1.0, ls="--", zorder=4)
    ax.text(3.4, 0.55, "参考光线", ha="center", fontsize=8, color="#666")

    # deflected ray: bends at free surface where slope is non-zero
    surf_x = 6.6
    surf_y = 3.35 + 0.32*np.sin(2*np.pi*(surf_x-0.4)/3.2)*np.exp(-(surf_x-5)**2/40)
    p_def_bottom = np.array([7.1, 0.3])
    # camera -> surface (above water, straight)
    ax.plot([cam[0], surf_x], [cam[1], surf_y],
            color="#b00", lw=1.5, zorder=4)
    # surface -> pattern (refracted, slightly tilted)
    ax.plot([surf_x, p_def_bottom[0]], [surf_y, p_def_bottom[1]],
            color="#b00", lw=1.5, zorder=4)
    ax.plot(p_def_bottom[0], p_def_bottom[1], "o", color="#b00",
            markersize=5, zorder=6)
    ax.plot(p_ref[0], p_ref[1], "o", color="#666", markersize=5, zorder=6)

    # displacement arrow at pattern level
    arr = FancyArrowPatch((p_ref[0], 0.3), (p_def_bottom[0], 0.3),
                          arrowstyle="->", mutation_scale=14,
                          color="#b00", lw=1.4)
    ax.add_patch(arr)
    ax.text((p_ref[0]+p_def_bottom[0])/2, 0.05,
            r"像素位移 $\mathbf{u}(x,y)$", ha="center", fontsize=10, color="#b00")

    # surface slope indicator
    ax.annotate("", xy=(surf_x+0.5, surf_y+0.18),
                xytext=(surf_x-0.5, surf_y-0.18),
                arrowprops=dict(arrowstyle="-|>", color="#1f4ea0", lw=1.2))
    ax.text(surf_x+0.55, surf_y+0.30, r"$\nabla\eta$",
            fontsize=11, color="#1f4ea0")

    # h_p^eff bracket
    ax.annotate("", xy=(0.15, 3.35), xytext=(0.15, 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color="#444", lw=0.9))
    ax.text(-0.05, 1.95, r"$h_p^{\mathrm{eff}}$", fontsize=11, color="#222",
            ha="right", va="center")

    out = os.path.join(OUT, "diag_refraction.png")
    plt.tight_layout(pad=0.2)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------- Slide 3 figure : pipeline architecture ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")

    # entry sources (GUI / CLI)
    def src_box(x, y, w, h, label, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=color, edgecolor="#333", lw=1.0))
        ax.text(x+w/2, y+h-0.30, label, ha="center", va="top",
                fontsize=11, weight="bold", color="#222")
        ax.text(x+w/2, y+0.18, sub, ha="center", va="bottom",
                fontsize=8.5, color="#444")

    src_box(0.15, 3.7, 1.9, 1.2, "GUI",
            "PyQt6 三栏\nSimTree·Scene·Props", "#fde7e7")
    src_box(0.15, 1.2, 1.9, 1.2, "CLI",
            "Typer\nopenfcd run/new/replay", "#e7eefd")
    # parity note
    ax.annotate("", xy=(2.05, 3.05), xytext=(2.05, 3.05),
                arrowprops=dict(arrowstyle="-"))
    ax.add_patch(FancyBboxPatch((0.25, 0.25), 1.7, 0.6,
        boxstyle="round,pad=0.01,rounding_size=0.07",
        facecolor="#fff4d6", edgecolor="#b08800", lw=0.9))
    ax.text(1.10, 0.55, "byte-equal\nresults.h5",
            ha="center", va="center", fontsize=8.5, color="#7a5800")

    # central pipeline (3 stages)
    stages = [
        ("Preprocess",
         "scan_frames\nROI / Polygon mask\n参考帧", "#eaf5ea"),
        ("Compute",
         "FCD: carriers · phase\n位移 u(x,y)\nPoisson 反演 → η", "#e6efff"),
        ("Postprocess",
         "η 渲染 / RMS / 波长\n图像导出 · 报告", "#f4e9fb"),
    ]
    sx0, sy, sw, sh, gap = 2.7, 2.3, 2.05, 1.55, 0.15
    centers = []
    for i, (t, body, c) in enumerate(stages):
        x = sx0 + i*(sw+gap)
        ax.add_patch(FancyBboxPatch((x, sy), sw, sh,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=c, edgecolor="#333", lw=1.1))
        ax.text(x+sw/2, sy+sh-0.22, t, ha="center", va="top",
                fontsize=11.5, weight="bold", color="#222")
        ax.text(x+sw/2, sy+0.18, body, ha="center", va="bottom",
                fontsize=8.6, color="#333")
        centers.append((x+sw/2, sy+sh/2))

    # arrows entry → preprocess
    for y0 in (4.3, 1.8):
        ax.add_patch(FancyArrowPatch((2.05, y0),
            (sx0, sy+sh/2 + (0.35 if y0>3 else -0.35)),
            arrowstyle="->", mutation_scale=14,
            color="#555", lw=1.2))

    # arrows between stages
    for (x1,y1),(x2,y2) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((x1+sw/2-0.02, y1),
                                     (x2-sw/2+0.02, y2),
            arrowstyle="-|>", mutation_scale=16, color="#222", lw=1.4))

    # storage box at right (artifacts)
    ax.add_patch(FancyBboxPatch((9.0, 1.2), 0.9, 2.7,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="#f5f5f5", edgecolor="#444", lw=1.0))
    ax.text(9.45, 3.7, "Artifacts", ha="center", fontsize=10,
            weight="bold", color="#222")
    ax.text(9.45, 3.30, ".ofcd/", ha="center", fontsize=8.5)
    ax.text(9.45, 3.00, "project.yaml", ha="center", fontsize=8)
    ax.text(9.45, 2.70, "session.json", ha="center", fontsize=8)
    ax.text(9.45, 2.30, "runs/{id}/", ha="center", fontsize=8.5,
            color="#7a2222", weight="bold")
    ax.text(9.45, 2.00, "results.h5", ha="center", fontsize=8)
    ax.text(9.45, 1.70, "manifest.yaml", ha="center", fontsize=8)
    ax.text(9.45, 1.40, "annotations/", ha="center", fontsize=8)

    # arrow Postprocess → Artifacts
    px, py = centers[-1]
    ax.add_patch(FancyArrowPatch((px+sw/2-0.02, py), (9.0, py),
        arrowstyle="-|>", mutation_scale=16, color="#222", lw=1.4))

    # bottom band: Stage protocol + parallel
    ax.add_patch(FancyBboxPatch((2.7, 0.25), centers[-1][0]+sw/2-2.7, 0.7,
        boxstyle="round,pad=0.01,rounding_size=0.07",
        facecolor="#ffffff", edgecolor="#7a2222", lw=1.0))
    ax.text((2.7+centers[-1][0]+sw/2)/2, 0.6,
            "统一 Stage 协议  ·  StageEvent / CancelToken  ·  loky 并行  ·  HDF5 SWMR",
            ha="center", va="center", fontsize=9.5, color="#7a2222")

    # title
    ax.text(5.0, 5.18, "Preprocess → Compute → Postprocess",
            ha="center", fontsize=11.5, weight="bold", color="#222")

    out = os.path.join(OUT, "diag_pipeline.png")
    plt.tight_layout(pad=0.2)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_refraction())
    print(fig_pipeline())
