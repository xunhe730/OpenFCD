"""Generate the 4-page OpenFCD group-meeting deck (PKU academic style, 16:9)."""
from __future__ import annotations
import sys
sys.path.insert(0,
    "/sessions/tender-focused-hamilton/mnt/.claude/skills/pku-academic-pptx/scripts")
from pkupptx import Deck

OUT_PPTX = "/sessions/tender-focused-hamilton/mnt/outputs/OpenFCD_组会汇报_v3.pptx"
ETA_FULL    = "/sessions/tender-focused-hamilton/mnt/outputs/fig_eta_full.png"
ETA_PROFILE = "/sessions/tender-focused-hamilton/mnt/outputs/fig_eta_profile.png"
DIAG_REFR   = "/sessions/tender-focused-hamilton/mnt/outputs/diag_refraction.png"
DIAG_PIPE   = "/sessions/tender-focused-hamilton/mnt/outputs/diag_pipeline.png"


deck = Deck(
    conference="课题组组会汇报  ·  Group Meeting",
    location_date="北京大学  ·  2026 年 5 月",
    aspect="16:9",
)

# ----- Slide 1 : Title -----
deck.title_slide(
    title="OpenFCD：基于快速棋盘格解调的\n自由液面快速重构平台",
    author="Xe   (xunhe730@gmail.com)",
    affiliation=[
        "北京大学  ·  XJ-robot 课题组",
        "自由液面流动 / 软体游泳机器人  ·  v0.0.1",
    ],
)

# ----- Slide 2 : Theory -----
deck.content_slide(
    title="研究背景与 FCD 原理",
    bullets=[
        [
            ("自由液面 η(x,y,t)", "blue"),
            "  经棋盘格折射  →  像素位移场 ",
            ("u(x,y)", "blue"),
        ],
        [
            "斜率—位移关系：  ",
            ("u = α · h_p^eff · ∇η", "red"),
            "    （α ≈ 0.25，近轴近似）",
        ],
        [
            ("FCD 解调", "blue"),
            "：在两条载波 k_c 处傅里叶解相位",
        ],
        {"level": 2,
         "text": "ϕ_k = arg( D_k · D_k^ref* ),   u = −( k / |k|² ) · ϕ_k"},
        [
            ("Poisson 反演", "blue"),
            " 恢复高度场：  ",
            ("η = ( α · h_p^eff )⁻¹ · ∇⁻²(∇·u)", "red"),
        ],
    ],
    images=[
        {"path": DIAG_REFR,
         "caption": "棋盘格 → 流体 → 相机：折射几何与位移定义"},
    ],
    citation="Wildeman, Exp. Fluids 59:97 (2018);   Moisy et al., Exp. Fluids 46:1021 (2009)",
)

# ----- Slide 3 : Architecture -----
deck.content_slide(
    title="软件架构：三阶段流水线 + GUI / CLI 同源",
    bullets=[
        [
            ("Preprocess → Compute → Postprocess", "blue"),
            "    统一 Stage 协议  ·  StageEvent / CancelToken",
        ],
        [
            "GUI (PyQt6 三栏) 与 CLI (Typer) 共享同一 pipeline  →  ",
            ("byte-equal results.h5", "red"),
        ],
        [
            ("项目即目录 *.ofcd/", "blue"),
            "    HDF5 SWMR 实时尾随  ·  loky 并行",
        ],
        [
            "技术栈：Python 3.11+ · NumPy/FFT · h5py · Pydantic v2",
        ],
    ],
    images=[
        {"path": DIAG_PIPE,
         "caption": "数据流与产物：单一 pipeline 同时驱动 GUI 与 CLI"},
    ],
    citation=None,
)

# ----- Slide 4 : Validation + Summary -----
deck.content_slide(
    title="应用案例：毫米尺度软体游泳波动场重构",
    bullets=[
        [
            ("验证实验", "blue"),
            "：BOS @ 20 Hz, s2；重构 η 量程 ",
            ("± 0.5 mm", "red"),
            "，px/mm ≈ 7.3",
        ],
        [
            "前后向波长 ",
            ("λ_F, λ_R", "red"),
            " 与波高直接读出，可量化推进 / 阻力波结构",
        ],
        [
            ("鲁棒性", "blue"),
            "：跨期参考帧的载波尺度归一化  +  可选空间高通抑制低频漂移",
        ],
        [
            "v0.0.1 状态：核心 + GUI 完整；",
            ("A0 黄金回归 / replay 字节一致性测试", "red"),
            " 进行中",
        ],
    ],
    images=[
        {"path": ETA_FULL,
         "caption": "η(x,y) 全场重构：障碍物绕射波纹与各向异性辐射"},
        {"path": ETA_PROFILE,
         "caption": "y=0 切线 η(x) 剖面：λ_F、λ_R、波高 h 直接量化"},
    ],
    citation="数据：../BOS/DATA/20Hz/s2,  Img000942",
)

deck.save(OUT_PPTX)
print("Saved:", OUT_PPTX)
