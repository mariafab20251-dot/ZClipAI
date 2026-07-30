"""Cinematic video effects — ffmpeg filtergraph stages for the "reel" look.
Ported from ClipForge's effects system.

Builds filtergraph segments that sit between the reframed video and the burned-in
captions: colour grades, glow/bloom, grain, vignette, gradients, letterbox bars,
sharpen, chromatic aberration. Everything is expressed as [in]...[out] segments
that join into a -filter_complex string.

Each effect is input-less — all processing happens on the single video stream.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Colour-grade presets -> ffmpeg filter chain.
COLOR_GRADES: dict[str, str] = {
    "none": "",
    "warm": "eq=saturation=1.10,colorbalance=rm=0.06:gm=0.02:bm=-0.06:rh=0.05:bh=-0.06",
    "cool": "eq=saturation=1.05,colorbalance=rm=-0.05:bm=0.06:bh=0.06",
    "teal_orange": (
        "colorbalance=rh=0.08:gh=0.02:bh=-0.05:bs=0.06:gs=0.02:rs=-0.05,"
        "eq=saturation=1.12:contrast=1.05"
    ),
    "vintage": "curves=preset=vintage",
    "vibrant": "eq=saturation=1.35:contrast=1.08:brightness=0.01",
    "bw": "hue=s=0,eq=contrast=1.10",
}

_GRAD_BANDS = 64


def _f(x: float, lo: float, hi: float) -> float:
    """Clamp a 0..100 'strength' value to [lo, hi]."""
    frac = max(0.0, min(100.0, float(x))) / 100.0
    return lo + frac * (hi - lo)


def _on(cfg: dict, key: str) -> bool:
    return bool(cfg.get(key))


def _num(cfg: dict, key: str, default: float) -> float:
    v = cfg.get(key)
    try:
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _gradient_bands(vw: int, vh: int, height_pct: float, strength: float, top: bool) -> str:
    """A comma-chain of drawbox strips approximating a smooth dark gradient."""
    h_grad = max(1, int(vh * max(0.0, min(0.8, height_pct / 100.0))))
    n = _GRAD_BANDS
    step = h_grad / n
    m = max(0.0, min(0.96, strength / 100.0))
    base = 0 if top else (vh - h_grad)

    boxes: List[str] = []
    for k in range(n):
        y = base + int(round(k * step))
        h = base + int(round((k + 1) * step)) - y
        if h <= 0:
            continue
        frac = (k + 0.5) / n
        eased = frac * frac * (3.0 - 2.0 * frac)
        alpha = m * eased if not top else m * (1.0 - eased)
        if alpha <= 0.002:
            continue
        boxes.append(f"drawbox=x=0:y={y}:w=iw:h={h}:color=black@{alpha:.4f}:t=fill")
    return ",".join(boxes)


def build_cinematic_stages(
    cfg: Optional[dict], in_label: str, vw: int, vh: int
) -> Tuple[List[str], str]:
    """Build cinematic filtergraph stages.

    Returns ``(stages, out_label)`` where ``stages`` is a list of ``[a]...[b]``
    segments and ``out_label`` is the label captions should consume. Returns
    ``([], in_label)`` when no effects are enabled.
    """
    if not cfg:
        return [], in_label

    stages: List[str] = []
    cur = in_label
    idx = 0

    def push(filters: str) -> None:
        nonlocal cur, idx
        nxt = f"cine{idx}"
        stages.append(f"[{cur}]{filters}[{nxt}]")
        cur, idx = nxt, idx + 1

    # 1) Colour grade (whole image).
    grade = COLOR_GRADES.get(str(cfg.get("color_grade") or "none"))
    if grade:
        push(grade)

    # 2) Glow / bloom — isolate highlights, blur, screen-blend back.
    if _on(cfg, "glow"):
        s = _f(_num(cfg, "glow_strength", 50), 6.0, 22.0)
        o = _f(_num(cfg, "glow_strength", 50), 0.35, 0.85)
        nxt = f"cine{idx}"
        stages.append(
            f"[{cur}]format=gbrp,split=2[{nxt}a][{nxt}b];"
            f"[{nxt}b]curves=all='0/0 0.55/0 0.8/0.55 1/1',format=gray,format=gbrp,"
            f"gblur=sigma={s:.1f}[{nxt}c];"
            f"[{nxt}a][{nxt}c]blend=all_mode=screen:all_opacity={o:.3f},"
            f"format=yuv420p[{nxt}]"
        )
        cur, idx = nxt, idx + 1

    # 3) Film grain.
    if _on(cfg, "grain"):
        n = int(round(_f(_num(cfg, "grain_strength", 40), 4.0, 32.0)))
        push(f"noise=alls={n}:allf=t+u")

    # 4) Vignette (darkened corners).
    if _on(cfg, "vignette"):
        ang = _f(_num(cfg, "vignette_strength", 50), 0.45, 1.25)
        push(f"vignette=angle={ang:.3f}")

    # 5) Bottom gradient (scrim under captions).
    if _on(cfg, "bottom_gradient"):
        bands = _gradient_bands(
            vw, vh, _num(cfg, "bottom_gradient_height", 25),
            _num(cfg, "bottom_gradient_strength", 70), top=False,
        )
        if bands:
            push(bands)

    # 6) Top gradient.
    if _on(cfg, "top_gradient"):
        bands = _gradient_bands(
            vw, vh, _num(cfg, "top_gradient_height", 20),
            _num(cfg, "top_gradient_strength", 60), top=True,
        )
        if bands:
            push(bands)

    # 7) Cinematic letterbox bars.
    if _on(cfg, "letterbox"):
        bh = max(1, int(vh * _f(_num(cfg, "letterbox_size", 50), 0.05, 0.14)))
        push(
            f"drawbox=x=0:y=0:w=iw:h={bh}:color=black:t=fill,"
            f"drawbox=x=0:y=ih-{bh}:w=iw:h={bh}:color=black:t=fill"
        )

    # 8) Sharpen / clarity.
    if _on(cfg, "sharpen"):
        amt = _f(_num(cfg, "sharpen_strength", 40), 0.2, 1.6)
        push(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={amt:.2f}")

    # 9) Chromatic aberration.
    if _on(cfg, "chroma_shift"):
        px = max(1, int(round(_f(_num(cfg, "chroma_shift_strength", 40), 1.0, 6.0))))
        push(f"rgbashift=rh=-{px}:bh={px}:edge=smear")

    return stages, cur
