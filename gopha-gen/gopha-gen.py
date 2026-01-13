#!/usr/bin/env python3
"""
Animated plasma background with neon balls — raw video generator for encoder testing.

Key features
- Palettes: --palette default|highcontrast
- Background contrast: --bg-contrast K
- Ball brightness gain: --balls-contrast K
- Occlusion stress: --occlusions
- Seamless loop: --makeloop
- Smoothing: --plasma-smooth

Output formats (planar)
- 8-bit:  yuv420p, yuv422p, yuv444p
- 10-bit: yuv420p10le, yuv422p10le, yuv444p10le
- RGB:    rgb24

Colorimetry
- --matrix bt601|bt709|bt2020nc
- --range limited|full

Tick wheel overlay
- --tickwheel to enable.
- --tickwheel-mode single|cumulative (default cumulative).
- --tickwheel-mix makes center cumulative and corners single.
- Positions: center, corners, or both.

Determinism / FPS conversion testing
- Scene state is a deterministic function of continuous time t (and --seed).
- Optional anchor snapping: frames near multiples of --anchor-period are snapped
  to the exact anchor time, so 50 fps and 60 fps will match at whole seconds.
"""

import argparse, os, sys, zlib
import numpy as np
from scipy.ndimage import gaussian_filter

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Plasma background with bouncing balls (raw video generator).")

    # Output
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--outfile", type=str, default="plasma.yuv")
    ap.add_argument("--output-format", type=str, default="yuv420p",
                    choices=[
                        "rgb24",
                        "yuv420p", "yuv422p", "yuv444p",
                        "yuv420p10le", "yuv422p10le", "yuv444p10le",
                    ])

    # Colorimetry
    ap.add_argument("--matrix", type=str, default="bt709",
                    choices=["bt601","bt709","bt2020nc"],
                    help="Y'CbCr conversion matrix (non-constant-luma variant for BT.2020).")
    ap.add_argument("--range", type=str, default="limited",
                    choices=["limited","full"], help="Signal range for YUV outputs.")

    # Randomness
    ap.add_argument("--seed", type=int, default=None)

    # Palette/contrast
    ap.add_argument("--palette", type=str, default="default", choices=["default","highcontrast"])
    ap.add_argument("--bg-contrast", type=float, default=1.0)
    ap.add_argument("--balls-contrast", type=float, default=1.0)

    # Plasma controls
    ap.add_argument("--plasma-bands", type=float, default=1.0)
    ap.add_argument("--plasma-speed", type=float, default=1.0)
    ap.add_argument("--plasma-motion-rate", type=float, default=1.0)
    ap.add_argument("--plasma-color-rate", type=float, default=1.0)
    ap.add_argument("--plasma-smooth", type=float, default=0.0)

    # Balls
    ap.add_argument("--balls", type=int, default=16)
    ap.add_argument("--radius-min", type=float, default=6.0)
    ap.add_argument("--radius-max", type=float, default=99.0)
    ap.add_argument("--speed-min", type=float, default=480.0)
    ap.add_argument("--speed-max", type=float, default=960.0)
    ap.add_argument("--ball-soft", type=float, default=0)
    ap.add_argument("--ball-specular", type=float, default=0)
    ap.add_argument("--ball-color", type=str, default="#39ff14")

    # Testing helpers
    ap.add_argument("--occlusions", action="store_true",
                    help="Encourage frequent overlaps/crossings.")

    # Seamless loop switch
    ap.add_argument("--makeloop", action="store_true",
                    help="Make the animation loop seamlessly over --duration.")

    # Anchor snapping (NEW)
    ap.add_argument("--anchor-period", type=float, default=1.0,
                    help="Seconds between scene anchor instants that must match across FPS (default 1.0). Set 0 to disable.")
    ap.add_argument("--anchor-snap", type=float, default=1e-6,
                    help="Snap tolerance in seconds for anchoring (default 1e-6).")

    # Tick wheel overlay
    ap.add_argument("--tickwheel", action="store_true",
                    help="Overlay frame 'tick wheel'. Default is time-based (so anchors match across FPS).")
    ap.add_argument("--tickwheel-framebased", action="store_true",
                    help="Legacy mode: active segment advances by frame index (will differ across FPS even at anchors).")
    ap.add_argument("--tickwheel-segs", type=int, default=60,
                    help="Number of segments in the wheel when time-based (or when you want a fixed look). Default 60.")
    ap.add_argument("--tickwheel-positions", type=str, default="both",
                    choices=["center","corners","both","default"],
                    help="Where to place the wheels. 'default' aliases to 'both'.")
    ap.add_argument("--tickwheel-radius", type=float, default=48.0,
                    help="Outer radius in pixels.")
    ap.add_argument("--tickwheel-thickness", type=float, default=12.0,
                    help="Ring thickness in pixels.")
    ap.add_argument("--tickwheel-gap", type=float, default=12.0,
                    help="Inner gap (hole) inside the ring.")
    ap.add_argument("--tickwheel-padding", type=float, default=32.0,
                    help="Padding from edges for corner wheels.")
    ap.add_argument("--tickwheel-alpha", type=float, default=1.0,
                    help="Overlay alpha 0..1.")
    ap.add_argument("--tickwheel-fg", type=str, default="#FFFFFF",
                    help="Active segment color.")
    ap.add_argument("--tickwheel-bg", type=str, default="#000000",
                    help="Ring background color.")
    ap.add_argument("--tickwheel-mode", type=str, default="cumulative",
                    choices=["single","cumulative"],
                    help="single = one white wedge moves; cumulative = ring fills frame-by-frame.")
    ap.add_argument("--tickwheel-mix", action="store_true",
                    help="Center wheel cumulative, corner wheels single-segment.")

    return ap.parse_args()

# -----------------------------
# Deterministic RNG substreams
# -----------------------------

def _tag_u32(tag: str) -> int:
    # Stable across runs/platforms (unlike Python's built-in hash()).
    return zlib.crc32(tag.encode("utf-8")) & 0xFFFFFFFF

def make_rng(seed, tag: str) -> np.random.Generator:
    """
    Create an RNG stream that depends only on (seed, tag), not on call order.
    This prevents FPS or feature flags from accidentally changing the initial state.
    """
    if seed is None:
        return np.random.default_rng()
    # SeedSequence accepts sequences of uint32-ish integers.
    ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, _tag_u32(tag)])
    return np.random.default_rng(ss)

# -----------------------------
# Helpers
# -----------------------------

def clamp01(x): return np.clip(x, 0.0, 1.0)

def parse_hex_color(s):
    s = s.strip()
    if s.startswith("#"): s = s[1:]
    if len(s) == 3: s = "".join([c*2 for c in s])
    if len(s) != 6: raise ValueError("Expected #RRGGBB or #RGB")
    r = int(s[0:2],16)/255.0; g = int(s[2:4],16)/255.0; b = int(s[4:6],16)/255.0
    return np.array([r,g,b], dtype=np.float32)

def apply_contrast(x, k):
    if k == 1.0: return x
    return clamp01(0.5 + (x - 0.5) * k)

def rgb_bytes(rgb):
    return (clamp01(rgb)*255.0 + 0.5).astype(np.uint8).tobytes(order="C")

def anchor_time(t, period, snap):
    """Snap t to the nearest integer multiple of period if within snap seconds."""
    if period is None or period <= 0:
        return t
    k = round(t / period)
    t_anchor = k * period
    if abs(t - t_anchor) <= snap:
        return t_anchor
    return t

# -----------------------------
# Palettes
# -----------------------------

def palette_default(t):
    keys = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    cols = np.array([
        [0.13, 0.60, 0.66],
        [0.60, 0.30, 0.62],
        [0.98, 0.45, 0.20],
        [1.00, 0.78, 0.25],
        [1.00, 0.95, 0.78],
    ], dtype=np.float32)
    t = clamp01(t)
    idx = np.minimum(np.searchsorted(keys, t, side="right") - 1, len(keys)-2)
    t0 = keys[idx]; t1 = keys[idx+1]
    c0 = cols[idx]; c1 = cols[idx+1]
    u = (t - t0) / np.maximum(1e-6, (t1 - t0))
    u = u*u*(3 - 2*u)
    return (c0*(1-u)[...,None] + c1*u[...,None]).astype(np.float32)

def palette_highcontrast(t):
    keys = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00], dtype=np.float32)
    cols = np.array([
        [0.03, 0.03, 0.03],
        [0.90, 0.10, 0.10],
        [0.95, 0.95, 0.05],
        [0.10, 0.95, 0.10],
        [0.10, 0.30, 1.00],
        [0.98, 0.98, 0.98],
    ], dtype=np.float32)
    t = clamp01(t)
    idx = np.minimum(np.searchsorted(keys, t, side="right") - 1, len(keys)-2)
    t0 = keys[idx]; t1 = keys[idx+1]
    c0 = cols[idx]; c1 = cols[idx+1]
    u = (t - t0) / np.maximum(1e-6, (t1 - t0))
    u = u*u*(3 - 2*u)
    return (c0*(1-u)[...,None] + c1*u[...,None]).astype(np.float32)

# -----------------------------
# Plasma background (time-based; deterministic across FPS)
# -----------------------------

def init_plasma_params(rng):
    # Randomize subtly but deterministically (depends only on seed/tagged RNG)
    return {
        "a1": float(rng.uniform(1.2, 2.1)),
        "b1": float(rng.uniform(1.0, 2.0)),
        "a2": float(rng.uniform(0.8, 1.6)),
        "b2": float(rng.uniform(1.0, 2.3)),
        "d":  float(rng.uniform(1.3, 2.3)),
        "pm1": float(rng.uniform(0, 2*np.pi)),
        "pm2": float(rng.uniform(0, 2*np.pi)),
        "pm3": float(rng.uniform(0, 2*np.pi)),
        "pc1": float(rng.uniform(0, 2*np.pi)),
        "pc2": float(rng.uniform(0, 2*np.pi)),
    }

def render_plasma(H, W, t_motion, t_color, bands, loop, palette_fn, params):
    """
    IMPORTANT: Must be deterministic and depend ONLY on (H,W,t_motion,t_color,bands,loop,palette_fn,params).
    No frame index, no per-frame RNG.
    """
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    s = bands * 2*np.pi / min(H, W)
    cx = 0.5*W; cy = 0.5*H

    a1, b1 = params["a1"]*s, params["b1"]*s
    a2, b2 = params["a2"]*s, params["b2"]*s
    d = params["d"]*s

    if loop:
        phi = t_motion
        F = (
            np.sin(a1*x + b1*y + 1.0*phi + params["pm1"]) +
            np.sin(a2*x - b2*y - 1.5*phi + params["pm2"]) +
            np.sin(d * np.sqrt((x-cx)**2 + (y-cy)**2) + 0.7*phi + params["pm3"])
        )
        F01 = (F - F.min()) / np.maximum(1e-6, (F.max() - F.min()))
        color_phase = 0.5*(1 + np.cos(2*np.pi*(F01 - 0.15*np.sin(t_color + params["pc1"])) + params["pc2"]))
        return palette_fn(color_phase)
    else:
        F = (
            np.sin(a1*x + b1*y + 0.7*t_motion + params["pm1"]) +
            np.sin(a2*x - b2*y - 0.9*t_motion + params["pm2"]) +
            np.sin(d * np.sqrt((x-cx)**2 + (y-cy)**2) + 0.5*t_motion + params["pm3"])
        )
        F01 = (F - F.min()) / np.maximum(1e-6, (F.max() - F.min()))
        color_phase = 0.5*(1 + np.cos(2*np.pi*(F01 - 0.15*np.sin(0.5*t_color + params["pc1"])) + params["pc2"]))
        return palette_fn(color_phase)

# -----------------------------
# Balls (time-based; deterministic across FPS)
# -----------------------------

def init_balls(n, W, H, rng, rmin, rmax, vmin, vmax):
    balls = []
    for _ in range(n):
        r = float(rng.uniform(rmin, rmax))
        x0 = float(rng.uniform(r+2, W - r - 2))
        y0 = float(rng.uniform(r+2, H - r - 2))
        ang = float(rng.uniform(0, 2*np.pi))
        speed = float(rng.uniform(vmin, vmax))
        vx = float(speed*np.cos(ang))
        vy = float(speed*np.sin(ang))
        balls.append(dict(x=x0, y=y0, r=r, vx=vx, vy=vy, x0=x0, y0=y0))
    return balls

def triangle_wave(u):
    u = u - np.floor(u)
    return 1.0 - np.abs(1.0 - 2.0*u)

def reflect_1d(x0, v, t, xmin, xmax):
    L = max(1e-9, (xmax - xmin))
    phase0 = (x0 - xmin) / L
    u = phase0 + (v / L) * t
    return xmin + L * triangle_wave(u)

def eval_balls_at_time(balls, t, W, H):
    for b in balls:
        r = b["r"]
        xmin, xmax = r + 2, W - r - 2
        ymin, ymax = r + 2, H - r - 2
        b["x"] = reflect_1d(b["x0"], b["vx"], t, xmin, xmax)
        b["y"] = reflect_1d(b["y0"], b["vy"], t, ymin, ymax)

def init_loop_phases(n_balls, rng, occl: bool):
    """
    Precompute loop phases once (deterministic). Avoids hidden per-frame RNG use.
    """
    if n_balls <= 0:
        return [], []
    if occl:
        bands = max(2, int(np.sqrt(n_balls)))
        phi_x = np.empty(n_balls, dtype=np.float64)
        phi_y = np.empty(n_balls, dtype=np.float64)
        for i in range(n_balls):
            band = i % bands
            base = band / bands
            phi_x[i] = base + 0.08*rng.standard_normal()
            phi_y[i] = (bands - 1 - band)/bands + 0.08*rng.standard_normal()
        phi_x = np.mod(phi_x, 1.0)
        phi_y = np.mod(phi_y, 1.0)
    else:
        phi_x = rng.random(n_balls)
        phi_y = rng.random(n_balls)
    return [float(v) for v in phi_x], [float(v) for v in phi_y]

def balls_positions_loop(balls, t, T, W, H, phi_x, phi_y, px=7, qy=5):
    # Pure function of time and precomputed phases (deterministic across FPS)
    u = (t / max(1e-6, T))
    for i, b in enumerate(balls):
        r = b["r"]
        xmin, xmax = r + 2, W - r - 2
        ymin, ymax = r + 2, H - r - 2
        tx = triangle_wave(px*u + phi_x[i])
        ty = triangle_wave(qy*u + phi_y[i])
        b["x"] = xmin + tx * (xmax - xmin)
        b["y"] = ymin + ty * (ymax - ymin)

def draw_ball_over(img, ball, color, soft=0.35, specular=0.35, ball_gain=1.0):
    H, W, _ = img.shape
    cx, cy, r = ball["x"], ball["y"], ball["r"]
    y0 = max(0, int(np.floor(cy - r - 2))); y1 = min(H, int(np.ceil(cy + r + 2)))
    x0 = max(0, int(np.floor(cx - r - 2))); x1 = min(W, int(np.ceil(cx + r + 2)))
    if x0>=x1 or y0>=y1: return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx; dy = yy - cy
    dist = np.sqrt(dx*dx + dy*dy)

    edge = (dist - r) / max(1e-3, r*0.20*max(1e-6, soft))
    alpha = clamp01(0.5 - 0.5*np.tanh(edge))

    inside = dist <= r
    nx = np.zeros_like(dist); ny = np.zeros_like(dist); nz = np.zeros_like(dist)
    safe = max(1e-6, r)
    nx[inside] = dx[inside]/safe; ny[inside] = dy[inside]/safe
    nz[inside] = np.sqrt(np.clip(1.0 - nx[inside]**2 - ny[inside]**2, 0.0, 1.0))

    L = np.array([-0.45, -0.55, 0.70], dtype=np.float32); L /= np.linalg.norm(L)
    lam = np.maximum(0.0, nx*L[0] + ny*L[1] + nz[2]).astype(np.float32)
    Hvec = (L + np.array([0,0,1], np.float32)); Hvec /= np.linalg.norm(Hvec)
    spec = np.maximum(0.0, nx*Hvec[0] + ny*Hvec[1] + nz[2]).astype(np.float32)**48

    shaded = clamp01(color[None,None,:] * ball_gain * (0.55 + 0.45*lam[...,None]) + specular*spec[...,None])

    region = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = region*(1 - alpha[...,None]) + shaded*alpha[...,None]

# -----------------------------
# Tick wheel overlay
# -----------------------------

def draw_tickwheel(img, cx, cy, R, thickness, hole, fg, bg, alpha, segments, active_idx, mode="cumulative"):
    H, W, _ = img.shape
    rmax = int(np.ceil(R))
    y0 = max(0, int(np.floor(cy - rmax))); y1 = min(H, int(np.ceil(cy + rmax)))
    x0 = max(0, int(np.floor(cx - rmax))); x1 = min(W, int(np.ceil(cx + rmax)))
    if x0>=x1 or y0>=y1: return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx; dy = yy - cy
    dist = np.sqrt(dx*dx + dy*dy)

    ring = (dist <= R) & (dist >= max(0.0, R - thickness))
    if hole > 0:
        ring &= dist >= max(0.0, R - thickness - hole)
    if not np.any(ring): return

    ang = np.arctan2(dy, dx)  # [-pi, pi]
    ang = np.where(ang < 0, ang + 2*np.pi, ang)
    seg_w = 2*np.pi / max(1, segments)
    seg_idx = np.floor(ang / seg_w).astype(np.int32)
    seg_idx = np.clip(seg_idx, 0, segments-1)

    if mode == "single":
        is_white = (seg_idx == (active_idx % segments))
    else:
        k = active_idx % segments
        is_white = seg_idx <= k

    is_white &= ring
    is_black = ring & (~is_white)

    region = img[y0:y1, x0:x1]
    out = region.copy()
    if np.any(is_black):
        out[is_black] = (1 - alpha) * region[is_black] + alpha * bg[None,:]
    if np.any(is_white):
        out[is_white] = (1 - alpha) * region[is_white] + alpha * fg[None,:]
    img[y0:y1, x0:x1] = out

def get_tickwheel_positions(W, H, R, pad, mode):
    pos = []
    if mode in ("center","both","default"):
        pos.append((0.5*W, 0.5*H))
    if mode in ("corners","both","default"):
        pos.extend([
            (pad + R,             pad + R),
            (W - pad - R,         pad + R),
            (pad + R,             H - pad - R),
            (W - pad - R,         H - pad - R),
        ])
    return pos

def tickwheel_active_idx_timebased(t, segments, spins_per_sec=1.0):
    # use floor() so that at integer seconds, idx is exactly 0
    phase = (t * spins_per_sec) % 1.0
    return int(np.floor(phase * max(1, segments))) % max(1, segments)

# -----------------------------
# Color matrices and YUV writers
# -----------------------------

def get_yuv_matrices(matrix: str):
    if matrix == "bt601":
        Kr, Kg, Kb = 0.2990, 0.5870, 0.1140
    elif matrix == "bt709":
        Kr, Kg, Kb = 0.2126, 0.7152, 0.0722
    elif matrix == "bt2020nc":
        Kr, Kg, Kb = 0.2627, 0.6780, 0.0593
    else:
        raise ValueError("Unknown matrix")
    return Kr, Kg, Kb

def _quantize_u16(x):
    return np.clip(np.round(x), 0, 1023).astype(np.uint16)

def rgb_to_ycbcr_planar(rgb, subsampling: str, bitdepth: int, rng: str, matrix: str):
    H, W, _ = rgb.shape
    R = clamp01(rgb[...,0]).astype(np.float32)
    G = clamp01(rgb[...,1]).astype(np.float32)
    B = clamp01(rgb[...,2]).astype(np.float32)

    Kr, Kg, Kb = get_yuv_matrices(matrix)

    Yf = Kr*R + Kg*G + Kb*B
    Cbf = 0.5 * (B - Yf) / (1 - Kb)
    Crf = 0.5 * (R - Yf) / (1 - Kr)

    if subsampling == "444":
        Yf_s, Cbf_s, Crf_s = Yf, Cbf, Crf
    elif subsampling == "422":
        Yf_s = Yf
        Cbf_s = 0.5*(Cbf[:, 0::2] + Cbf[:, 1::2])
        Crf_s = 0.5*(Crf[:, 0::2] + Crf[:, 1::2])
    elif subsampling == "420":
        Yf_s = Yf
        Cbf_s = 0.25*(Cbf[0::2,0::2] + Cbf[0::2,1::2] + Cbf[1::2,0::2] + Cbf[1::2,1::2])
        Crf_s = 0.25*(Crf[0::2,0::2] + Crf[0::2,1::2] + Crf[1::2,0::2] + Crf[1::2,1::2])
    else:
        raise ValueError("Bad subsampling")

    if bitdepth == 8:
        if rng == "limited":
            Y  = np.clip(np.round(16.0  + 219.0 * Yf_s),  0, 255).astype(np.uint8)
            Cb = np.clip(np.round(128.0 + 224.0 * Cbf_s), 0, 255).astype(np.uint8)
            Cr = np.clip(np.round(128.0 + 224.0 * Crf_s), 0, 255).astype(np.uint8)
        else:
            Y  = np.clip(np.round(255.0 * Yf_s),               0, 255).astype(np.uint8)
            Cb = np.clip(np.round(255.0 * (0.5 + Cbf_s)),      0, 255).astype(np.uint8)
            Cr = np.clip(np.round(255.0 * (0.5 + Crf_s)),      0, 255).astype(np.uint8)
    elif bitdepth == 10:
        if rng == "limited":
            Y  = _quantize_u16(64.0  + 876.0 * Yf_s)
            Cb = _quantize_u16(512.0 + 896.0 * Cbf_s)
            Cr = _quantize_u16(512.0 + 896.0 * Crf_s)
        else:
            Y  = _quantize_u16(1023.0 * Yf_s)
            Cb = _quantize_u16(1023.0 * (0.5 + Cbf_s))
            Cr = _quantize_u16(1023.0 * (0.5 + Crf_s))
    else:
        raise ValueError("Unsupported bitdepth")

    return Y, Cb, Cr

def write_planar_yuv(fh, Y, U, V, bitdepth: int):
    if bitdepth == 8:
        fh.write(Y.tobytes(order="C"))
        fh.write(U.tobytes(order="C"))
        fh.write(V.tobytes(order="C"))
    else:
        if Y.dtype != np.uint16: raise ValueError("10-bit expects uint16 buffers")
        if Y.dtype.byteorder == ">" or (Y.dtype.byteorder == "=" and sys.byteorder == "big"):
            Y = Y.byteswap().newbyteorder()
            U = U.byteswap().newbyteorder()
            V = V.byteswap().newbyteorder()
        fh.write(Y.tobytes(order="C"))
        fh.write(U.tobytes(order="C"))
        fh.write(V.tobytes(order="C"))

# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    W, H = args.width, args.height
    FPS = float(args.fps)
    frames = int(round(args.duration * FPS))
    T = max(1e-6, float(args.duration))

    # Geometry constraints for subsampling
    fmt = args.output_format
    subsampling = None
    bitdepth = 8
    if fmt == "rgb24":
        pass
    else:
        if "420" in fmt:
            subsampling = "420"
            if (W % 2) or (H % 2):
                print("Error: yuv420 formats require even width and height.", file=sys.stderr)
                sys.exit(1)
        elif "422" in fmt:
            subsampling = "422"
            if (W % 2):
                print("Error: yuv422 formats require even width.", file=sys.stderr)
                sys.exit(1)
        elif "444" in fmt:
            subsampling = "444"
        else:
            print("Unknown output-format", file=sys.stderr); sys.exit(1)
        if "10le" in fmt:
            bitdepth = 10

    # IMPORTANT: deterministic substreams (FPS/feature-proof)
    rng_plasma = make_rng(args.seed, "plasma_params")
    rng_balls  = make_rng(args.seed, "balls_init")
    rng_loop   = make_rng(args.seed, "loop_phases")

    # Palette selector
    palette_fn = palette_default if args.palette == "default" else palette_highcontrast

    # Plasma params
    plasma_params = init_plasma_params(rng_plasma)

    # Balls
    balls = init_balls(
        args.balls, W, H, rng_balls,
        rmin=args.radius_min, rmax=args.radius_max,
        vmin=args.speed_min, vmax=args.speed_max
    )
    ball_color = parse_hex_color(args.ball_color)

    # Rates
    motion_k = args.plasma_speed * args.plasma_motion_rate
    color_k  = args.plasma_speed * args.plasma_color_rate

    # Loop params
    loop_motion_cycles = 3
    loop_color_cycles  = 1
    loop_ball_x_cycles = 7
    loop_ball_y_cycles = 5

    # Loop phases (precomputed, deterministic)
    loop_phi_x, loop_phi_y = init_loop_phases(len(balls), rng_loop, occl=args.occlusions)

    # Tickwheel colors
    tw_fg = parse_hex_color(args.tickwheel_fg)
    tw_bg = parse_hex_color(args.tickwheel_bg)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    f = open(args.outfile, "wb")

    try:
        for i in range(frames):
            # Continuous time for this frame, then optionally anchor-snap it.
            t = float(i) / FPS
            t_eff = anchor_time(t, args.anchor_period, args.anchor_snap)

            if args.makeloop:
                phi_motion = 2*np.pi * loop_motion_cycles * (t_eff / T)
                phi_color  = 2*np.pi * loop_color_cycles  * (t_eff / T)
                bg = render_plasma(H, W, phi_motion, phi_color, args.plasma_bands,
                                   loop=True, palette_fn=palette_fn, params=plasma_params)
                if args.plasma_smooth > 0:
                    for c in range(3):
                        bg[:,:,c] = gaussian_filter(bg[:,:,c], sigma=args.plasma_smooth)
                bg = apply_contrast(bg, args.bg_contrast)

                # Looping ball motion (time-based) with precomputed phases
                balls_positions_loop(
                    balls, t_eff, T, W, H, loop_phi_x, loop_phi_y,
                    px=loop_ball_x_cycles, qy=loop_ball_y_cycles
                )
            else:
                t_motion = t_eff * motion_k
                t_color  = t_eff * color_k
                bg = render_plasma(H, W, t_motion, t_color, args.plasma_bands,
                                   loop=False, palette_fn=palette_fn, params=plasma_params)
                if args.plasma_smooth > 0:
                    for c in range(3):
                        bg[:,:,c] = gaussian_filter(bg[:,:,c], sigma=args.plasma_smooth)
                bg = apply_contrast(bg, args.bg_contrast)

                # FPS-independent analytic ball positions:
                eval_balls_at_time(balls, t_eff, W, H)

            # Draw balls
            for b in balls:
                draw_ball_over(bg, b, ball_color, soft=args.ball_soft,
                               specular=args.ball_specular, ball_gain=args.balls_contrast)

            # Tick wheel overlay (optional)
            if args.tickwheel:
                if args.tickwheel_framebased:
                    segs = max(1, int(round(FPS)))
                    active_idx = i % segs
                else:
                    segs = max(1, int(args.tickwheel_segs))
                    active_idx = tickwheel_active_idx_timebased(t_eff, segs, spins_per_sec=1.0)

                R = float(args.tickwheel_radius)
                thickness = float(args.tickwheel_thickness)
                hole = float(args.tickwheel_gap)
                pad = float(args.tickwheel_padding)
                alpha = float(np.clip(args.tickwheel_alpha, 0.0, 1.0))

                posmode = "both" if args.tickwheel_positions == "default" else args.tickwheel_positions
                positions = get_tickwheel_positions(W, H, R, pad, posmode)

                mode_all = args.tickwheel_mode
                if args.tickwheel_mix:
                    center_pos = []
                    corner_pos = []
                    for (cx, cy) in positions:
                        is_center = abs(cx - 0.5*W) < 0.5 and abs(cy - 0.5*H) < 0.5
                        if is_center:
                            center_pos.append((cx, cy))
                        else:
                            corner_pos.append((cx, cy))
                    for (cx, cy) in center_pos:
                        draw_tickwheel(bg, cx, cy, R, thickness, hole, tw_fg, tw_bg, alpha, segs, active_idx, mode="cumulative")
                    for (cx, cy) in corner_pos:
                        draw_tickwheel(bg, cx, cy, R, thickness, hole, tw_fg, tw_bg, alpha, segs, active_idx, mode="single")
                else:
                    for (cx, cy) in positions:
                        draw_tickwheel(bg, cx, cy, R, thickness, hole, tw_fg, tw_bg, alpha, segs, active_idx, mode=mode_all)

            # Write output
            if fmt == "rgb24":
                f.write(rgb_bytes(bg))
            else:
                Y, U, V = rgb_to_ycbcr_planar(bg, subsampling, bitdepth, args.range, args.matrix)
                write_planar_yuv(f, Y, U, V, bitdepth)

            if (i+1) % max(1, int(round(FPS))) == 0:
                print(f"{i+1}/{frames} frames ({(i+1)/FPS:.1f}s)", flush=True)
    finally:
        f.close()

    print(f"Done. Wrote {frames} frames to {args.outfile}")
    if fmt == "rgb24":
        print(f"Play: ffplay -f rawvideo -pixel_format rgb24 -video_size {W}x{H} -framerate {FPS} {args.outfile}")
    else:
        pix = fmt
        print(f"Play: ffplay -f rawvideo -pixel_format {pix} -video_size {W}x{H} -framerate {FPS} {args.outfile}")

if __name__ == "__main__":
    main()