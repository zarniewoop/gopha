import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math


class SlateScene:
    """
    Displays a still background image with:
      - Frame number + SMPTE timecode (HH:MM:SS:FF) on the left
      - Wall-clock time (HH:MM:SS.mmm) on the right
      - A pie-segment dial in the centre, one segment per frame per second

    Config keys (under 'scene'):
      image_path     : path to source image, must match video width x height
      font_path      : (optional) path to a .ttf font
      font_size      : (optional) font size in pixels, default 72
      text_color     : (optional) [R,G,B] in nit scale, default peak_lum white
      bg_color       : (optional) [R,G,B] in nit scale for text backing box
      dial_cx        : (optional) dial centre X, default video centre
      dial_cy        : (optional) dial centre Y, default video centre
      dial_radius    : (optional) outer radius in pixels, default 180
      dial_color_on  : (optional) [R,G,B] nit scale for filled segments
      dial_color_off : (optional) [R,G,B] nit scale for empty segments
      dial_gap_deg   : (optional) gap between segments in degrees, default 2.0
    """

    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.cfg = config
        self.peak_lum = float(config["hdr"]["peak_luminance"])

        scene_cfg = config.get("scene", {})
        self.fps = float(config["video"]["fps"])
        self.fps_round = int(round(self.fps))

        # ── background image ────────────────────────────────────────────────
        image_path = scene_cfg.get("image_path")
        if not image_path:
            raise ValueError("scene.image_path is required for slate scene")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"scene.image_path not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        if img.width != width or img.height != height:
            raise ValueError(
                f"Image size {img.width}x{img.height} does not match "
                f"video size {width}x{height}. No scaling is applied."
            )

        arr = np.asarray(img, dtype=np.float64) / 255.0
        arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        self.background = arr * self.peak_lum

        # ── font ─────────────────────────────────────────────────────────────
        font_path = scene_cfg.get("font_path")
        font_size = int(scene_cfg.get("font_size", 72))
        try:
            self.font = (
                ImageFont.truetype(font_path, font_size)
                if font_path
                else ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", font_size
                )
            )
        except IOError:
            self.font = ImageFont.load_default()

        # ── colours (stored as 8-bit for Pillow) ─────────────────────────────
        def nit_to_8bit(v):
            return int(np.clip(v / self.peak_lum * 255, 0, 255))

        def cfg_color_8bit(key, default_nit):
            c = scene_cfg.get(key, default_nit)
            return tuple(nit_to_8bit(v) for v in c)

        self.text_color = cfg_color_8bit(
            "text_color", [self.peak_lum, self.peak_lum, self.peak_lum]
        )
        self.bg_color = cfg_color_8bit("bg_color", [0, 0, 0])

        # ── dial params ───────────────────────────────────────────────────────
        self.dial_cx = int(scene_cfg.get("dial_cx", width // 2))
        self.dial_cy = int(scene_cfg.get("dial_cy", height // 2))
        self.dial_radius = int(scene_cfg.get("dial_radius", 180))
        self.dial_color_on = cfg_color_8bit(
            "dial_color_on", [self.peak_lum, self.peak_lum * 0.6, 0]
        )
        self.dial_color_off = cfg_color_8bit(
            "dial_color_off", [self.peak_lum * 0.15] * 3
        )
        self.dial_gap_deg = float(scene_cfg.get("dial_gap_deg", 2.0))

        # text margin from edges
        self.margin = int(scene_cfg.get("text_margin", 60))

        self._render_count = 0  # fallback deterministic frame counter

    def _timecode(self, frame_number):
        """Return (hh, mm, ss, ff) SMPTE timecode from absolute frame number."""
        fps = self.fps_round
        ff = frame_number % fps
        total_s = frame_number // fps
        ss = total_s % 60
        mm = (total_s // 60) % 60
        hh = total_s // 3600
        return hh, mm, ss, ff

    def _wallclock(self, frame_number):
        """Return HH:MM:SS.mmm string derived purely from frame number."""
        total_ms = int(round(frame_number / self.fps * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        ss = total_s % 60
        mm = (total_s // 60) % 60
        hh = total_s // 3600
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

    def _draw_dial(self, draw, frame_number):
        """Draw a pie-segment dial; one segment per frame in the current second."""
        n_segs = self.fps_round
        # which segment is the current frame within this second
        current_seg = frame_number % n_segs

        cx, cy = self.dial_cx, self.dial_cy
        r = self.dial_radius
        inner_r = r * 0.35           # inner hole radius
        gap = self.dial_gap_deg
        seg_span = (360.0 / n_segs) - gap

        for seg in range(n_segs):
            # start at top (-90°) going clockwise
            start_angle = -90.0 + seg * (360.0 / n_segs)
            end_angle = start_angle + seg_span

            color = self.dial_color_on if seg <= current_seg else self.dial_color_off

            # Outer arc (Pillow pieslice from bounding box)
            bbox_outer = [cx - r, cy - r, cx + r, cy + r]
            bbox_inner = [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r]

            # Draw outer pie slice then punch inner hole in bg colour
            draw.pieslice(bbox_outer, start=start_angle, end=end_angle, fill=color)
            draw.pieslice(bbox_inner, start=start_angle, end=end_angle,
                          fill=self._sample_bg_center())

        # solid centre circle
        draw.ellipse(
            [cx - inner_r + 2, cy - inner_r + 2, cx + inner_r - 2, cy + inner_r - 2],
            fill=self.bg_color,
        )

        # frame-within-second number in centre
        centre_label = str(frame_number % n_segs)
        bbox = draw.textbbox((0, 0), centre_label, font=self.font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, cy - th // 2),
            centre_label,
            font=self.font,
            fill=self.text_color,
        )

    def _sample_bg_center(self):
        """Return an 8-bit RGB tuple from the background at the dial centre."""
        px = np.clip(
            self.background[self.dial_cy, self.dial_cx] / self.peak_lum * 255,
            0, 255,
        ).astype(np.uint8)
        return tuple(int(v) for v in px)

    def _draw_text_with_box(self, draw, xy, text):
        pad = 10
        bbox = draw.textbbox(xy, text, font=self.font)
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=self.bg_color,
        )
        draw.text(xy, text, font=self.font, fill=self.text_color)

    def render(self, t, frame_index=None):
        # Never depend on timebase t for displayed counters.
        if frame_index is None:
            frame_number = self._render_count
            self._render_count += 1
        else:
            frame_number = int(frame_index)

        hh, mm, ss, ff = self._timecode(frame_number)
        smpte = f"F {frame_number:06d}   {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
        wallclock = self._wallclock(frame_number)

        # Start from background
        canvas = Image.fromarray(
            np.clip(self.background / self.peak_lum * 255, 0, 255).astype(np.uint8)
        )
        draw = ImageDraw.Draw(canvas)

        # ── left: frame + SMPTE timecode ─────────────────────────────────────
        self._draw_text_with_box(draw, (self.margin, self.margin), smpte)

        # ── right: wall-clock time ────────────────────────────────────────────
        bbox = draw.textbbox((0, 0), wallclock, font=self.font)
        tw = bbox[2] - bbox[0]
        self._draw_text_with_box(
            draw, (self.width - self.margin - tw, self.margin), wallclock
        )

        # ── centre: dial ──────────────────────────────────────────────────────
        self._draw_dial(draw, frame_number)

        # Convert back to linear-light nits
        result = np.asarray(canvas, dtype=np.float64) / 255.0
        result = np.where(
            result <= 0.04045, result / 12.92, ((result + 0.055) / 1.055) ** 2.4
        )
        return result * self.peak_lum