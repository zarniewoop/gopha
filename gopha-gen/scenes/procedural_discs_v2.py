import numpy as np


class ProceduralDiscsSceneV2:
    """
    Full-gamut HSV plasma background with slotted rotating discs.

    The plasma is generated in HSV space: hue sweeps the entire colour
    wheel via low-frequency sine waves while saturation stays high,
    guaranteeing vivid, deeply-saturated colours instead of washed-out
    mid-tones.

    Config keys (under scene:):
      plasma_seed       – RNG seed (default: 42)
      plasma_hue_scale  – controls hue wrap density (default: 0.15)
      plasma_sat_base   – minimum saturation floor (default: 0.85)
      plasma_val_base   – midpoint brightness (default: 0.58)
      discs             – list of disc descriptors:
        color             – [R, G, B] base colour (0–1)
        size              – disc radius in NDC
        radius            – orbit radius for bounce
        speed_x / speed_y – oscillation frequencies
        rotation_speed    – disc spin (turns/sec)
        n_slots           – number of radial slot cutouts (default: 6)
        slot_width        – angular fraction cut per slot (default: 0.25)
    """

    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.cfg = config
        self.peak_lum = config["hdr"]["peak_luminance"]

        x = np.linspace(-1, 1, width, dtype=np.float64)
        y = np.linspace(-1, 1, height, dtype=np.float64)
        self.X, self.Y = np.meshgrid(x, y)

        scene_cfg = config.get("scene", {})
        seed = scene_cfg.get("plasma_seed", 42)
        rng = np.random.default_rng(seed)

        # Tuning knobs (configurable)
        self.hue_scale = scene_cfg.get("plasma_hue_scale", 0.15)
        self.sat_base = scene_cfg.get("plasma_sat_base", 0.85)
        self.val_base = scene_cfg.get("plasma_val_base", 0.58)

        # ---- Hue waves: few, low-frequency → broad colour regions ----
        N_HUE = 6
        self.hue_freq_x = rng.uniform(0.5, 3.5, N_HUE)
        self.hue_freq_y = rng.uniform(0.5, 3.5, N_HUE)
        self.hue_freq_t = rng.uniform(0.05, 0.35, N_HUE)
        self.hue_phase = rng.uniform(0, 2 * np.pi, N_HUE)
        self.hue_amp = rng.uniform(0.6, 1.4, N_HUE)

        # ---- Saturation modulation (gentle ripple, stays high) ----
        N_SAT = 3
        self.sat_freq_x = rng.uniform(1.0, 4.0, N_SAT)
        self.sat_freq_y = rng.uniform(1.0, 4.0, N_SAT)
        self.sat_freq_t = rng.uniform(0.02, 0.15, N_SAT)
        self.sat_phase = rng.uniform(0, 2 * np.pi, N_SAT)

        # ---- Value / brightness modulation ----
        N_VAL = 4
        self.val_freq_x = rng.uniform(0.5, 3.0, N_VAL)
        self.val_freq_y = rng.uniform(0.5, 3.0, N_VAL)
        self.val_freq_t = rng.uniform(0.03, 0.20, N_VAL)
        self.val_phase = rng.uniform(0, 2 * np.pi, N_VAL)

        # ---- Disc motion phase offsets ----
        discs = scene_cfg.get("discs", [])
        self.disc_phase_x = rng.uniform(0, 2 * np.pi, len(discs))
        self.disc_phase_y = rng.uniform(0, 2 * np.pi, len(discs))

    # ------------------------------------------------------------------
    # HSV → RGB  (vectorised, all inputs are 2-D [H, W] arrays in [0,1])
    # ------------------------------------------------------------------

    def _hsv_to_rgb(self, h, s, v):
        h6 = (h % 1.0) * 6.0
        sector = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)

        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        tv = v * (1.0 - s * (1.0 - f))

        rgb = np.empty((self.height, self.width, 3), dtype=np.float64)

        m = sector == 0; rgb[m, 0] = v[m];  rgb[m, 1] = tv[m]; rgb[m, 2] = p[m]
        m = sector == 1; rgb[m, 0] = q[m];  rgb[m, 1] = v[m];  rgb[m, 2] = p[m]
        m = sector == 2; rgb[m, 0] = p[m];  rgb[m, 1] = v[m];  rgb[m, 2] = tv[m]
        m = sector == 3; rgb[m, 0] = p[m];  rgb[m, 1] = q[m];  rgb[m, 2] = v[m]
        m = sector == 4; rgb[m, 0] = tv[m]; rgb[m, 1] = p[m];  rgb[m, 2] = v[m]
        m = sector == 5; rgb[m, 0] = v[m];  rgb[m, 1] = p[m];  rgb[m, 2] = q[m]

        return rgb

    # ------------------------------------------------------------------
    # Plasma background
    # ------------------------------------------------------------------

    def _plasma(self, t):
        """HSV-based plasma that sweeps the full colour gamut."""
        X, Y = self.X, self.Y

        # ---- Hue: sum of sine waves, then wrap to [0, 1] ----
        hue = np.zeros_like(X)
        for i in range(len(self.hue_freq_x)):
            hue += self.hue_amp[i] * np.sin(
                self.hue_freq_x[i] * np.pi * X
                + self.hue_freq_y[i] * np.pi * Y
                + self.hue_freq_t[i] * 2.0 * np.pi * t
                + self.hue_phase[i]
            )
        # Scale controls how many times we wrap through the hue wheel
        hue = (hue * self.hue_scale) % 1.0

        # ---- Saturation: high floor with gentle spatial ripple ----
        sat_raw = np.zeros_like(X)
        for i in range(len(self.sat_freq_x)):
            sat_raw += np.sin(
                self.sat_freq_x[i] * np.pi * X
                + self.sat_freq_y[i] * np.pi * Y
                + self.sat_freq_t[i] * 2.0 * np.pi * t
                + self.sat_phase[i]
            )
        sat_range = 1.0 - self.sat_base
        sat = self.sat_base + sat_range * np.tanh(sat_raw)  # ≈ [0.70, 1.0]

        # ---- Value: broader swing for dark pockets and bright spots ----
        val_raw = np.zeros_like(X)
        for i in range(len(self.val_freq_x)):
            val_raw += np.sin(
                self.val_freq_x[i] * np.pi * X
                + self.val_freq_y[i] * np.pi * Y
                + self.val_freq_t[i] * 2.0 * np.pi * t
                + self.val_phase[i]
            )
        val_swing = min(self.val_base, 1.0 - self.val_base)
        val = self.val_base + val_swing * np.tanh(val_raw * 0.45)

        rgb = self._hsv_to_rgb(hue, sat, val)
        rgb *= self.peak_lum
        return rgb

    # ------------------------------------------------------------------
    # Disc motion
    # ------------------------------------------------------------------

    def _bounce_pos(self, t, radius, speed_x, speed_y, phase_x, phase_y):
        """Smooth-step bouncing position with ease-in/ease-out."""
        raw_x = np.sin(2 * np.pi * speed_x * t + phase_x)
        raw_y = np.cos(2 * np.pi * speed_y * t + phase_y)

        def smoothstep(v):
            u = (v + 1.0) * 0.5
            u = np.clip(u, 0, 1)
            s = u * u * (3 - 2 * u)
            return s * 2.0 - 1.0

        return smoothstep(raw_x) * radius, smoothstep(raw_y) * radius

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, t, frame_index=None):
        img = self._plasma(t)

        discs = self.cfg["scene"].get("discs", [])

        for idx, d in enumerate(discs):
            cx, cy = self._bounce_pos(
                t,
                d["radius"],
                d["speed_x"],
                d["speed_y"],
                self.disc_phase_x[idx],
                self.disc_phase_y[idx],
            )

            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx * dx + dy * dy)
            disc_size = d["size"]

            # ---- Radial slots (pie-wedge angular cutouts) ----
            n_slots = d.get("n_slots", d.get("n_gaps", 6))
            slot_width = d.get("slot_width", d.get("gap_width", 0.25))
            rotation = t * d.get("rotation_speed", 0.5) * 2.0 * np.pi
            theta = np.arctan2(dy, dx) + rotation
            theta_norm = (theta % (2.0 * np.pi)) / (2.0 * np.pi)
            slot_pos = (theta_norm * n_slots) % 1.0
            in_slot = slot_pos < slot_width

            # Within disc radius AND not inside a slot → opaque disc pixel
            disc_mask = (r < disc_size) & (~in_slot)

            # ---- Pseudo-3D shading ----
            r_safe = np.minimum(r, disc_size - 1e-8)
            nz = np.sqrt(np.clip(disc_size**2 - r_safe**2, 0, None)) / disc_size
            nx_n = dx / (disc_size + 1e-8)
            ny_n = dy / (disc_size + 1e-8)

            # Slowly orbiting directional light
            Lx = np.cos(2.0 * np.pi * t * 0.12)
            Ly = np.sin(2.0 * np.pi * t * 0.12)
            Lz = 0.75
            L_len = np.sqrt(Lx**2 + Ly**2 + Lz**2)
            Lx, Ly, Lz = Lx / L_len, Ly / L_len, Lz / L_len

            lambert = np.clip(nx_n * Lx + ny_n * Ly + nz * Lz, 0.0, 1.0)
            spec = lambert ** 48

            # Subtle concentric ring texture
            r_norm = r / disc_size
            texture = 0.75 + 0.25 * np.sin(8.0 * r_norm * np.pi)

            lum = (texture * lambert * 0.55 + spec * 1.8) * self.peak_lum
            color = np.array(d["color"], dtype=np.float64)

            # Later discs naturally occlude earlier ones and the background
            for c in range(3):
                img[..., c][disc_mask] = lum[disc_mask] * color[c]

        return img