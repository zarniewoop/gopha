import numpy as np

class ProceduralDiscsScene:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.cfg = config
        self.peak_lum = config["hdr"]["peak_luminance"]

        x = np.linspace(-1, 1, width, dtype=np.float64)
        y = np.linspace(-1, 1, height, dtype=np.float64)
        self.X, self.Y = np.meshgrid(x, y)

        # Seed the RNG for plasma effect
        seed = config["scene"].get("plasma_seed", 42)
        rng = np.random.default_rng(seed)

        # Pre-generate random plasma parameters
        N_WAVES = 12
        self.plasma_freq_x  = rng.uniform(1.0, 6.0,  N_WAVES)
        self.plasma_freq_y  = rng.uniform(1.0, 6.0,  N_WAVES)
        self.plasma_freq_t  = rng.uniform(0.05, 0.5, N_WAVES)
        self.plasma_phase   = rng.uniform(0, 2*np.pi, N_WAVES)
        self.plasma_colors  = rng.uniform(0.0, 1.0, (N_WAVES, 3))

        # Pre-compute easing lookup for disc bounce
        # Each disc gets its own phase offset from the seed
        discs = config["scene"]["discs"]
        self.disc_phase_x = rng.uniform(0, 2*np.pi, len(discs))
        self.disc_phase_y = rng.uniform(0, 2*np.pi, len(discs))

    def _plasma(self, t):
        """Brightly coloured plasma background with per-region variance."""
        plasma = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(len(self.plasma_freq_x)):
            wave = np.sin(
                self.plasma_freq_x[i] * np.pi * self.X +
                self.plasma_freq_y[i] * np.pi * self.Y +
                self.plasma_freq_t[i] * 2 * np.pi * t +
                self.plasma_phase[i]
            )
            # Each wave contributes a saturated colour
            for c in range(3):
                plasma[..., c] += wave * self.plasma_colors[i, c]

        # Normalise to [0, 1] then boost saturation / brightness
        plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min() + 1e-8)
        plasma = np.power(plasma, 0.5)          # gamma lift – makes it vivid
        plasma *= self.peak_lum * 0.6           # scale to HDR headroom
        return plasma

    def _bounce_pos(self, t, radius, speed_x, speed_y, phase_x, phase_y):
        """
        Smooth-step bouncing position with acceleration/deceleration.
        Uses a sine wave passed through a smoothstep to give ease-in/ease-out feel.
        """
        # Raw sine oscillation [-1, 1]
        raw_x = np.sin(2 * np.pi * speed_x * t + phase_x)
        raw_y = np.cos(2 * np.pi * speed_y * t + phase_y)

        # Smoothstep: remap [-1,1] -> [0,1] -> smoothstep -> [-1,1]
        def smoothstep(v):
            u = (v + 1.0) * 0.5          # -> [0,1]
            u = np.clip(u, 0, 1)
            s = u * u * (3 - 2 * u)      # classic smoothstep
            return s * 2.0 - 1.0         # -> [-1,1]

        cx = smoothstep(raw_x) * radius
        cy = smoothstep(raw_y) * radius
        return cx, cy

    def render(self, t, frame_index=None):
        # Plasma background
        img = self._plasma(t)

        discs = self.cfg["scene"]["discs"]

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
            r = np.sqrt(dx*dx + dy*dy)

            disc_size = d["size"]

            # --- Gaps: ring-shaped cutouts inside the disc ---
            n_gaps     = d.get("n_gaps", 4)
            gap_width  = d.get("gap_width", 0.04)
            # Normalised radial position within disc [0,1]
            r_norm = r / disc_size
            # Evenly spaced rings in radial space
            ring_pos = (r_norm * n_gaps) % 1.0
            gap_mask = ring_pos < gap_width / (disc_size / n_gaps + 1e-8)

            mask = (r < disc_size) & (~gap_mask)

            theta = np.arctan2(dy, dx) + t * d["rotation_speed"] * 2 * np.pi

            # Radial stripe texture
            texture = 0.5 + 0.5 * np.sin(20 * theta)

            # Sphere normals
            nz = np.sqrt(np.clip(disc_size**2 - r*r, 0, None)) / disc_size
            nx = dx / disc_size
            ny = dy / disc_size

            # Moving light
            Lx = np.cos(2 * np.pi * t * 0.2)
            Ly = np.sin(2 * np.pi * t * 0.2)
            Lz = 0.7

            lambert = np.clip(nx*Lx + ny*Ly + nz*Lz, 0, 1)
            spec    = np.clip(lambert, 0, 1) ** 32

            lum = texture * lambert * 0.5 + spec * 2.0
            lum *= self.peak_lum

            color = np.array(d["color"], dtype=np.float64)

            # Discs naturally occlude one another because we paint in order
            for c in range(3):
                img[..., c][mask] = lum[mask] * color[c]

        return img