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

    def render(self, t):
        img = np.zeros((self.height, self.width, 3), dtype=np.float64)

        # Background gradient
        bg = 0.05 + 0.05 * np.sin(2 * np.pi * t * 0.1)
        img[:] = bg

        discs = self.cfg["scene"]["discs"]

        for d in discs:
            cx = d["radius"] * np.sin(2 * np.pi * d["speed_x"] * t)
            cy = d["radius"] * np.cos(2 * np.pi * d["speed_y"] * t)

            dx = self.X - cx
            dy = self.Y - cy
            r = np.sqrt(dx*dx + dy*dy)

            mask = r < d["size"]

            theta = np.arctan2(dy, dx) + t * d["rotation_speed"] * 2*np.pi

            # Radial stripe texture
            texture = 0.5 + 0.5 * np.sin(20 * theta)

            # Sphere normal
            nz = np.sqrt(np.clip(d["size"]**2 - r*r, 0, None)) / d["size"]
            nx = dx / d["size"]
            ny = dy / d["size"]

            # Moving light
            Lx = np.cos(2*np.pi*t*0.2)
            Ly = np.sin(2*np.pi*t*0.2)
            Lz = 0.7

            lambert = np.clip(nx*Lx + ny*Ly + nz*Lz, 0, 1)

            # Specular
            spec = np.clip(lambert, 0, 1) ** 32

            lum = (
                texture * lambert * 0.5 +
                spec * 2.0
            )

            lum *= self.peak_lum

            color = np.array(d["color"], dtype=np.float64)

            for c in range(3):
                img[..., c][mask] = lum[mask] * color[c]

        return img