# gopha-gen Developer Guide: Adding New Scenes

This guide explains how to add and wire new scene generators into `gopha-gen`.

---

## 1) Scene Lifecycle in gopha-gen

Current render flow:

1. `main.py` loads YAML config
2. A scene object is created
3. For each frame:
   - `t = Timebase.time_for_frame(i)`
   - `rgb = scene.render(t)`
   - `writer.write_frame(rgb)`

A scene is responsible for generating a frame in **linear-light RGB**.

---

## 2) Scene Interface Contract

Every scene should implement:

- `__init__(self, width, height, config)`
- `render(self, t) -> np.ndarray`

### Requirements

- Return shape: `(height, width, 3)`
- Type: float array (`float32` or `float64`)
- Values: non-negative linear light (typically in nit-like scale; writer applies OETF/matrix/range)
- Determinism: if randomness is used, seed from config

---

## 3) Create a New Scene File

Add a file under `scenes/`, for example:

`scenes/motion_wedges.py`

Example skeleton:

```python
import numpy as np

class MotionWedgesScene:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.cfg = config
        self.peak_lum = config["hdr"]["peak_luminance"]

        x = np.linspace(-1, 1, width, dtype=np.float64)
        y = np.linspace(-1, 1, height, dtype=np.float64)
        self.X, self.Y = np.meshgrid(x, y)

        scene_cfg = config.get("scene", {})
        seed = scene_cfg.get("seed", 1234)
        self.rng = np.random.default_rng(seed)

    def render(self, t):
        img = np.zeros((self.height, self.width, 3), dtype=np.float64)

        # Example animated pattern
        v = 0.5 + 0.5 * np.sin(20.0 * self.X + 2.0 * np.pi * 0.5 * t)
        img[..., 0] = v * self.peak_lum
        img[..., 1] = (1.0 - v) * self.peak_lum * 0.6
        img[..., 2] = 0.2 * self.peak_lum

        return img
```

---


## 4) YAML Config Pattern

Use `scene.name` to select scene, and include scene-specific parameters under `scene`:

```yaml
scene:
  name: motion_wedges
  seed: 1234
  # scene-specific params here
```

For existing scene:

```yaml
scene:
  name: procedural_discs
  plasma_seed: 6969
  discs:
    - color: [1.0, 0.3, 0.1]
      size: 0.25
      radius: 0.6
      speed_x: 0.3
      speed_y: 0.2
      rotation_speed: 0.5
      n_gaps: 5
      gap_width: 0.03
```

---

## 5) Recommended Design Rules for New Scenes

- Generation must be deterministic; runnning the same config N times should result in the same output N times
- Generation must be identical (pixel perfect) every anchor period seconds 
- Generation of intermediate frames is supposed to be effectively acting as interpolations between the anchor periods, so the intra anchor content should not change based on frame rate OTHER than the number of 'steps'
- Keep per-frame work vectorized (`numpy`) to avoid Python pixel loops
- Precompute static grids/tables in `__init__`
- Avoid mutating global state
- Keep random behavior reproducible via config seed <<--
- Clip/sanitize values only when needed (writer also clips)
- Document scene-specific config keys in comments