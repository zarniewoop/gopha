# gopha-gen

`gopha-gen` is a configurable synthetic video generator for frame-rate conversion and motion-quality testing.

It currently renders:
- animated procedural discs (including occlusion and cut-out gaps)
- seeded plasma background
- deterministic timebase stepping
- Y4M output with selectable chroma and range

## Features

- **Deterministic generation** from YAML config
- **Disc motion with easing** (acceleration/deceleration feel)
- **Multiple discs** with natural occlusion by draw order
- **Disc gaps** for occlusion/interpolation artifact testing
- **Seeded plasma background** with high temporal/spatial variance
- **Y4M writer** with:
  - bit depth: 8–16
  - chroma: `444`, `422`, `420`
  - range: `limited` / `full` (plus aliases)
  - per-frame tags (`FRAME ...`)

## Project Structure

- `main.py` — entry point
- `config.py` — YAML loader
- `timebase.py` — frame time generation with anchor snap
- `scenes/procedural_discs.py` — scene renderer
- `yuv.py` — Y4M writer + RGB→YUV conversion
- `color.py` — OETF functions (`pq`, `hlg`, `bt709`)
- `example.yaml` — sample config

## Requirements

- Python 3.10+
- `numpy`
- `pyyaml`

Install dependencies:

```bash
python3 -m pip install numpy pyyaml
```

## Usage

Run from the `gopha-gen` folder:

```bash
python3 main.py --config example.yaml
```

## Configuration

Example (`example.yaml`):

```yaml
video:
  width: 1920
  height: 1080
  fps: 50
  duration: 10

timebase:
  anchor_period: 1.0

hdr:
  peak_luminance: 1000

scene:
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

output:
  filename: output_50p.y4m
  bitdepth: 10
  transfer: pq
  matrix: bt2020
  chroma: "444"      # 444 | 422 | 420
  range: "limited"   # limited | tv | studio | full | pc | jpeg
  frame_tags: "XTEST=1"
```

## Output Notes

- Output file is **Y4M** (`YUV4MPEG2`).
- Y4M always includes base stream info (size/fps/chroma).
- Range is written using an extension tag (`XCOLORRANGE=...`).
- Extra `frame_tags` are appended on each `FRAME` header line.

## Validation Rules

- `bitdepth` must be `8..16`
- `chroma` must be one of `444`, `422`, `420`
- `422` and `420` require even width
- `420` requires even height

## Tips

- Use multiple discs in `scene.discs` for overlap/occlusion tests.
- Change `plasma_seed` to generate different deterministic backgrounds.
- For strict downstream color metadata workflows, consider remux/encode to a container/codec that carries standardized color signaling (e.g., via FFmpeg).