# Gophamet

**Gophamet** is a video quality metrics tool from the **Gopha** toolset, purpose-built for evaluating **Frame Rate Conversion (FRC)** solutions. It compares a reference video against a converted video and produces per-frame quality metrics, error visualisation videos, and an HTML report.

## Background

The Gopha workflow generates identical YUV test sequences at both 50p and 60p using the companion tool **gopha-gen**. These sequences feature green balls bouncing across the screen over a colour-cycling plasma background, and are engineered so that every Nth frame (where N is the frame rate) is an exact MD5-level match between the two rates.

The typical pipeline is:

1. Encode the 50p YUV sequence to H.264 and HEVC transport streams (`.TS`) at 22 Mbps CFR/CBR.
2. Feed those into your Frame Rate Converter to produce 60i or 60p outputs at matching stream properties.
3. Use **Gophamet** to compare the FRC output against the original generated 60p YUV reference.

Because all motion in the test sequences is linear, a perfect FRC with ideal motion estimation should reproduce the 60p reference exactly.

## Requirements

- **Python 3.8+**
- **FFmpeg** on your `PATH` (with support for `libvmaf`, `psnr`, `ssim` filters)
- Optional but recommended: FFmpeg built with the **zscale** filter (from the `zimg` library) for higher-quality colour normalisation. Gophamet will fall back to `scale` + `setparams` if `zscale` is unavailable.

## Files

| File | Purpose |
|---|---|
| `gophamet.py` | Main analysis script |
| `config.json` | All runtime configuration (resolution, pixel formats, metrics toggles, etc.) |

## Configuration (`config.json`)

Edit `config.json` in the working directory before running. Key fields:

| Field | Description | Default |
|---|---|---|
| `size` | Frame dimensions (e.g. `"1920x1080"`) | `"1920x1080"` |
| `rate` | Target frame rate of both reference and converted | `60` |
| `reference.pixel_format` | Pixel format of the reference input | `"yuv444p"` |
| `converted.pixel_format` | Pixel format of the converted input | `"yuv420p"` |
| `metric_processing.pixel_format` | Internal format used during metric computation | `"yuv444p"` |
| `color.*` | Colour space normalisation settings (range, matrix, primaries, transfer, zscale preference) | BT.709 TV range |
| `rewrap.enabled` | Rewrap both inputs to lossless FFV1 mezzanines before analysis (recommended) | `true` |
| `cycle_length` | Number of frames per phase cycle for cycle analysis (typically equal to `rate`) | `60` |
| `metrics.*` | Toggle individual metrics on/off: `vmaf`, `ssim`, `psnr`, `temporal_ssim`, `cycle_analysis` | all `true` |
| `error_video.*` | Toggle error visualisation outputs: `difference`, `amplified_difference`, `heatmap` | all `true` |
| `output_directory` | Where all results are written | `"results"` |
| `vmaf_model_path` | Custom VMAF model path, or `null` for the FFmpeg default | `null` |

## Usage

```bash
python gophamet.py -r <reference_video> -c <converted_video>
```

**Arguments:**

| Flag | Description |
|---|---|
| `-r`, `--reference` | Path to the reference video (raw `.yuv`, `.y4m`, or any container FFmpeg can read) |
| `-c`, `--converted` | Path to the FRC-converted video to evaluate |

**Example:**

```bash
python gophamet.py -r ref_60p.yuv -c frc_output_60p.ts
```

For raw YUV inputs (`.yuv`, `.iyuv`, `.i420`, `.nv12`), the `size`, `rate`, and corresponding `pixel_format` from `config.json` are used to interpret the file. Container formats and `.y4m` files are read directly.

## Outputs

All outputs land in the directory specified by `output_directory` (default: `results/`).

### Metric Data

| File | Contents |
|---|---|
| `vmaf.json` | Raw VMAF JSON output from libvmaf |
| `vmaf.csv` | Per-frame VMAF scores |
| `ssim.csv` | Per-frame SSIM scores |
| `ssim.log` | Raw ffmpeg SSIM log |
| `psnr.csv` | Per-frame PSNR and MSE (avg, Y, U, V) |
| `psnr.log` | Raw ffmpeg PSNR log |
| `temporal_ssim_conv.csv` | Per-frame temporal SSIM (frame-to-frame consistency of the converted video) |
| `cycle_analysis.csv` | Mean MSE per phase position within the frame-rate cycle |

### Error Visualisation Videos

| File | Description |
|---|---|
| `error_difference.mp4` | Grayscale absolute difference between reference and converted |
| `error_amplified.mp4` | Amplified (×8) grayscale difference for spotting subtle errors |
| `error_heatmap.mp4` | Colour-mapped heatmap (blue → green → red) of amplified error |

### Report

| File | Description |
|---|---|
| `report.html` | Interactive HTML report with Chart.js graphs for VMAF, SSIM, PSNR over time, and cycle-phase MSE |

Open `results/report.html` in any browser to view the report. The CSV files can be imported directly into Excel for further analysis.

## Processing Pipeline

When you run Gophamet, the following happens in order:

1. **Mezzanine rewrap** (if enabled) — both inputs are re-encoded to lossless FFV1 `.mkv` with deterministic CFR timestamps, eliminating container timing issues.
2. **Colour normalisation** — both streams are converted to a common pixel format, colour space, and range via `zscale` (or `scale`+`setparams` fallback).
3. **Metric computation** — VMAF, SSIM, PSNR, and Temporal SSIM are each computed via FFmpeg filter graphs.
4. **Cycle analysis** — MSE values are bucketed by phase position within the frame-rate cycle to reveal periodic FRC artefact patterns.
5. **Error video generation** — difference, amplified difference, and heatmap videos are rendered.
6. **Report generation** — all results are compiled into `report.html`.

## Tips

- **Cycle analysis** is particularly useful for spotting periodic artefacts introduced by FRC — if certain phase positions consistently show higher MSE, the converter is struggling with those interpolated frames.
- **Temporal SSIM** measures frame-to-frame consistency of the converted output only. A drop indicates temporal artefacts like judder or blending glitches.
- Keep `rewrap.enabled: true` unless you're certain both inputs already have clean, deterministic timestamps.