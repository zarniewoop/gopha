import argparse
import subprocess
import json
import os
import sys
import csv
import re
from pathlib import Path
from statistics import mean
from collections import defaultdict

############################################################
# Utility
############################################################

def is_raw_yuv_path(p: str) -> bool:
    """
    Heuristic for "raw-like" inputs that may not have headers.
    NOTE: .y4m actually has a header, but treating it as "raw-like" is harmless
    only if you DON'T force -f rawvideo for it. Here we *do not* force rawvideo
    for .y4m (see input_args_for_path).
    """
    ext = Path(p).suffix.lower()
    return ext in [".yuv", ".iyuv", ".i420", ".nv12"]


def input_args_for_path(path: str, size: str, rate: int, pix_fmt: str):
    """
    Build ffmpeg input args for either:
    - raw YUV (needs -f rawvideo -s -pix_fmt -framerate)
    - container/headered files (just -i)
    """
    # y4m has its own header and should be read as-is.
    if Path(path).suffix.lower() == ".y4m":
        return ["-i", path]

    if is_raw_yuv_path(path):
        return [
            "-f", "rawvideo",
            "-s", size,
            "-pix_fmt", pix_fmt,
            "-framerate", str(rate),
            "-i", path,
        ]
    else:
        return ["-i", path]


def run_cmd(cmd):
    print("\n[RUNNING]")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ffmpeg_supports_filter(filter_name: str) -> bool:
    """
    Detect whether ffmpeg has a given filter (e.g., 'zscale').
    """
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return filter_name in p.stdout
    except Exception:
        return False


############################################################
# Log Parsers
############################################################

def parse_psnr_log(path):
    frames = []
    with open(path) as f:
        for line in f:
            # Typical ffmpeg psnr line contains:
            # n:0 mse_avg:... mse_y:... mse_u:... mse_v:... psnr_avg:... psnr_y:... psnr_u:... psnr_v:...
            m = re.search(
                r"n:(\d+)\s+"
                r".*?mse_avg:([\d\.]+)\s+mse_y:([\d\.]+)\s+mse_u:([\d\.]+)\s+mse_v:([\d\.]+)\s+"
                r".*?psnr_avg:([\d\.]+)\s+psnr_y:([\d\.]+)\s+psnr_u:([\d\.]+)\s+psnr_v:([\d\.]+)",
                line
            )
            if m:
                frames.append({
                    "frame": int(m.group(1)),
                    "mse_avg": float(m.group(2)),
                    "mse_y": float(m.group(3)),
                    "mse_u": float(m.group(4)),
                    "mse_v": float(m.group(5)),
                    "psnr_avg": float(m.group(6)),
                    "psnr_y": float(m.group(7)),
                    "psnr_u": float(m.group(8)),
                    "psnr_v": float(m.group(9)),
                })
    return frames


def parse_ssim_log(path):
    frames = []
    with open(path) as f:
        for line in f:
            m = re.search(r"n:(\d+).*All:([\d\.]+)", line)
            if m:
                frames.append({
                    "frame": int(m.group(1)),
                    "ssim": float(m.group(2))
                })
    return frames


def parse_vmaf_json(path):
    with open(path) as f:
        data = json.load(f)

    frames = []
    for i, frame in enumerate(data["frames"]):
        frames.append({
            "frame": i,
            "vmaf": frame["metrics"]["vmaf"]
        })
    return frames


############################################################
# Temporal SSIM
############################################################

def compute_temporal_ssim(video, out_log, normalize_chain):
    """
    Temporal SSIM is computed on the converted stream only, comparing t vs t-1.
    """
    filtergraph = f"""
    [0:v]{normalize_chain},split[a][b];
    [b]trim=start_frame=1,setpts=PTS-STARTPTS[b1];
    [a][b1]ssim=stats_file={out_log}
    """

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-filter_complex", filtergraph,
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    run_cmd(cmd)


############################################################
# Cycle Analysis
############################################################

def compute_cycle_analysis(psnr_data, cycle_length):
    buckets = defaultdict(list)

    for row in psnr_data:
        phase = row["frame"] % cycle_length
        buckets[phase].append(row["mse_avg"])

    results = []
    for phase in sorted(buckets.keys()):
        results.append({
            "phase": phase,
            "mean_mse": mean(buckets[phase])
        })

    return results


############################################################
# HTML Report
############################################################

def generate_html_report(out_dir, summary, datasets):
    html_path = os.path.join(out_dir, "report.html")

    def js_array(data, key):
        return [row[key] for row in data]

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Frame Rate Conversion Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: Arial; margin: 40px; }}
canvas {{ max-width: 100%; height: 400px; }}
h1, h2 {{ margin-top: 40px; }}
</style>
</head>
<body>

<h1>Frame Rate Conversion Analysis Report</h1>

<h2>Summary</h2>
<ul>
<li>Mean VMAF: {summary.get('mean_vmaf', 'N/A')}</li>
<li>Mean SSIM: {summary.get('mean_ssim', 'N/A')}</li>
<li>Mean PSNR (avg): {summary.get('mean_psnr', 'N/A')}</li>
<li>Mean PSNR (Y): {summary.get('mean_psnr_y', 'N/A')}</li>
<li>Mean PSNR (U): {summary.get('mean_psnr_u', 'N/A')}</li>
<li>Mean PSNR (V): {summary.get('mean_psnr_v', 'N/A')}</li>
<li>Mean MSE (avg): {summary.get('mean_mse', 'N/A')}</li>
<li>Mean Temporal SSIM (Conv): {summary.get('mean_temporal_ssim_conv', 'N/A')}</li>
</ul>

<h2>VMAF Over Time</h2>
<canvas id="vmafChart"></canvas>

<h2>SSIM Over Time</h2>
<canvas id="ssimChart"></canvas>

<h2>PSNR (avg) Over Time</h2>
<canvas id="psnrChart"></canvas>

<h2>Cycle Phase MSE</h2>
<canvas id="cycleChart"></canvas>

<script>

const frames = {js_array(datasets.get("psnr", []), "frame")};

new Chart(vmafChart, {{
    type: 'line',
    data: {{
        labels: frames,
        datasets: [{{
            label: 'VMAF',
            data: {js_array(datasets.get("vmaf", []), "vmaf")},
            borderColor: 'blue',
            fill: false
        }}]
    }}
}});

new Chart(ssimChart, {{
    type: 'line',
    data: {{
        labels: frames,
        datasets: [{{
            label: 'SSIM',
            data: {js_array(datasets.get("ssim", []), "ssim")},
            borderColor: 'green',
            fill: false
        }}]
    }}
}});

new Chart(psnrChart, {{
    type: 'line',
    data: {{
        labels: frames,
        datasets: [{{
            label: 'PSNR (avg)',
            data: {js_array(datasets.get("psnr", []), "psnr_avg")},
            borderColor: 'red',
            fill: false
        }}]
    }}
}});

new Chart(cycleChart, {{
    type: 'bar',
    data: {{
        labels: {js_array(datasets.get("cycle", []), "phase")},
        datasets: [{{
            label: 'Mean MSE per Phase',
            data: {js_array(datasets.get("cycle", []), "mean_mse")},
            backgroundColor: 'orange'
        }}]
    }}
}});

</script>

</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html)


############################################################
# MAIN
############################################################

def main():
    parser = argparse.ArgumentParser(prog="gophamet")
    parser.add_argument("-r", "--reference", required=True,
                        help="Reference input. If raw YUV, size/pix_fmt/rate from config.json are used.")
    parser.add_argument("-c", "--converted", required=True,
                        help="Converted input. If raw YUV, size/pix_fmt/rate from config.json are used.")
    args = parser.parse_args()

    config = load_config()

    size = config["size"]
    rate = int(config["rate"])

    # Raw-input pixel formats can differ
    ref_pix_fmt = config["reference"]["pixel_format"]
    conv_pix_fmt = config.get("converted", {}).get("pixel_format", ref_pix_fmt)

    analysis_fmt = config["metric_processing"]["pixel_format"]
    cycle_length = int(config["cycle_length"])
    out_dir = config["output_directory"]

    # Color normalization config
    color_cfg = config.get("color", {})
    desired_range = color_cfg.get("range", "tv")
    desired_matrix = color_cfg.get("matrix", "bt709")
    desired_primaries = color_cfg.get("primaries", "bt709")
    desired_transfer = color_cfg.get("transfer", "bt709")
    prefer_zscale = bool(color_cfg.get("use_zscale", True))

    have_zscale = ffmpeg_supports_filter("zscale")
    use_zscale = prefer_zscale and have_zscale

    ensure_dir(out_dir)

    # ---------------------------------------------
    # Mezzanine / Rewrap configuration
    # ---------------------------------------------
    rewrap_cfg = config.get("rewrap", {})
    rewrap_enabled = bool(rewrap_cfg.get("enabled", True))
    codec = rewrap_cfg.get("codec", "ffv1")
    level = str(rewrap_cfg.get("ffv1_level", 3))
    g = str(rewrap_cfg.get("g", 1))
    slicecrc = str(rewrap_cfg.get("slicecrc", 1))

    clean_ref = os.path.join(out_dir, "clean_ref.mkv")
    clean_conv = os.path.join(out_dir, "clean_converted.mkv")

    def rewrap_to_mezzanine(input_path, input_pix_fmt, output_path):
        """
        Stable CFR lossless mezzanine from either raw YUV input or container input.
        If raw, uses size/rate/pix_fmt from config to interpret the bytes.
        """
        cmd = ["ffmpeg", "-y", "-fflags", "+genpts"]
        cmd += input_args_for_path(input_path, size=size, rate=rate, pix_fmt=input_pix_fmt)
        cmd += [
            "-map", "0:v:0",
            "-an",
            # Force CFR + deterministic timestamps:
            "-vf", f"fps={rate},setpts=N/{rate}/TB,setsar=1",
            "-c:v", codec,
        ]
        if codec == "ffv1":
            cmd += ["-level", level, "-g", g, "-slicecrc", slicecrc]
        cmd += [output_path]
        run_cmd(cmd)

    # Actually run mezzanine stage if enabled
    if rewrap_enabled:
        rewrap_to_mezzanine(args.reference, ref_pix_fmt, clean_ref)
        rewrap_to_mezzanine(args.converted, conv_pix_fmt, clean_conv)
        args.reference = clean_ref
        args.converted = clean_conv

    datasets = {}
    summary = {}

    ########################################################
    # Deterministic normalization chains (timebase + colors)
    ########################################################

    # Absolute alignment: frame index -> timestamp.
    timebase_chain = f"fps={rate},setpts=N/{rate}/TB"

    def make_normalize_chain():
        if use_zscale:
            return (
                f"{timebase_chain},"
                f"format={analysis_fmt},"
                f"zscale="
                f"matrixin={desired_matrix}:transferin={desired_transfer}:primariesin={desired_primaries}:rangein={desired_range}:"
                f"matrix={desired_matrix}:transfer={desired_transfer}:primaries={desired_primaries}:range={desired_range},"
                f"setsar=1"
            )
        else:
            return (
                f"{timebase_chain},"
                f"format={analysis_fmt},"
                f"scale=in_range={desired_range}:out_range={desired_range}:"
                f"in_color_matrix={desired_matrix}:out_color_matrix={desired_matrix},"
                f"setparams=range={desired_range}:colorspace={desired_matrix}:"
                f"color_primaries={desired_primaries}:color_trc={desired_transfer},"
                f"setsar=1"
            )

    normalize_chain = make_normalize_chain()

    def build_metric_filter(metric_filter: str) -> str:
        return f"""
            [0:v]{normalize_chain}[ref];
            [1:v]{normalize_chain}[conv];
            [ref][conv]{metric_filter}
            """

    ########################################################
    # VMAF
    ########################################################

    if config["metrics"].get("vmaf", False):
        vmaf_json = os.path.join(out_dir, "vmaf.json")

        vmaf_cfg = config.get("vmaf", {})
        model_path = config.get("vmaf_model_path", None) or vmaf_cfg.get("model_path", None)
        vmaf_opts = f"log_path={vmaf_json}:log_fmt=json"
        if model_path:
            vmaf_opts += f":model_path={model_path}"

        filtergraph = build_metric_filter(f"libvmaf={vmaf_opts}")

        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-vsync", "vfr",
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        vmaf_data = parse_vmaf_json(vmaf_json)
        write_csv(os.path.join(out_dir, "vmaf.csv"), ["frame", "vmaf"], vmaf_data)
        datasets["vmaf"] = vmaf_data
        if vmaf_data:
            summary["mean_vmaf"] = round(mean([x["vmaf"] for x in vmaf_data]), 3)

    ########################################################
    # SSIM
    ########################################################

    if config["metrics"].get("ssim", False):
        ssim_log = os.path.join(out_dir, "ssim.log")
        filtergraph = build_metric_filter(f"ssim=stats_file={ssim_log}")

        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-vsync", "vfr",
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        ssim_data = parse_ssim_log(ssim_log)
        write_csv(os.path.join(out_dir, "ssim.csv"), ["frame", "ssim"], ssim_data)
        datasets["ssim"] = ssim_data
        if ssim_data:
            summary["mean_ssim"] = round(mean([x["ssim"] for x in ssim_data]), 5)

    ########################################################
    # PSNR + MSE
    ########################################################

    if config["metrics"].get("psnr", False):
        psnr_log = os.path.join(out_dir, "psnr.log")
        filtergraph = build_metric_filter(f"psnr=stats_file={psnr_log}")

        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-vsync", "vfr",
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        psnr_data = parse_psnr_log(psnr_log)
        psnr_fields = ["frame", "mse_avg", "mse_y", "mse_u", "mse_v", "psnr_avg", "psnr_y", "psnr_u", "psnr_v"]
        write_csv(os.path.join(out_dir, "psnr.csv"), psnr_fields, psnr_data)

        datasets["psnr"] = psnr_data
        if psnr_data:
            summary["mean_psnr"] = round(mean([x["psnr_avg"] for x in psnr_data]), 3)
            summary["mean_mse"] = round(mean([x["mse_avg"] for x in psnr_data]), 6)
            summary["mean_psnr_y"] = round(mean([x["psnr_y"] for x in psnr_data]), 3)
            summary["mean_psnr_u"] = round(mean([x["psnr_u"] for x in psnr_data]), 3)
            summary["mean_psnr_v"] = round(mean([x["psnr_v"] for x in psnr_data]), 3)

        if config["metrics"].get("cycle_analysis", False) and psnr_data:
            cycle_data = compute_cycle_analysis(psnr_data, cycle_length)
            write_csv(os.path.join(out_dir, "cycle_analysis.csv"), ["phase", "mean_mse"], cycle_data)
            datasets["cycle"] = cycle_data

    ########################################################
    # Temporal SSIM (Converted only)
    ########################################################

    if config["metrics"].get("temporal_ssim", False):
        temp_log = os.path.join(out_dir, "temporal_ssim_conv.log")
        compute_temporal_ssim(args.converted, temp_log, normalize_chain)

        temp_data = parse_ssim_log(temp_log)
        write_csv(os.path.join(out_dir, "temporal_ssim_conv.csv"), ["frame", "ssim"], temp_data)

        if temp_data:
            summary["mean_temporal_ssim_conv"] = round(mean([x["ssim"] for x in temp_data]), 5)

    ########################################################
    # Error Videos
    ########################################################

    def build_error_filter(extra: str) -> str:
        return f"""
            [0:v]{normalize_chain}[ref];
            [1:v]{normalize_chain}[conv];
            [ref][conv]blend=all_mode=difference{extra}
            """

    # 1) Raw Difference (Grayscale)
    if config.get("error_video", {}).get("difference", False):
        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", build_error_filter(",format=gray"),
            "-r", str(rate),
            "-c:v", "libx264", "-crf", "0",
            os.path.join(out_dir, "error_difference.mp4")
        ]
        run_cmd(cmd)

    # 2) Amplified Difference
    if config.get("error_video", {}).get("amplified_difference", False):
        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", build_error_filter(",format=gray,lut=y='clip(val*8,0,255)'"),
            "-r", str(rate),
            "-c:v", "libx264", "-crf", "0",
            os.path.join(out_dir, "error_amplified.mp4")
        ]
        run_cmd(cmd)

    # 3) Heatmap
    if config.get("error_video", {}).get("heatmap", False):
        cmd = [
            "ffmpeg", "-y",
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex",
            build_error_filter(
                ",format=gray,"
                "lut=y='clip(val*8,0,255)',"
                "format=rgb24,"
                "lutrgb="
                "r='if(lt(val,128),0,2*(val-128))':"
                "g='if(lt(val,128),2*val,2*(255-val))':"
                "b='if(lt(val,128),2*(128-val),0)'"
            ),
            "-r", str(rate),
            "-c:v", "libx264", "-crf", "0",
            os.path.join(out_dir, "error_heatmap.mp4")
        ]
        run_cmd(cmd)

    ########################################################
    # HTML Report
    ########################################################

    generate_html_report(out_dir, summary, datasets)

    print("\n✅ Full analysis complete.")
    print(f"📊 Open: {out_dir}/report.html")
    if prefer_zscale and not have_zscale:
        print("ℹ️  Note: zscale filter not found; used scale+setparams fallback for color normalization.")


if __name__ == "__main__":
    main()