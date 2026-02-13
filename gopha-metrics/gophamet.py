#!/usr/bin/env python3

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


############################################################
# Log Parsers
############################################################

def parse_psnr_log(path):
    frames = []
    with open(path) as f:
        for line in f:
            m = re.search(r"n:(\d+).*mse_avg:([\d\.]+).*psnr_avg:([\d\.]+)", line)
            if m:
                frames.append({
                    "frame": int(m.group(1)),
                    "mse": float(m.group(2)),
                    "psnr": float(m.group(3))
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

def compute_temporal_ssim(video, out_log, analysis_fmt, size, rate):
    filtergraph = f"""
    format={analysis_fmt},scale={size},fps={rate},
    split[a][b];
    [b]trim=start_frame=1,setpts=PTS-STARTPTS[b1];
    [a][b1]ssim=stats_file={out_log}
    """

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-filter_complex", filtergraph,
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
        buckets[phase].append(row["mse"])

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
<li>Mean PSNR: {summary.get('mean_psnr', 'N/A')}</li>
<li>Mean MSE: {summary.get('mean_mse', 'N/A')}</li>
<li>Mean Temporal SSIM (Conv): {summary.get('mean_temporal_ssim_conv', 'N/A')}</li>
</ul>

<h2>VMAF Over Time</h2>
<canvas id="vmafChart"></canvas>

<h2>SSIM Over Time</h2>
<canvas id="ssimChart"></canvas>

<h2>PSNR Over Time</h2>
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
            label: 'PSNR',
            data: {js_array(datasets.get("psnr", []), "psnr")},
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
    parser = argparse.ArgumentParser(prog="refgen")
    parser.add_argument("-r", "--reference", required=True)
    parser.add_argument("-c", "--converted", required=True)
    args = parser.parse_args()

    config = load_config()

    size = config["size"]
    rate = config["rate"]
    ref_pix_fmt = config["reference_pixel_format"]
    analysis_fmt = config["analysis_pixel_format"]
    cycle_length = config["cycle_length"]
    out_dir = config["output_directory"]

    ensure_dir(out_dir)

    datasets = {}
    summary = {}

    ########################################################
    # Build Common Filter Graph Prefix
    ########################################################

    def build_filter(metric_filter):
        return f"""
        [0:v]format={analysis_fmt},scale={size},fps={rate}[ref];
        [1:v]format={analysis_fmt},scale={size},fps={rate}[conv];
        [ref][conv]{metric_filter}
        """

    ########################################################
    # VMAF
    ########################################################

    if config["metrics"]["vmaf"]:
        vmaf_json = os.path.join(out_dir, "vmaf.json")

        filtergraph = build_filter(
            f"libvmaf=log_path={vmaf_json}:log_fmt=json"
        )

        cmd = [
            "ffmpeg", "-y",
            "-s", size,
            "-pix_fmt", ref_pix_fmt,
            "-r", str(rate),
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        vmaf_data = parse_vmaf_json(vmaf_json)
        write_csv(os.path.join(out_dir, "vmaf.csv"), ["frame", "vmaf"], vmaf_data)
        datasets["vmaf"] = vmaf_data
        summary["mean_vmaf"] = round(mean([x["vmaf"] for x in vmaf_data]), 3)

    ########################################################
    # SSIM
    ########################################################

    if config["metrics"]["ssim"]:
        ssim_log = os.path.join(out_dir, "ssim.log")

        filtergraph = build_filter(
            f"ssim=stats_file={ssim_log}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-s", size,
            "-pix_fmt", ref_pix_fmt,
            "-r", str(rate),
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        ssim_data = parse_ssim_log(ssim_log)
        write_csv(os.path.join(out_dir, "ssim.csv"), ["frame", "ssim"], ssim_data)
        datasets["ssim"] = ssim_data
        summary["mean_ssim"] = round(mean([x["ssim"] for x in ssim_data]), 5)

    ########################################################
    # PSNR + MSE
    ########################################################

    if config["metrics"]["psnr"]:
        psnr_log = os.path.join(out_dir, "psnr.log")

        filtergraph = build_filter(
            f"psnr=stats_file={psnr_log}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-s", size,
            "-pix_fmt", ref_pix_fmt,
            "-r", str(rate),
            "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", filtergraph,
            "-f", "null", "-"
        ]
        run_cmd(cmd)

        psnr_data = parse_psnr_log(psnr_log)
        write_csv(os.path.join(out_dir, "psnr.csv"),
                  ["frame", "mse", "psnr"], psnr_data)

        datasets["psnr"] = psnr_data
        summary["mean_psnr"] = round(mean([x["psnr"] for x in psnr_data]), 3)
        summary["mean_mse"] = round(mean([x["mse"] for x in psnr_data]), 3)

        ####################################################
        # Cycle Analysis
        ####################################################
        if config["metrics"]["cycle_analysis"]:
            cycle_data = compute_cycle_analysis(psnr_data, cycle_length)
            write_csv(os.path.join(out_dir, "cycle_analysis.csv"),
                      ["phase", "mean_mse"], cycle_data)
            datasets["cycle"] = cycle_data

    ########################################################
    # Temporal SSIM
    ########################################################

    if config["metrics"]["temporal_ssim"]:
        temp_log = os.path.join(out_dir, "temporal_ssim_conv.log")
        compute_temporal_ssim(args.converted, temp_log,
                              analysis_fmt, size, rate)

        temp_data = parse_ssim_log(temp_log)
        write_csv(os.path.join(out_dir, "temporal_ssim_conv.csv"),
                  ["frame", "ssim"], temp_data)

        summary["mean_temporal_ssim_conv"] = round(
            mean([x["ssim"] for x in temp_data]), 5)

    ########################################################
    # Error Videos
    ########################################################

    def build_error_filter(extra):
        return f"""
        [0:v]format={analysis_fmt},scale={size},fps={rate}[ref];
        [1:v]format={analysis_fmt},scale={size},fps={rate}[conv];
        [ref][conv]blend=all_mode=difference{extra}
        """

    if config["error_video"]["difference"]:
        cmd = [
            "ffmpeg", "-y",
            "-s", size, "-pix_fmt", ref_pix_fmt, "-r", str(rate), "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", build_error_filter(""),
            "-c:v", "libx264", "-crf", "0",
            os.path.join(out_dir, "error_difference.mp4")
        ]
        run_cmd(cmd)

    if config["error_video"]["amplified_difference"]:
        cmd = [
            "ffmpeg", "-y",
            "-s", size, "-pix_fmt", ref_pix_fmt, "-r", str(rate), "-i", args.reference,
            "-i", args.converted,
            "-filter_complex", build_error_filter(",eq=contrast=5:brightness=0.1"),
            "-c:v", "libx264", "-crf", "0",
            os.path.join(out_dir, "error_amplified.mp4")
        ]
        run_cmd(cmd)

    if config["error_video"]["heatmap"]:
        cmd = [
            "ffmpeg", "-y",
            "-s", size, "-pix_fmt", ref_pix_fmt, "-r", str(rate), "-i", args.reference,
            "-i", args.converted,
            "-filter_complex",
            build_error_filter(",format=gray,pseudocolor=preset=jet"),
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


if __name__ == "__main__":
    main()