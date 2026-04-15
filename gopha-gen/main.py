import argparse
from config import load_config
from timebase import Timebase
from scenes import create_scene
from yuv import YUVWriter

def main():
    parser = argparse.ArgumentParser(description="FRCGen v2")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    width = int(config["video"]["width"])
    height = int(config["video"]["height"])
    fps = float(config["video"]["fps"])
    duration = float(config["video"]["duration"])
    total_frames = int(round(fps * duration))

    tb = Timebase(
        fps=fps,
        anchor_period=config.get("timebase", {}).get("anchor_period", 1.0),
        anchor_snap=0.5 / fps,
    )

    scene_name = config.get("scene", {}).get("name", "procedural_discs")
    scene = create_scene(scene_name, width, height, config)
    writer = YUVWriter(config)

    print(f"Rendering scene '{scene_name}' — {total_frames} frames at {fps} fps...")
    progress_interval = max(1, int(round(fps)))

    try:
        for i in range(total_frames):
            t = tb.time_for_frame(i)
            rgb = scene.render(t, frame_index=i)
            writer.write_frame(rgb)

            if i % progress_interval == 0:
                print(f"  Rendered {i}/{total_frames}")
    finally:
        writer.close()

    print(f"Done. Output: {config['output']['filename']}")

if __name__ == "__main__":
    main()