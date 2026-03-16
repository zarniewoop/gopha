import argparse
from config import load_config
from timebase import Timebase
from scenes.procedural_discs import ProceduralDiscsScene
from yuv import YUVWriter
import os

def main():
    parser = argparse.ArgumentParser(description="FRCGen v2")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    width = config["video"]["width"]
    height = config["video"]["height"]
    fps = config["video"]["fps"]
    duration = config["video"]["duration"]
    total_frames = int(fps * duration)

    tb = Timebase(
        fps=fps,
        anchor_period=config["timebase"]["anchor_period"],
        anchor_snap=0.5 / fps
    )

    scene = ProceduralDiscsScene(width, height, config)

    writer = YUVWriter(config)

    print("Rendering...")
    for i in range(total_frames):
        t = tb.time_for_frame(i)
        rgb = scene.render(t)
        writer.write_frame(rgb)

        if i % fps == 0:
            print(f"Rendered {i}/{total_frames}")

    writer.close()
    print("Done.")

if __name__ == "__main__":
    main()