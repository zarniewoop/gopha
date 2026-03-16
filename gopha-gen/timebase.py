import numpy as np

class Timebase:
    def __init__(self, fps, anchor_period=1.0, anchor_snap=None):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.anchor_period = anchor_period
        self.anchor_snap = anchor_snap

    def time_for_frame(self, i):
        t = i * self.frame_duration

        if self.anchor_period is not None:
            k = round(t / self.anchor_period)
            anchor = k * self.anchor_period
            if abs(t - anchor) < self.anchor_snap:
                t = anchor

        return float(t)