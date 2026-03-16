import numpy as np
from color import pq_oetf, hlg_oetf, bt709_oetf

class YUVWriter:
    def __init__(self, config):
        self.cfg = config
        self.width = config["video"]["width"]
        self.height = config["video"]["height"]
        self.bitdepth = config["output"]["bitdepth"]
        self.transfer = config["output"]["transfer"]
        self.matrix = config["output"]["matrix"]
        self.range = config["output"]["range"]

        self.file = open(config["output"]["filename"], "wb")

    def rgb_to_yuv(self, rgb):

        if self.transfer == "pq":
            rgb = pq_oetf(rgb)
        elif self.transfer == "hlg":
            rgb = hlg_oetf(rgb)
        else:
            rgb = bt709_oetf(rgb)

        if self.matrix == "bt2020":
            Kr = 0.2627
            Kb = 0.0593
        else:
            Kr = 0.2126
            Kb = 0.0722

        Kg = 1 - Kr - Kb

        Y = Kr*rgb[...,0] + Kg*rgb[...,1] + Kb*rgb[...,2]
        Cb = (rgb[...,2] - Y) / (2*(1-Kb))
        Cr = (rgb[...,0] - Y) / (2*(1-Kr))

        return Y, Cb, Cr

    def write_frame(self, rgb):

        Y, Cb, Cr = self.rgb_to_yuv(rgb)

        if self.bitdepth == 10:
            scale = 1023
            y_min, y_max = 64, 940
            c_min, c_max = 64, 960
        else:
            scale = 255
            y_min, y_max = 16, 235
            c_min, c_max = 16, 240

        Y = np.clip(Y, 0, 1)
        Cb = np.clip(Cb + 0.5, 0, 1)
        Cr = np.clip(Cr + 0.5, 0, 1)

        Y = y_min + Y * (y_max - y_min)
        Cb = c_min + Cb * (c_max - c_min)
        Cr = c_min + Cr * (c_max - c_min)

        Y = Y.astype(np.uint16)
        Cb = Cb.astype(np.uint16)
        Cr = Cr.astype(np.uint16)

        self.file.write(Y.tobytes())
        self.file.write(Cb[::2, ::2].tobytes())
        self.file.write(Cr[::2, ::2].tobytes())

    def close(self):
        self.file.close()