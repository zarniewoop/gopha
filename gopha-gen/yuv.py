import numpy as np
from fractions import Fraction
from color import pq_oetf, hlg_oetf, bt709_oetf


class YUVWriter:
    def __init__(self, config):
        self.cfg = config
        self.width = int(config["video"]["width"])
        self.height = int(config["video"]["height"])
        self.fps = config["video"]["fps"]

        out = config["output"]
        self.bitdepth = int(out["bitdepth"])
        self.transfer = str(out["transfer"]).lower()
        self.matrix = str(out["matrix"]).lower()
        self.range = str(out.get("range", "limited")).lower()
        self.chroma = str(out.get("chroma", "420")).lower()  # 444, 422, 420
        self.default_frame_tags = out.get("frame_tags", "")

        self._validate_config()

        self.file = open(out["filename"], "wb")
        self._write_y4m_header()

    def _validate_config(self):
        if self.bitdepth < 8 or self.bitdepth > 16:
            raise ValueError("output.bitdepth must be between 8 and 16")

        if self.chroma not in {"444", "422", "420"}:
            raise ValueError("output.chroma must be one of: 444, 422, 420")

        if self.chroma in {"422", "420"} and (self.width % 2 != 0):
            raise ValueError(f"YUV {self.chroma} requires even width")

        if self.chroma == "420" and (self.height % 2 != 0):
            raise ValueError("YUV 420 requires even height")

        valid_ranges = {"limited", "tv", "studio", "full", "pc", "jpeg"}
        if self.range not in valid_ranges:
            raise ValueError(
                f"Unsupported output.range '{self.range}'. "
                "Use one of: limited, tv, studio, full, pc, jpeg"
            )

    def _fps_rational(self):
        frac = Fraction(str(self.fps)).limit_denominator(1_000_000)
        return frac.numerator, frac.denominator

    def _is_full_range(self):
        return self.range in {"full", "pc", "jpeg"}

    def _range_levels(self):
        max_code = (1 << self.bitdepth) - 1

        if self._is_full_range():
            return 0, max_code, 0, max_code

        scale = 1 << (self.bitdepth - 8)
        y_min, y_max = 16 * scale, 235 * scale
        c_min, c_max = 16 * scale, 240 * scale
        return y_min, y_max, c_min, c_max

    def _dtype(self):
        return np.uint8 if self.bitdepth <= 8 else np.uint16

    def _y4m_chroma_token(self):
        if self.bitdepth <= 8:
            if self.chroma == "444":
                return "C444"
            if self.chroma == "422":
                return "C422"
            return "C420mpeg2"
        return f"C{self.chroma}p{self.bitdepth}"

    def _write_y4m_header(self):
        fps_num, fps_den = self._fps_rational()
        chroma_token = self._y4m_chroma_token()
        color_range_tag = "FULL" if self._is_full_range() else "LIMITED"

        header = (
            f"YUV4MPEG2 "
            f"W{self.width} "
            f"H{self.height} "
            f"F{fps_num}:{fps_den} "
            f"Ip "
            f"A0:0 "
            f"{chroma_token} "
            f"XCOLORRANGE={color_range_tag}\n"
        )
        self.file.write(header.encode("ascii"))

    def _format_frame_tags(self, frame_tags):
        tags = self.default_frame_tags if frame_tags is None else frame_tags
        if not tags:
            return ""

        if isinstance(tags, str):
            s = tags.strip().replace("\n", " ")
            return f" {s}" if s else ""

        if isinstance(tags, dict):
            parts = [f"{k}={v}" for k, v in tags.items()]
            return f" {' '.join(parts)}" if parts else ""

        parts = [str(x).strip().replace("\n", " ") for x in tags if str(x).strip()]
        return f" {' '.join(parts)}" if parts else ""

    def rgb_to_yuv(self, rgb):
        if self.transfer == "pq":
            rgb = pq_oetf(rgb)
        elif self.transfer == "hlg":
            rgb = hlg_oetf(rgb)
        else:
            rgb = bt709_oetf(rgb)

        if self.matrix == "bt2020":
            kr = 0.2627
            kb = 0.0593
        else:
            kr = 0.2126
            kb = 0.0722

        kg = 1.0 - kr - kb

        y = kr * rgb[..., 0] + kg * rgb[..., 1] + kb * rgb[..., 2]
        cb = (rgb[..., 2] - y) / (2.0 * (1.0 - kb))
        cr = (rgb[..., 0] - y) / (2.0 * (1.0 - kr))
        return y, cb, cr

    def _subsample_chroma(self, cb, cr):
        if self.chroma == "444":
            return cb, cr
        if self.chroma == "422":
            return cb[:, ::2], cr[:, ::2]
        # 420
        return cb[::2, ::2], cr[::2, ::2]

    def write_frame(self, rgb, frame_tags=None):
        y, cb, cr = self.rgb_to_yuv(rgb)
        y_min, y_max, c_min, c_max = self._range_levels()
        dtype = self._dtype()

        y = np.clip(y, 0.0, 1.0)
        cb = np.clip(cb + 0.5, 0.0, 1.0)
        cr = np.clip(cr + 0.5, 0.0, 1.0)

        y = y_min + y * (y_max - y_min)
        cb = c_min + cb * (c_max - c_min)
        cr = c_min + cr * (c_max - c_min)

        y = np.rint(y).astype(dtype)
        cb = np.rint(cb).astype(dtype)
        cr = np.rint(cr).astype(dtype)

        cb_p, cr_p = self._subsample_chroma(cb, cr)

        tag_suffix = self._format_frame_tags(frame_tags)
        self.file.write(f"FRAME{tag_suffix}\n".encode("ascii"))
        self.file.write(y.tobytes())
        self.file.write(cb_p.tobytes())
        self.file.write(cr_p.tobytes())

    def close(self):
        self.file.close()