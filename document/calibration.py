"""Page templates and printer/pen calibration profiles (PRD sections 7, 9, 13)."""
from dataclasses import dataclass, field
import json


@dataclass
class PageTemplate:
    width_mm: float = 210.0     # A4 portrait default
    height_mm: float = 297.0
    margin_left_mm: float = 20.0
    margin_right_mm: float = 15.0
    margin_top_mm: float = 20.0
    margin_bottom_mm: float = 20.0
    lined: bool = True
    line_spacing_mm: float = 8.0        # ruling pitch
    paragraph_spacing_mm: float = 4.0   # extra gap after a paragraph
    handwriting_size_mm: float = 6.0    # target x-height-ish scale of output

    @property
    def text_width_mm(self):
        return self.width_mm - self.margin_left_mm - self.margin_right_mm

    @property
    def text_height_mm(self):
        return self.height_mm - self.margin_top_mm - self.margin_bottom_mm


@dataclass
class PrinterProfile:
    name: str = "generic"
    bed_width_mm: float = 220.0
    bed_height_mm: float = 220.0
    usable_x_min: float = 0.0
    usable_x_max: float = 220.0
    usable_y_min: float = 0.0
    usable_y_max: float = 220.0
    dialect: str = "marlin"  # currently only affects the header/comments

    # paper placement on the bed
    paper_origin_x_mm: float = 10.0   # bed-space location of paper's top-left corner
    paper_origin_y_mm: float = 10.0
    paper_rotation_deg: float = 0.0

    # pen
    pen_x_offset_mm: float = 0.0   # offset of pen tip from toolhead reference point
    pen_y_offset_mm: float = 0.0
    pen_up_z_mm: float = 5.0
    pen_down_z_mm: float = 0.0

    # motion
    travel_speed_mm_s: float = 150.0
    writing_speed_mm_s: float = 60.0
    z_speed_mm_s: float = 30.0

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s))


DEFAULT_PRINTER = PrinterProfile()
DEFAULT_PAGE = PageTemplate()
