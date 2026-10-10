"""
G-code generation (PRD sections 10-12).

Pipeline: Document coords (mm, page-space, per layout.py) -> paper placement
on the bed -> pen offset -> printer coordinates -> G-code moves.

Each page becomes its own clearly-separated G-code section (PRD section 6:
the MVP cannot auto-feed paper, so the user must be told when to swap pages).
"""
import math
from dataclasses import dataclass, field
from typing import List


class GCodeValidationError(Exception):
    pass


@dataclass
class ValidationIssue:
    page: int
    kind: str      # 'outside_paper' | 'outside_printer'
    detail: str


def _paper_to_printer(x_mm, y_mm, printer):
    """Page coords -> printer bed coords: rotate about paper origin, translate
    by paper placement, then add pen offset."""
    theta = math.radians(printer.paper_rotation_deg)
    rx = x_mm * math.cos(theta) - y_mm * math.sin(theta)
    ry = x_mm * math.sin(theta) + y_mm * math.cos(theta)
    px = printer.paper_origin_x_mm + rx + printer.pen_x_offset_mm
    py = printer.paper_origin_y_mm + ry + printer.pen_y_offset_mm
    return px, py


def validate_document(document, page_template, printer) -> List[ValidationIssue]:
    issues = []
    for pi, page in enumerate(document.pages):
        for line in page.lines:
            for stroke in line.strokes:
                for x, y in stroke.points_mm:
                    if not (0 <= x <= page.width_mm and 0 <= y <= page.height_mm):
                        issues.append(ValidationIssue(pi + 1, 'outside_paper',
                            f"point ({x:.1f},{y:.1f})mm falls outside the {page.width_mm}x{page.height_mm}mm page"))
                        continue
                    if not (page_template.margin_left_mm <= x <= page.width_mm - page_template.margin_right_mm
                            and page_template.margin_top_mm <= y <= page.height_mm - page_template.margin_bottom_mm):
                        issues.append(ValidationIssue(pi + 1, 'outside_margin',
                            f"point ({x:.1f},{y:.1f})mm falls outside the configured margins"))
                    px, py = _paper_to_printer(x, y, printer)
                    if not (printer.usable_x_min <= px <= printer.usable_x_max
                            and printer.usable_y_min <= py <= printer.usable_y_max):
                        issues.append(ValidationIssue(pi + 1, 'outside_printer',
                            f"page point ({x:.1f},{y:.1f})mm maps to printer coords "
                            f"({px:.1f},{py:.1f})mm, outside the usable "
                            f"[{printer.usable_x_min},{printer.usable_x_max}]x"
                            f"[{printer.usable_y_min},{printer.usable_y_max}]mm workspace"))
    return issues


def generate_gcode(document, page_template, printer, raise_on_error=True) -> str:
    """Returns full G-code as a string, one clearly delimited section per page,
    with an M0 pause between pages for manual paper replacement."""
    if printer.pen_up_z_mm < 0 or printer.pen_down_z_mm < 0:
        raise GCodeValidationError("Z heights must be >= 0")
    if printer.pen_down_z_mm >= printer.pen_up_z_mm:
        raise GCodeValidationError("pen_down_z_mm must be less than pen_up_z_mm")

    issues = validate_document(document, page_template, printer)
    hard_errors = [i for i in issues if i.kind in ('outside_paper', 'outside_printer')]
    if hard_errors and raise_on_error:
        msg = "\n".join(f"  page {i.page}: [{i.kind}] {i.detail}" for i in hard_errors[:20])
        raise GCodeValidationError(f"{len(hard_errors)} geometry error(s), e.g.:\n{msg}")

    lines = []
    lines.append(f"; AI Handwriter G-code — printer profile '{printer.name}' ({printer.dialect})")
    lines.append(f"; {len(document.pages)} page(s), page size {page_template.width_mm}x{page_template.height_mm}mm")
    if issues:
        lines.append(f"; WARNING: {len(issues)} validation issue(s) present (see log) — proceeding anyway")
    lines.append("G21 ; millimeters")
    lines.append("G90 ; absolute positioning")
    lines.append(f"G1 F{printer.travel_speed_mm_s * 60:.0f}")
    lines.append(f"G1 Z{printer.pen_up_z_mm} F{printer.z_speed_mm_s * 60:.0f} ; pen up")

    travel_f = printer.travel_speed_mm_s * 60
    write_f = printer.writing_speed_mm_s * 60
    z_f = printer.z_speed_mm_s * 60

    for pi, page in enumerate(document.pages):
        lines.append("")
        lines.append(f"; ===== PAGE {pi + 1} of {len(document.pages)} =====")
        if pi > 0:
            lines.append(f"M117 Load page {pi + 1} of {len(document.pages)}")
            lines.append(f"M0 ; PAUSE: insert/position page {pi + 1}, then resume")

        pen_is_down = False
        for line in page.lines:
            for stroke in line.strokes:
                pts = stroke.points_mm
                if len(pts) < 2:
                    continue
                x0, y0 = _paper_to_printer(pts[0][0], pts[0][1], printer)
                if pen_is_down:
                    lines.append(f"G1 Z{printer.pen_up_z_mm} F{z_f:.0f} ; pen up")
                    pen_is_down = False
                lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_f:.0f}")
                lines.append(f"G1 Z{printer.pen_down_z_mm} F{z_f:.0f} ; pen down")
                pen_is_down = True
                for x, y in pts[1:]:
                    px, py = _paper_to_printer(x, y, printer)
                    lines.append(f"G1 X{px:.3f} Y{py:.3f} F{write_f:.0f}")
        if pen_is_down:
            lines.append(f"G1 Z{printer.pen_up_z_mm} F{z_f:.0f} ; pen up")

    lines.append("")
    lines.append("; end of document")
    return "\n".join(lines)


def calibration_pattern_gcode(printer, size_mm=20.0) -> str:
    """A simple square + crosshair at the paper origin, used to visually
    determine pen X/Y offset and writing height (PRD section 13)."""
    lines = [
        "; Calibration pattern: square + crosshair at paper origin",
        "G21", "G90",
        f"G1 Z{printer.pen_up_z_mm} F{printer.z_speed_mm_s*60:.0f}",
    ]
    travel_f = printer.travel_speed_mm_s * 60
    write_f = printer.writing_speed_mm_s * 60
    z_f = printer.z_speed_mm_s * 60
    square = [(0, 0), (size_mm, 0), (size_mm, size_mm), (0, size_mm), (0, 0)]
    cross = [((size_mm/2, size_mm/2 - 3), (size_mm/2, size_mm/2 + 3)),
             ((size_mm/2 - 3, size_mm/2), (size_mm/2 + 3, size_mm/2))]

    def pen_to(path):
        x0, y0 = _paper_to_printer(*path[0], printer)
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{travel_f:.0f}")
        lines.append(f"G1 Z{printer.pen_down_z_mm} F{z_f:.0f}")
        for x, y in path[1:]:
            px, py = _paper_to_printer(x, y, printer)
            lines.append(f"G1 X{px:.3f} Y{py:.3f} F{write_f:.0f}")
        lines.append(f"G1 Z{printer.pen_up_z_mm} F{z_f:.0f}")

    pen_to(square)
    for a, b in cross:
        pen_to([a, b])
    lines.append("; end calibration pattern")
    return "\n".join(lines)
