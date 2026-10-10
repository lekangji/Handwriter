"""Digital rendering of a Document: on-screen/PNG/SVG preview (PRD section 8).

The vector stroke data (mm-space points from layout.py) is the canonical
representation; this module just draws it. The same points feed gcode.py,
so the PNG/SVG preview and the physical G-code output are guaranteed to
originate from identical geometry (PRD section 4/15's core requirement).
"""
import io
import svgwrite


def document_to_svg(document, page_template, filename_prefix):
    """Writes one SVG file per page: '{filename_prefix}_page{N}.svg'. Returns paths."""
    paths = []
    for i, page in enumerate(document.pages):
        fname = f"{filename_prefix}_page{i+1}.svg"
        dwg = svgwrite.Drawing(fname, size=(f"{page.width_mm}mm", f"{page.height_mm}mm"),
                                viewBox=f"0 0 {page.width_mm} {page.height_mm}")
        dwg.add(dwg.rect(insert=(0, 0), size=(page.width_mm, page.height_mm), fill='white'))

        if page_template.lined:
            y = page_template.margin_top_mm
            while y <= page.height_mm - page_template.margin_bottom_mm + 0.01:
                dwg.add(dwg.line(start=(page_template.margin_left_mm, y),
                                  end=(page.width_mm - page_template.margin_right_mm, y),
                                  stroke='#a8c8f0', stroke_width=0.2))
                y += page_template.line_spacing_mm

        for line in page.lines:
            for stroke in line.strokes:
                pts = stroke.points_mm
                if len(pts) < 2:
                    continue
                path_d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
                dwg.add(dwg.path(d=path_d, stroke='black', stroke_width=0.35,
                                  fill='none', stroke_linecap='round', stroke_linejoin='round'))
        dwg.save()
        paths.append(fname)
    return paths


def document_to_png(document, page_template, filename_prefix, dpi=150):
    """Renders each page to PNG via matplotlib, using the exact same mm-space
    stroke geometry as the SVG/G-code paths. Returns list of file paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = []
    for i, page in enumerate(document.pages):
        fig_w, fig_h = page.width_mm / 25.4, page.height_mm / 25.4
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_xlim(0, page.width_mm)
        ax.set_ylim(page.height_mm, 0)  # y-down to match our page coordinate convention
        ax.set_aspect('equal')
        ax.axis('off')

        if page_template.lined:
            y = page_template.margin_top_mm
            while y <= page.height_mm - page_template.margin_bottom_mm + 0.01:
                ax.plot([page_template.margin_left_mm, page.width_mm - page_template.margin_right_mm],
                        [y, y], color='#a8c8f0', linewidth=0.5, zorder=0)
                y += page_template.line_spacing_mm

        for line in page.lines:
            for stroke in line.strokes:
                pts = stroke.points_mm
                if len(pts) < 2:
                    continue
                ax.plot(pts[:, 0], pts[:, 1], 'k-', linewidth=1.0,
                        solid_capstyle='round', zorder=1,
                        marker='o' if (pts == pts[0]).all() else None,
                        markersize=1.0, markeredgewidth=0)

        fname = f"{filename_prefix}_page{i+1}.png"
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        paths.append(fname)
    return paths
