"""2D segment geometry: bounding boxes and Liang-Barsky clipping."""

from typing import Optional, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def build_bounding_box(p1: Point, p2: Point) -> BBox:
    """Axis-aligned bounding box of a segment.

    Args:
        p1 (Point): First endpoint (x, y).
        p2 (Point): Second endpoint (x, y).

    Returns:
        BBox: (x0, y0, x1, y1) with x0 <= x1 and y0 <= y1.
    """
    return (
        min(p1[0], p2[0]), min(p1[1], p2[1]),
        max(p1[0], p2[0]), max(p1[1], p2[1]),
    )


def liang_barsky(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> Optional[Tuple[Point, Point]]:
    """Clip a segment to an axis-aligned rectangle.

    Args:
        x1, y1, x2, y2: Segment endpoints.
        xmin, ymin, xmax, ymax: Rectangle bounds.

    Returns:
        Optional[Tuple[Point, Point]]: The clipped segment's entry and
            exit points in travel order (p1 -> p2), or None when the
            segment misses the rectangle entirely.

    Raises:
        ValueError: If the rectangle bounds are inverted.
    """
    if xmax < xmin or ymax < ymin:
        raise ValueError("rectangle bounds must satisfy min <= max")
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - xmin), (dx, xmax - x1),
        (-dy, y1 - ymin), (dy, ymax - y1),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    return (
        (x1 + t0 * dx, y1 + t0 * dy),
        (x1 + t1 * dx, y1 + t1 * dy),
    )