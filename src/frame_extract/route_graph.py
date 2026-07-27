"""Route DAG construction and subdivision at coverage-zone boundaries.

Given ordered GPX nodes and a set of axis-aligned coverage zones, the
subdivision inserts nodes exactly where the route crosses a zone
boundary, with crossing times linearly interpolated between the
bracketing fixes. Every edge of the result carries a ``coverage``
label; edges outside all zones carry NO_COVERAGE.

Zone overlap caveat: where two zones overlap, an edge segment inside
both receives whichever zone's split was applied last in traversal
order. If overlapping zones matter, make labels hierarchical upstream.
"""

import logging
from datetime import timedelta
from typing import Dict, List

import networkx as nx

from .geometry import build_bounding_box, liang_barsky
from .overlap_ind import find_overlapping

logger = logging.getLogger(__name__)

NO_COVERAGE = "no_coverage"


def create_route_dag(node_data: List[dict]) -> nx.DiGraph:
    """Build a directed path graph from ordered track nodes.

    Args:
        node_data (List[dict]): Nodes with 'coords' and 'time' keys.

    Returns:
        nx.DiGraph: Path graph; node i connects to i+1.

    Raises:
        ValueError: If fewer than two nodes are supplied.
    """
    if len(node_data) < 2:
        raise ValueError("route requires at least two track nodes")
    graph = nx.DiGraph()
    for i, data in enumerate(node_data):
        graph.add_node(i, coords=data["coords"], time=data.get("time"))
        if i > 0:
            graph.add_edge(i - 1, i)
    return graph


def interpolate_time(t1, t2, fraction: float):
    """Linearly interpolate a datetime at a fractional distance.

    Args:
        t1: Start datetime, or None.
        t2: End datetime, or None.
        fraction (float): Position along the edge in [0, 1].

    Returns:
        The interpolated datetime, or None when either end is missing.
    """
    if t1 is None or t2 is None:
        return None
    delta = t2 - t1
    return t1 + timedelta(seconds=delta.total_seconds() * fraction)


def _same_point(a, b, eps: float = 1e-12) -> bool:
    """Coordinate equality with a tolerance for clip-derived floats.

    Args:
        a: First (x, y) point.
        b: Second (x, y) point.
        eps (float): Componentwise tolerance.

    Returns:
        bool: True when the points coincide within eps.
    """
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def subdivide_route_fast(graph: nx.DiGraph, coverage_zones: Dict[str, tuple]) -> nx.DiGraph:
    """Subdivide edges and interpolate timestamps at coverage boundaries.

    The sweep-line overlap detector narrows the edge-to-zone candidate
    pairs; Liang-Barsky then computes exact crossing points, and the
    crossing time is interpolated from the fractional distance along
    the edge.

    Args:
        graph (nx.DiGraph): Route path graph from create_route_dag.
        coverage_zones (Dict[str, tuple]): zone name -> bbox
            (x0, y0, x1, y1) in the same coordinate space as node
            coords (lon, lat).

    Returns:
        nx.DiGraph: Subdivided graph; every edge carries a
            ``coverage`` attribute.
    """
    rectangles = []
    rect_mapping = {}

    for i, (zone_id, bbox) in enumerate(coverage_zones.items()):
        rectangles.append(bbox)
        rect_mapping[i] = {"type": "zone", "id": zone_id, "bbox": bbox}

    offset = len(coverage_zones)

    edges_list = list(graph.edges)
    for i, (u, v) in enumerate(edges_list):
        bbox = build_bounding_box(graph.nodes[u]["coords"], graph.nodes[v]["coords"])
        # Inflate by a hair: an axis-aligned edge has a zero-height or
        # zero-width bbox, which the sweep-line pruning would treat as
        # covering no area and silently drop. Edges are only pruning
        # candidates here; Liang-Barsky still computes exact crossings.
        pad = 1e-9
        bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        rectangles.append(bbox)
        rect_mapping[offset + i] = {"type": "edge", "u": u, "v": v}

    overlap_sets = find_overlapping(rectangles)
    splits = {e: [] for e in edges_list}

    if overlap_sets:
        for overlap_group in overlap_sets:
            edges_in_group = [
                idx for idx in overlap_group if rect_mapping[idx]["type"] == "edge"
            ]
            zones_in_group = [
                idx for idx in overlap_group if rect_mapping[idx]["type"] == "zone"
            ]

            for e_idx in edges_in_group:
                for z_idx in zones_in_group:
                    edge_data = rect_mapping[e_idx]
                    zone_data = rect_mapping[z_idx]

                    u = edge_data["u"]
                    v = edge_data["v"]
                    u_coords = graph.nodes[u]["coords"]
                    v_coords = graph.nodes[v]["coords"]
                    z_box = zone_data["bbox"]

                    intersection = liang_barsky(
                        u_coords[0], u_coords[1], v_coords[0], v_coords[1],
                        z_box[0], z_box[1], z_box[2], z_box[3]
                    )

                    if intersection:
                        p1, p2 = intersection

                        total_dist_sq = (
                            (v_coords[0] - u_coords[0]) ** 2
                            + (v_coords[1] - u_coords[1]) ** 2
                        )

                        if total_dist_sq == 0:
                            f1, f2 = 0.0, 1.0
                        else:
                            d1_sq = (
                                (p1[0] - u_coords[0]) ** 2
                                + (p1[1] - u_coords[1]) ** 2
                            )
                            d2_sq = (
                                (p2[0] - u_coords[0]) ** 2
                                + (p2[1] - u_coords[1]) ** 2
                            )
                            # Square roots for accurate linear time interpolation
                            f1 = (d1_sq ** 0.5) / (total_dist_sq ** 0.5)
                            f2 = (d2_sq ** 0.5) / (total_dist_sq ** 0.5)

                        splits[(u, v)].extend([
                            (f1, p1[0], p1[1], zone_data["id"]),
                            (f2, p2[0], p2[1], zone_data["id"]),
                        ])

    subdivided = nx.DiGraph()
    node_counter = max(graph.nodes) + 1

    for u, v in edges_list:
        u_time = graph.nodes[u]["time"]
        v_time = graph.nodes[v]["time"]

        if not splits[(u, v)]:
            subdivided.add_node(u, coords=graph.nodes[u]["coords"], time=u_time)
            subdivided.add_node(v, coords=graph.nodes[v]["coords"], time=v_time)
            subdivided.add_edge(u, v, coverage=NO_COVERAGE)
            continue

        # Sort splits by fraction to preserve travel flow
        edge_splits = sorted(splits[(u, v)], key=lambda item: item[0])

        current_u = u
        subdivided.add_node(current_u, coords=graph.nodes[u]["coords"], time=u_time)

        for frac, sx, sy, zone_id in edge_splits:
            if _same_point((sx, sy), subdivided.nodes[current_u]["coords"]):
                continue

            new_node = node_counter
            node_counter += 1

            # Interpolate the exact time the boundary was crossed
            split_time = interpolate_time(u_time, v_time, frac)

            subdivided.add_node(new_node, coords=(sx, sy), time=split_time)
            subdivided.add_edge(current_u, new_node, coverage=zone_id)
            current_u = new_node

        if not _same_point(
            subdivided.nodes[current_u]["coords"], graph.nodes[v]["coords"]
        ):
            subdivided.add_node(v, coords=graph.nodes[v]["coords"], time=v_time)
            subdivided.add_edge(current_u, v, coverage=NO_COVERAGE)

    return subdivided
