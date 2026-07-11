import networkx as nx
from datetime import timedelta
from .overlap_ind import find_overlapping
from .geometry import build_bounding_box, liang_barsky

def create_route_dag(node_data):
    """Constructs a directed path graph from a sequence of node dictionaries."""
    G = nx.DiGraph()
    for i, data in enumerate(node_data):
        # Store both coords (for geometry) and time (for video sync)
        G.add_node(i, coords=data['coords'], time=data.get('time'))
        if i > 0:
            G.add_edge(i - 1, i)
    return G

def interpolate_time(t1, t2, fraction):
    """Linearly interpolates a datetime given a fractional distance."""
    if not t1 or not t2:
        return None
    delta = t2 - t1
    return t1 + timedelta(seconds=delta.total_seconds() * fraction)

def subdivide_route_fast(G, coverage_zones):
    """Subdivides edges and interpolates timestamps at coverage boundaries."""
    rectangles = []
    rect_mapping = {} 
    
    for i, (zone_id, bbox) in enumerate(coverage_zones.items()):
        rectangles.append(bbox)
        rect_mapping[i] = {"type": "zone", "id": zone_id, "bbox": bbox}
        
    offset = len(coverage_zones)
    
    edges_list = list(G.edges)
    for i, (u, v) in enumerate(edges_list):
        bbox = build_bounding_box(G.nodes[u]['coords'], G.nodes[v]['coords'])
        rectangles.append(bbox)
        rect_mapping[offset + i] = {"type": "edge", "u": u, "v": v}
        
    overlap_sets = find_overlapping(rectangles)
    splits = {e: [] for e in edges_list}
    
    if overlap_sets:
        for overlap_group in overlap_sets:
            edges_in_group = [idx for idx in overlap_group if rect_mapping[idx]["type"] == "edge"]
            zones_in_group = [idx for idx in overlap_group if rect_mapping[idx]["type"] == "zone"]
            
            for e_idx in edges_in_group:
                for z_idx in zones_in_group:
                    edge_data = rect_mapping[e_idx]
                    zone_data = rect_mapping[z_idx]
                    
                    u = edge_data["u"]
                    v = edge_data["v"]
                    u_coords = G.nodes[u]['coords']
                    v_coords = G.nodes[v]['coords']
                    z_box = zone_data["bbox"]
                    
                    intersection = liang_barsky(
                        u_coords[0], u_coords[1], v_coords[0], v_coords[1],
                        z_box[0], z_box[1], z_box[2], z_box[3]
                    )
                    
                    if intersection:
                        p1, p2 = intersection
                        
                        # Total squared length of the original edge
                        total_dist_sq = (v_coords[0] - u_coords[0])**2 + (v_coords[1] - u_coords[1])**2
                        
                        # Calculate fractional distance [0.0 to 1.0] for temporal interpolation
                        if total_dist_sq == 0:
                            f1, f2 = 0.0, 1.0
                        else:
                            d1_sq = (p1[0] - u_coords[0])**2 + (p1[1] - u_coords[1])**2
                            d2_sq = (p2[0] - u_coords[0])**2 + (p2[1] - u_coords[1])**2
                            # Use square root for accurate linear time interpolation
                            f1 = (d1_sq ** 0.5) / (total_dist_sq ** 0.5)
                            f2 = (d2_sq ** 0.5) / (total_dist_sq ** 0.5)
                        
                        splits[(u, v)].extend([
                            (f1, p1[0], p1[1], zone_data["id"]),
                            (f2, p2[0], p2[1], zone_data["id"])
                        ])

    subdivided_G = nx.DiGraph()
    node_counter = max(G.nodes) + 1
    
    for u, v in edges_list:
        u_time = G.nodes[u]['time']
        v_time = G.nodes[v]['time']
        
        if not splits[(u, v)]:
            subdivided_G.add_node(u, coords=G.nodes[u]['coords'], time=u_time)
            subdivided_G.add_node(v, coords=G.nodes[v]['coords'], time=v_time)
            subdivided_G.add_edge(u, v, coverage="Default/Unknown")
            continue
            
        # Sort splits by fraction to preserve travel flow
        edge_splits = sorted(splits[(u, v)], key=lambda item: item[0])
        
        current_u = u
        subdivided_G.add_node(current_u, coords=G.nodes[u]['coords'], time=u_time)
        
        for frac, sx, sy, zone_id in edge_splits:
            if (sx, sy) == subdivided_G.nodes[current_u]['coords']:
                continue
                
            new_node = node_counter
            node_counter += 1
            
            # Interpolate the exact time the boundary was crossed
            split_time = interpolate_time(u_time, v_time, frac)
            
            subdivided_G.add_node(new_node, coords=(sx, sy), time=split_time)
            subdivided_G.add_edge(current_u, new_node, coverage=zone_id)
            current_u = new_node
            
        if subdivided_G.nodes[current_u]['coords'] != G.nodes[v]['coords']:
            subdivided_G.add_node(v, coords=G.nodes[v]['coords'], time=v_time)
            subdivided_G.add_edge(current_u, v, coverage="Default/Unknown")

    return subdivided_G