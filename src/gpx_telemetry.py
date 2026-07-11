import gpxpy
import gpxpy.gpx

def parse_gpx_to_nodes(file_path):
    """
    Reads a GPX file and extracts coordinates and timestamps.
    Returns a list of dicts: [{'coords': (lon, lat), 'time': datetime}, ...]
    """
    with open(file_path, 'r') as f:
        gpx = gpxpy.parse(f)
        
    nodes = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                nodes.append({
                    'coords': (point.longitude, point.latitude),
                    'time': point.time # timezone-aware datetime object
                })
    return nodes