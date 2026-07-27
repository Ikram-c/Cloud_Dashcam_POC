#!/usr/bin/env python3
"""Build a static GitHub Pages demo of the CCTV Archive panel.

Extracts the panel HTML from webapp/server.py textually (no runtime
dependencies beyond the standard library) and injects a canned-data
fetch stub, so the deployed page demonstrates the full click-through
flow - folder browsing, batch progress, movement search with the
activity ring, timeline, and per-object movement list - without any
backend.

Usage:
    python scripts/build_demo_page.py --out site
"""

import argparse
import sys
from pathlib import Path

SERVER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "frame_extract" / "webapp" / "server.py"
)
HTML_START = '_INDEX_HTML = """'
HTML_END = '"""\n'

DEMO_STUB = """
<script>
const DEMO_NOTE = "Static demo - data is canned, no backend attached.";
let phase = 0;
const CANNED = {
  "/api/networks": {networks: ["ee", "o2", "three", "vodafone"],
                    default: "vodafone"},
  "/api/prefs": {prefs: {tab: "extract"}},
  "/api/browse": {path: "/data/videos", parent: "/data",
    folders: ["morning_run"],
    videos: [
      {name: "CAR01_20260720_083000_CAM01.mp4", path: "/v/1.mp4",
       size_mb: 220.4, processed: true},
      {name: "CAR01_20260720_090000_CAM01.mp4", path: "/v/2.mp4",
       size_mb: 198.1, processed: false}]},
};
function statusBody() {
  phase += 1;
  if (phase < 4) {
    return {state: "running", progress: 0.25 * phase,
      network: "vodafone", zones: {}, error: null,
      stats: {sampled: 120 * phase, blur: 6 * phase,
              underexposed: 2, overexposed: 1,
              duplicate: 9 * phase, reflectance: phase},
      current_video: "CAR01_20260720_083000_CAM01.mp4",
      video_index: 0, video_count: 2, summaries: []};
  }
  return {state: "done", progress: 1.0, network: "vodafone",
    zones: {urban_4g: 5, rural_edge: 2, no_coverage: 1},
    error: null, stats: {}, current_video: "",
    video_index: 1, video_count: 2, summaries: [
      {video: "CAR01_20260720_083000_CAM01.mp4",
       frames_sampled: 480, frames_kept: 391, rejected_blur: 24,
       rejected_exposure: 12, rejected_duplicate: 53,
       reflectance_frames: 3},
      {video: "CAR01_20260720_090000_CAM01.mp4",
       frames_sampled: 500, frames_kept: 445, rejected_blur: 11,
       rejected_exposure: 4, rejected_duplicate: 40,
       reflectance_frames: 1}]};
}
window.fetch = function (path, options) {
  let body;
  if (path === "/api/status") {
    body = statusBody();
  } else if (path === "/api/extract") {
    phase = 0;
    body = {started: true, network: "vodafone"};
  } else if (path === "/api/upload") {
    return Promise.resolve({ok: false, json: () =>
      Promise.resolve({detail: DEMO_NOTE})});
  } else {
    body = CANNED[path] || {};
  }
  return Promise.resolve({ok: true,
    json: () => Promise.resolve(body)});
};
</script>
"""


def extract_panel_html() -> str:
    """Pull the panel HTML constant out of server.py textually.

    Returns:
        str: The panel document.

    Raises:
        SystemExit: If the constant cannot be located.
    """
    text = SERVER_PATH.read_text(encoding="utf-8")
    start = text.find(HTML_START)
    if start < 0:
        raise SystemExit("panel HTML not found in server.py")
    start += len(HTML_START)
    end = text.find(HTML_END, start)
    if end < 0:
        raise SystemExit("panel HTML terminator not found")
    return text[start:end]


def build(out_dir: Path) -> Path:
    """Write the demo site.

    Args:
        out_dir (Path): Output directory.

    Returns:
        Path: The written index.html.
    """
    html = extract_panel_html()
    marker = '<script>\n"use strict";'
    if marker not in html:
        raise SystemExit("panel script marker not found")
    html = html.replace(marker, DEMO_STUB + marker)
    html = html.replace("\\\\u", "\\u")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Build the Pages demo")
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args()
    index = build(args.out)
    print(f"demo written: {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
