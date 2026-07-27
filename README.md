# Cloud_Dashcam_POC - panel and deployment updates

Append these sections to the main README (kept separate so this
mirror never overwrites your original README.md).

## Web control panel (updated)

`frame-extract-ui --config config.yaml` now serves a modernised panel
(port 8321): three destinations (Add footage, Options, Results) in a
fixed 49px tab bar that becomes a 260-320px sidebar on wide screens,
light and dark modes at >= 4.5:1 text contrast, 44px touch targets,
and plain-language labels throughout. Videos are selected by
clicking: a native "Choose files" picker, a "Choose a folder"
variant, and drag-and-drop copy footage to this machine in bounded
chunked uploads with a progress bar, while a server-side folder
browser (44px selectable rows, sizes, "Processed" badges) covers
footage already here. A run processes the explicit selection (staged
into a private run folder), a whole folder, or the built-in local
demo. The Options tab holds the six quality-check toggles ("Skip
blurry frames", "Skip too dark or bright", "Skip repeated frames",
"Daylight awareness", "Shadow-free snapshots", "Group by phone
signal"), the mobile network, route recording source, and sampling
choice - all auto-saved implicitly every 30 s and restored on load.
Results shows live per-frame progress ("Working on <video> - video 2
of 5", frames checked/skipped/snapshots), then a "Processing
Complete" card with per-video kept/rejected summaries and per-zone
coverage chips.

The panel API is unchanged where it existed (`create_app`,
`ExtractRequest`, `_apply_request`, `/api/status`, `/api/networks`,
`/api/extract` with 409/400 semantics) and gains `/api/browse`,
`/api/upload`, and `/api/prefs`. All existing webapp tests pass
unmodified; 12 new ones cover the additions.

## Deploying via GitHub

Three GitHub Actions workflows ship in `.github/workflows/`:

- **ci.yml** - on every push and PR: the style gate
  (`scripts/check_style.py`) plus the full offline suite on Python
  3.11 and 3.12 (installs libgl1/libglib2.0-0/libmediainfo0v5 and
  `.[metadata,ui,coverage,dev]`).
- **release.yml** - on a `v*` tag: tests, wheel + sdist attached to a
  GitHub Release, and a Docker image pushed to
  `ghcr.io/<owner>/<repo>` using only the built-in `GITHUB_TOKEN`.
- **demo-pages.yml** - a static, backend-free demo of the panel
  published to GitHub Pages on every push to main (enable Pages once:
  Settings -> Pages -> Source: GitHub Actions).

Run the released container with:

```bash
docker run -p 8321:8321 -v $PWD/data:/data ghcr.io/<owner>/<repo>
```

Footage lives in the mounted volume (`/data/videos`, outputs in
`/data/extracted_frames`); the container config binds 0.0.0.0, so
keep it behind your own network controls. Note the test-coverage
fixture now pins its timezone to UTC, fixing the July/BST failure -
CI is green from the first push.
