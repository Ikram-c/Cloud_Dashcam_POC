# Deploying Cloud_Dashcam_POC on PythonAnywhere

Two supported routes. Route A (WSGI) uses the normal Web tab and works
on any account, including free. Route B (ASGI) uses PythonAnywhere's
experimental uvicorn sites via the `pa` CLI.

## 0. Get the code onto PythonAnywhere

Open a Bash console and either clone your GitHub repo:

```bash
git clone https://github.com/<you>/Cloud_Dashcam_POC.git
```

or upload the zip via the Files tab and `unzip` it. Then:

```bash
cd ~/Cloud_Dashcam_POC
bash deploy/pythonanywhere/setup_pythonanywhere.sh
```

This creates the `dashcam` virtualenv, installs the package with the
panel dependencies plus the `a2wsgi` adapter, and creates the data
folders (videos, extracted_frames) inside the project.

## Route A - classic Web app (WSGI, any account)

1. Web tab -> Add a new web app -> Manual configuration -> the Python
   version matching your virtualenv.
2. In the Virtualenv section enter: `/home/<you>/.virtualenvs/dashcam`
3. Open the WSGI configuration file link and replace its entire
   contents with:

```python
import sys
sys.path.insert(
    0, "/home/<you>/Cloud_Dashcam_POC/deploy/pythonanywhere",
)
from wsgi_app import application
```

4. Reload the web app. The panel is live at
   `https://<you>.pythonanywhere.com`.

## Route B - ASGI beta (uvicorn, pa CLI)

```bash
pip install --user --upgrade pythonanywhere
pa website create --domain <you>.pythonanywhere.com \
    --command '/home/<you>/.virtualenvs/dashcam/bin/uvicorn \
--app-dir /home/<you>/Cloud_Dashcam_POC/deploy/pythonanywhere \
--uds $DOMAIN_SOCKET asgi_app:app'
```

An API token must be set up first (Account -> API token). The feature
is beta: no static-file mappings and limited web UI management.

## Platform notes and limits

- `config.pythonanywhere.yaml` is used by both entry points: all
  paths are relative to the project folder, the cloud archive uses
  the offline mock, and the weather forecast is disabled so the panel never waits on outbound internet (free-tier outbound traffic is allowlist-only; illumination degrades to geometry-only automatically). Edit it to change behaviour.
- Web workers are recycled by the platform, and free accounts have
  limited CPU seconds per day. The panel's background processing
  works, but for long batches prefer running the CLI in a console or
  a Scheduled/Always-on task:
  `~/.virtualenvs/dashcam/bin/frame-extract --config config.pythonanywhere.yaml --videos videos --output extracted_frames`
- Free accounts have a 512 MB disk quota - video processing fills it
  quickly. Keep test clips short or upgrade for real footage.
- Uploads from the panel arrive in 8 MB parts, well under the
  platform's request-size limit.
- The panel has no authentication. On PythonAnywhere it is public at
  your domain: enable the Web tab's password protection (Route A;
  paid feature on some plans) or treat the deployment as a demo with
  non-sensitive footage.
