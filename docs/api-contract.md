# API contract — fall detection service

The backend should treat this service as a black box: send a video, get a decision. No
Python import, no shared database, no shared model files.

```
POST /analyze        multipart video  -> fall decision (JSON)
GET  /health         liveness + model loaded?
GET  /model          model metadata and threshold
```

Schema version: **1.0**. Additive fields may appear within a minor version; breaking
changes bump the major version.

## Run it

```powershell
# local
uvicorn src.inference.service:app --host 0.0.0.0 --port 8100

# docker
docker build -t elderlife-fall .
docker run --rm -p 8100:8100 elderlife-fall
```

Interactive docs are served at `http://localhost:8100/docs`.

## `GET /health`

```json
{ "status": "ok", "model_loaded": true, "schema_version": "1.0" }
```

`model_loaded` is `false` until the first analysis or `/model` call. Use this endpoint for
container liveness only — it does not imply the model was exercised.

## `GET /model`

```json
{
  "name": "elderlife-fall-tcn",
  "version": "1.0.0",
  "architecture": "FallTCN(width=32, dilations=(1,2,4,8), layernorm)",
  "decision_threshold": 0.5,
  "window_seconds": 3.2,
  "pose_fps": 15.0,
  "trained_events": { "events": 70, "falls": 30, "adl": 40 }
}
```

## `POST /analyze`

`multipart/form-data` with a single field named `file`.

Query parameter `layout`:

| value | meaning |
|---|---|
| `camera` (default) | ordinary RGB frames |
| `urfd` | URFD side-by-side composite (depth left, RGB right); the service crops it |

`POST /analyze/urfd` is a shortcut for `POST /analyze?layout=urfd`.

### Example

```bash
curl -X POST "http://localhost:8100/analyze?layout=urfd" \
     -F "file=@data/raw/urfd/cam0/fall-01-cam0.mp4"
```

### Response `200`

```json
{
  "schema_version": "1.0",
  "is_fall": true,
  "score": 0.999,
  "threshold": 0.5,
  "confidence": "high",
  "onset_s": 1.6,
  "windows": [
    { "start_s": 0.0, "end_s": 3.2, "score": 0.02 },
    { "start_s": 0.8, "end_s": 4.0, "score": 0.91 },
    { "start_s": 1.6, "end_s": 4.8, "score": 0.999 }
  ],
  "frames": 159,
  "fps": 30.0,
  "duration_s": 5.3,
  "pose_coverage": 0.75,
  "latency_ms": 2148.0,
  "model": {
    "name": "elderlife-fall-tcn",
    "version": "1.0.0",
    "architecture": "FallTCN(width=32, dilations=(1,2,4,8), layernorm)",
    "decision_threshold": 0.5,
    "window_seconds": 3.2,
    "pose_fps": 15.0,
    "trained_events": { "events": 70, "falls": 30, "adl": 40 }
  }
}
```

### Field semantics

| field | use it for |
|---|---|
| `is_fall` | the decision at the documented threshold |
| `score` | the highest window score in the clip (0–1) |
| `confidence` | coarse bucket (`high` ≥ 0.8, `medium` ≥ 0.6, else `low`). Presentation only — do not drive logic off it |
| `onset_s` | start of the most fall-like window; show this to the carer |
| `windows` | the evidence timeline; lets an operator see *why* |
| `pose_coverage` | fraction of frames where the person was tracked. Below ~0.3 the answer is unreliable |
| `latency_ms` | server-side analysis time for this request |

### Errors

| status | meaning |
|---|---|
| `400` | empty upload |
| `413` | larger than 200 MB |
| `422` | the bytes are not a decodable video |
| `503` | checkpoint not found on the server |

## Integration guidance

1. **One tracker per camera.** `src/inference/alerting.py::FallEventTracker` turns window
   scores into alerts with a debounce and a cooldown. It is per-stream state; two cameras
   must not share one.
2. **Do not alert on `is_fall` alone.** A single high window can be a fast sit-down. The
   tracker exists to absorb that.
3. **Latency is seconds per clip, not milliseconds.** For a live camera, cut 3-second
   windows and call the service as they arrive, or port `FallDetector` in-process; do not
   re-upload an ever-growing buffer.
4. **Store the model metadata with every alert.** `model.version` and `threshold` belong
   in the alert record, otherwise historical alerts cannot be interpreted after a model
   update.
5. **Privacy.** Uploads are written to a temporary file and deleted after analysis. The
   service never returns images. If the backend logs request bodies, it is logging video.
