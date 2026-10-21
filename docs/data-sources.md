# Data sources for fall detection & activity recognition

Selection criteria: **vision-based** (RGB/depth video), public, permissively licensed
for academic use, annotated at event level, small enough to work with on a 4 GB GPU.

---

## 1. UR Fall Detection Dataset (URFD) — SELECTED for MVP

| Field | Value |
|---|---|
| Sequences | 70 (30 falls + 40 ADL) |
| Modalities | RGB video, depth (PNG16), accelerometer (x-IMU), sync data |
| Cameras | cam0 = side/front view, cam1 = ceiling-mounted |
| Subjects | not publicly specified (small cohort) |
| Page | http://fenix.ur.edu.pl/~mkepski/ds/uf.html |
| Direct videos | `http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4` |
| Extracted features | `urfall-cam0-falls.csv`, `urfall-cam0-adls.csv` (per-frame posture labels) |
| License | **CC BY-NC-SA 4.0** — non-commercial academic use only |
| Citation | Kwolek, B., Kepski, M. (2014). *Human fall detection on embedded platform using depth maps and wireless accelerometer.* CMPB 117(3), 489–501. |

**Why it was chosen**
- Small (≈140 MB of MP4), downloads over plain HTTP, no registration.
- Falls *and* ADLs recorded in the same room, same protocol → the model must learn the
  posture transition, not the background.
- Labelled per-frame posture (`-1` not lying, `1` lying, `0` in-between = the fall
  motion itself) — an honest ground truth for event-level evaluation.
- Two camera viewpoints → free sanity check for generalization.

**Limitations (must be stated in the report)**
- Small number of distinct subjects → subject-wise generalization cannot be measured
  properly. This is a real weakness that must be written down, not hidden.
- Simulated, acted falls performed by young volunteers — not real elderly falls.
- Single room, controlled lighting → domain gap for real deployment.
- Class imbalance: 30 fall vs 40 ADL sequences, but far more ADL frames.

---

## 2. Alternatives (evaluated, not used in MVP)

### Le2i / Multiple Cameras Fall Dataset
- ~191 videos, 4 scenarios, 2 camera views, real indoor scenes (coffee room, home,
  office, lecture room). ADL includes sitting/lying down quickly.
- License: free for research; attribution requested.
- **Why not now:** larger download, less structured annotation.
- **Use later for:** external validation of the final model.

### UP-Fall Detection Dataset
- 17 subjects, 11 activities × 3 repetitions × 5 sensor modalities + 2 cameras.
- Acquired with an ethical committee protocol; **requires a signed request form**.
- **Why not now:** request latency + heavy download.
- **Use later for:** inter-subject generalization if the team can obtain it.

### CAUCAFall
- 10 subjects, 4 activities + falls, RGB video, 2 views, ~2 GB.
- License: research use.
- **Why not now:** 2 GB at the current link speed is too slow for Phase 0.

### SisFall / MobiFall / FallAllD
- **Wearable** accelerometer/gyroscope datasets — **not compatible** with this module.
- Listed here explicitly so the team does not mix them up: this project is
  video-based, so a fall must be visible to a camera.

---

## 3. Attribution block for the final report

> This work uses the UR Fall Detection Dataset created by Michal Kepski and Bogdan
> Kwolek at the University of Rzeszow, available at
> http://fenix.ur.edu.pl/~mkepski/ds/uf.html under CC BY-NC-SA 4.0, and described in
> Kwolek, B., Kepski, M. (2014), *Human fall detection on embedded platform using depth
> maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine,
> 117(3), 489-501.
