# Visual feature definitions

Classification of current visual / CV features. Serving still scores the
original 70-feature schema; extra keys are extracted for reports and future
training.

## Performer-related

| Feature | Calculation | Range | Meaning | Failure cases |
| --- | --- | --- | --- | --- |
| `hand_motion_energy_*` | Mean MediaPipe hand-centroid displacement | ~0–1 | Performer hand movement | No consecutive hand detections → None |
| `performer_motion_energy_*` | Mean residual after warping the previous frame by the camera affine; falls back to hand motion | 0–255 residual | Local motion not explained by camera | Tracking failed and no hands → None |
| `motion_subject_fraction` | performer / (performer + scaled camera translation) | 0–1 | Share of motion attributed to the subject | Missing if either side is unavailable |
| framing visibility / centering / size | MediaPipe pose/face/hand | 0–1 | Who is in frame and how they are placed | Missing person → centering None + flag |

## Camera-related

| Feature | Calculation | Range | Meaning | Failure cases |
| --- | --- | --- | --- | --- |
| `camera_stability_score` | `1 / (1 + t/0.02 + rot/4 + scale/0.04)` from ORB+RANSAC pairs; else `1 - min(1, pixel_energy/25)` | 0–1 | Higher = steadier camera | Textureless / dark frames fall back to the pixel proxy |
| `camera_translation_mean` | Mean \|t\| / frame diagonal | ≥0 | Global shift | None if tracking failed |
| `camera_rotation_mean` | Mean absolute rotation (degrees) | ≥0 | Roll | None if tracking failed |
| `camera_scale_change_mean` | Mean \|scale-1\| | ≥0 | Zoom / distance change | None if tracking failed |
| `camera_tracking_failed` | 1 when ORB/RANSAC could not estimate a transform | 0/1 | Explicit missingness for the new method | Always defined |

## Composition-related

Framing features (`subject_centering_score`, `subject_size_ratio`, `face_size_ratio`, visibility ratios). Unusual aspect ratios are handled by metadata flags, not by inventing a subject.

## Image-quality-related

Brightness, contrast, sharpness (Laplacian variance), blur (`1/(1+sharpness)`), colorfulness (Hasler–Süsstrunk), plus `brightness_std_full` / `contrast_std_full` for temporal variation.

## Environment / temporal

| Feature | Calculation | Notes |
| --- | --- | --- |
| `motion_energy_*` | Mean abs grayscale frame difference | Combined performer + camera + lighting |
| `motion_consistency` | `1/(1+std(pair energy))` | Low when motion is bursty |
| `shot_cut_count` / `shot_cut_frequency` / `average_shot_duration` | Large energy jumps vs median+MAD | Heuristic; sampled at ~3 fps so it is coarse |

## Edge cases

- Face/hands/body missing → None + failure flags, not a semantic zero.
- Extremely dark / static / very short videos → tracking often fails; pixel proxy or None.
- Fast cuts → high `shot_cut_frequency` and low `motion_consistency`.
- Multiple people → first detected hand/pose is used; not a multi-person tracker.
