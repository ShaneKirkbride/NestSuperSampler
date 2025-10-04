# Nest Super Sampler

High quality frame extraction for Nest security camera recordings. The project reads a
timestamped MP4, samples the most useful frames, optionally stabilises night footage,
and exports super-resolved stills plus a CSV manifest.

## Project structure

```
.
├── NestSuperSampler.py        # Legacy entry point kept for compatibility
├── nest_super_sampler/        # Package with SOLID modules
│   ├── builders.py            # Factories to assemble the pipeline
│   ├── cli.py                 # Argument parsing and main() entry point
│   ├── config.py              # `PipelineConfig` dataclass and `SuperResAlgo`
│   ├── interfaces.py          # Protocol interfaces shared across modules
│   ├── pipeline.py            # Orchestrates reading, processing, writing
│   ├── preprocessors.py       # Glare, denoise, deblur processors
│   ├── samplers.py            # Frame sampling strategies
│   ├── sinks.py               # Disk image + metadata writer
│   ├── supersamplers.py       # Bicubic + DNN super-resolution backends
│   └── video_reader.py        # OpenCV based video reader
├── models/                    # Place DNN `.pb` models here (optional)
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

OpenCV's DNN super-resolution models (`EDSR_x4.pb`, etc.) must be downloaded into the
`models/` directory when using anything other than the bicubic fallback.

## Usage

Run the pipeline via the maintained script or the package module:

```bash
python NestSuperSampler.py path/to/input.mp4 --out ./export
# or
python -m nest_super_sampler path/to/input.mp4 --out ./export
```

For daylight footage recorded at the standard Nest camera frame rate (30 FPS), pass the
lighting and FPS hints to activate motion blur compensation:

```bash
python -m nest_super_sampler path/to/daytime.mp4 \
  --out ./export \
  --motion-comp \
  --lighting day \
  --fps-hint 30
```

To transfer sharp daytime structure into a noisy night capture, feed either a cached model or
raw daytime footage:

```bash
python -m nest_super_sampler path/to/night.mp4 \
  --out ./export \
  --lighting night \
  --motion-comp --fps-hint 15 \
  --daytime-transfer \
  --daytime-reference path/to/daytime_reference.mp4
```

Key arguments:

- `--n`: keep every _n_-th frame (defaults to `15`).
- `--scene`: add adaptive scene-change sampling.
- `--algo`: one of `edsr`, `espcn`, `fsrcnn`, `lapsrn`, or `bicubic` (fallback).
- `--scale`: super-resolution scale factor.
- `--models`: directory containing `.pb` model files (required for DNN algorithms).
- `--glare`, `--denoise`, `--deblur`: enable pre-processing stages tuned for night video.
- `--daytime-transfer`: restore night detail using statistics from daytime media supplied
  via `--daytime-reference` (video/images) or a cached `.npz` from `--daytime-model`.

Use `python -m nest_super_sampler --help` to see the full list of switches.

## Recommended parameter presets

| Scenario | Suggested Flags |
| --- | --- |
| Daytime overview recording | `--n 12 --scene --algo bicubic --scale 2 --fmt jpg --jpgq 90`
| Daytime vehicle tracking (motion deblur) | `--motion-comp --lighting day --fps-hint 30 --algo edsr --scale 2`
| Night-time with headlights/glare | `--scene --glare --glare-v 0.9 --glare-s 0.35 --glare-k 2.8 --denoise --dn-hl 10 --dn-hc 7`
| Vehicle identification (close-up) | `--n 6 --scene --algo edsr --scale 4 --models ./models --orig-too --max 300`
| Quick triage (fast export) | `--n 20 --algo bicubic --scale 2 --fmt png --max 120`

Adjust `--scene-thresh` (0.25–0.35) to control how sensitive scene change detection is, and
`--scene-gap` (15–30 frames) to avoid rapid-fire captures. When enabling denoising, keep the
search window between 21–35 for a balance between speed and smoothing.

## Extensibility

The codebase follows SOLID principles. Each component honours a small interface so you can
swap implementations (for example, by replacing `OpenCVVideoReader` with an FFmpeg-backed
reader). The `build_pipeline` helper wires together dependencies from `PipelineConfig` to the
`SuperSamplingPipeline`, keeping the CLI thin and testable.

Programmatic use of the adaptive blur compensation profile mirrors the CLI switches:

```python
from pathlib import Path

import cv2

from nest_super_sampler import (
    LightingCondition,
    PipelineConfig,
    derive_capture_profile,
    fit_daytime_detail_model,
)
from nest_super_sampler.builders import build_pipeline
from nest_super_sampler.detail_models import DaytimeDetailModel

day_model: DaytimeDetailModel = fit_daytime_detail_model(
    [cv2.imread("reference_day.jpg"), cv2.imread("reference_day_2.jpg")]
)
model_path = Path("./cached_daytime_model.npz")
day_model.save(model_path)

cfg = PipelineConfig(
    input_video=Path("night.mp4"),
    output_dir=Path("./export"),
    enable_motion_compensation=True,
    lighting=LightingCondition.NIGHT,
    capture_fps_hint=15.0,
    enable_daytime_detail_transfer=True,
    daytime_model_path=model_path,
)
profile = derive_capture_profile(cfg, {"fps": 15.0})
print("Estimated blur strength:", profile.motion_blur_strength())
pipeline = build_pipeline(cfg)
stats = pipeline.run()
```

## CSV output

After processing, `frames_metadata.csv` is written next to the exported images. Columns:

- `frame_idx`, `timestamp_s`
- `orig_w`, `orig_h`
- `sr_w`, `sr_h`
- `fps` (copied from video metadata)

## Troubleshooting

- _`OpenCV contrib (dnn_superres) not available`_: install `opencv-contrib-python` per
  `requirements.txt`.
- Missing `.pb` models: download the official OpenCV super-resolution models and place them in
  the folder specified by `--models` (defaults to `./models`).
- Memory pressure: lower `--scale` or increase `--n` to process fewer, smaller frames.
