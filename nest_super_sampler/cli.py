"""Command line entry point for the Nest Super Sampler."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .builders import build_pipeline
from .config import PipelineConfig, SuperResAlgo


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Create the CLI parser and return parsed arguments."""

    parser = argparse.ArgumentParser(description="Nest night video pipeline")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--scene-thresh", type=float, default=0.30)
    parser.add_argument("--scene-gap", type=int, default=20)
    parser.add_argument("--algo", type=str, default="edsr", choices=[a.value for a in SuperResAlgo])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--fmt", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--jpgq", type=int, default=95)
    parser.add_argument("--orig-too", action="store_true")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--strict-dnn", action="store_true")

    # Preprocessing flags
    parser.add_argument("--glare", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--deblur", action="store_true")
    parser.add_argument("--glare-v", type=float, default=0.90)
    parser.add_argument("--glare-s", type=float, default=0.40)
    parser.add_argument("--glare-tau", type=float, default=0.75)
    parser.add_argument("--glare-k", type=float, default=3.0)
    parser.add_argument("--glare-dilate", type=int, default=2)
    parser.add_argument("--dn-hl", type=int, default=7)
    parser.add_argument("--dn-hc", type=int, default=5)
    parser.add_argument("--dn-t", type=int, default=7)
    parser.add_argument("--dn-s", type=int, default=21)
    parser.add_argument("--db-alpha", type=float, default=0.6)
    parser.add_argument("--db-sigma", type=float, default=1.2)

    sharpen_group = parser.add_argument_group("Sharpening", "Post-processing sharpening options")
    sharpen_group.add_argument("--sharpen", action="store_true")
    sharpen_group.add_argument("--sh-alpha", type=float, default=0.35)
    sharpen_group.add_argument("--sh-sigma", type=float, default=1.0)
    sharpen_group.add_argument("--sh-edge", type=float, default=0.015)

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Convert CLI arguments into :class:`PipelineConfig`."""

    return PipelineConfig(
        input_video=args.input,
        output_dir=args.out,
        sample_every_n_frames=args.n,
        enable_scene_change_sampling=args.scene,
        scene_hist_threshold=args.scene_thresh,
        scene_min_gap_frames=args.scene_gap,
        upscale_factor=args.scale,
        algo=SuperResAlgo(args.algo),
        model_dir=args.models,
        image_format=args.fmt,
        jpg_quality=args.jpgq,
        write_full_size_only=not bool(args.orig_too),
        max_frames=args.max,
        fail_on_missing_superres=args.strict_dnn,
        enable_glare=args.glare,
        enable_denoise=args.denoise,
        enable_deblur=args.deblur,
        glare_v_thresh=args.glare_v,
        glare_s_thresh=args.glare_s,
        glare_knee_tau=args.glare_tau,
        glare_knee_strength=args.glare_k,
        glare_dilate_px=args.glare_dilate,
        denoise_h_luma=args.dn_hl,
        denoise_h_chroma=args.dn_hc,
        denoise_template=args.dn_t,
        denoise_search=args.dn_s,
        deblur_alpha=args.db_alpha,
        deblur_sigma=args.db_sigma,
        enable_sharpen=args.sharpen,
        sharpen_alpha=args.sh_alpha,
        sharpen_sigma=args.sh_sigma,
        sharpen_edge_clip=args.sh_edge,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point used by ``python -m nest_super_sampler`` and scripts."""

    args = parse_args(argv)
    cfg = config_from_args(args)
    pipeline = build_pipeline(cfg)
    stats = pipeline.run()
    print(f"Done. Stats: {stats}")


if __name__ == "__main__":  # pragma: no cover
    main()
