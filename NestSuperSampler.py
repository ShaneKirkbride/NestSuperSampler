#!/usr/bin/env python3
"""
Nest video processing:
  - Frame sampling
  - Optional glare/denoise/deblur
  - Super-resolution
  - Export images + CSV metadata
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Protocol, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd

# ---------------------------- Config & Enums ----------------------------

class SuperResAlgo(str, Enum):
    EDSR = "edsr"
    ESPCN = "espcn"
    FSRCNN = "fsrcnn"
    LAPSRN = "lapsrn"
    BICUBIC = "bicubic"  # fallback

@dataclass(frozen=True)
class PipelineConfig:
    # IO
    input_video: Path
    output_dir: Path

    # Sampling
    sample_every_n_frames: int = 15
    enable_scene_change_sampling: bool = True
    scene_hist_threshold: float = 0.30
    scene_min_gap_frames: int = 20

    # Superres
    upscale_factor: int = 2
    algo: SuperResAlgo = SuperResAlgo.EDSR
    model_dir: Optional[Path] = None

    # Emit
    image_format: str = "png"
    jpg_quality: int = 95
    write_full_size_only: bool = True

    # Limits
    max_frames: Optional[int] = None
    fail_on_missing_superres: bool = False

    # Preprocessing toggles
    enable_glare: bool = False
    enable_denoise: bool = False
    enable_deblur: bool = False

    # Glare params
    glare_v_thresh: float = 0.90
    glare_s_thresh: float = 0.40
    glare_knee_tau: float = 0.75
    glare_knee_strength: float = 3.0
    glare_dilate_px: int = 2

    # Denoise params
    denoise_h_luma: int = 7
    denoise_h_chroma: int = 5
    denoise_template: int = 7
    denoise_search: int = 21

    # Deblur params
    deblur_alpha: float = 0.6
    deblur_sigma: float = 1.2

# ------------------------------- Interfaces ----------------------------

class IVideoReader(Protocol):
    def info(self) -> Dict[str, float]: ...
    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]: ...

class IFrameSampler(Protocol):
    def should_keep(self, idx: int, ts_sec: float, frame_bgr: np.ndarray) -> bool: ...

class ISuperSampler(Protocol):
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray: ...

class IImageSink(Protocol):
    def write(self, frame_idx:int, ts_sec:float, original_bgr:np.ndarray, upscaled_bgr:np.ndarray, metadata:Dict[str,float])->None: ...
    def close(self)->None: ...

class IFrameProcessor(Protocol):
    def process(self, frame_bgr: np.ndarray) -> np.ndarray: ...

# ------------------------------- Video Reader --------------------------

class OpenCVVideoReader(IVideoReader):
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def info(self)->Dict[str,float]:
        duration = (self._frame_count/self._fps) if (self._fps>0 and self._frame_count>0) else 0.0
        return {"width":self._width,"height":self._height,"fps":self._fps,"frame_count":float(self._frame_count),"duration_s":duration}

    def frames(self)->Iterator[Tuple[int,float,np.ndarray]]:
        idx=0
        while True:
            ok, frame = self._cap.read()
            if not ok: break
            ts_sec = idx/self._fps if self._fps>0 else 0.0
            yield idx, ts_sec, frame
            idx+=1
        self._cap.release()

# -------------------------------- Samplers -----------------------------

class NthFrameSampler(IFrameSampler):
    def __init__(self, n:int)->None:
        if n<=0: raise ValueError("n must be >= 1")
        self._n=n
    def should_keep(self, idx:int, ts_sec:float, frame_bgr:np.ndarray)->bool:
        return (idx % self._n)==0

class SceneChangeSampler(IFrameSampler):
    def __init__(self, hist_threshold:float, min_gap_frames:int)->None:
        self._thresh=float(hist_threshold); self._min_gap=int(min_gap_frames)
        self._last_kept_idx:Optional[int]=None; self._last_hist:Optional[np.ndarray]=None
    @staticmethod
    def _norm_hist(img_bgr: np.ndarray)->np.ndarray:
        hsv=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
        cv2.normalize(hist, hist); return hist
    def should_keep(self, idx:int, ts_sec:float, frame_bgr:np.ndarray)->bool:
        hist=self._norm_hist(frame_bgr)
        if self._last_hist is None:
            self._last_hist=hist; self._last_kept_idx=idx; return True
        if self._last_kept_idx is not None and (idx-self._last_kept_idx)<self._min_gap: return False
        score=cv2.compareHist(self._last_hist, hist, cv2.HISTCMP_CORREL)
        change=1.0-float(score)
        if change>=self._thresh:
            self._last_hist=hist; self._last_kept_idx=idx; return True
        return False

class CompositeSampler(IFrameSampler):
    def __init__(self, samplers: List[IFrameSampler])->None:
        self._samplers=samplers
    def should_keep(self, idx:int, ts_sec:float, frame_bgr:np.ndarray)->bool:
        return any(s.should_keep(idx, ts_sec, frame_bgr) for s in self._samplers)

# ------------------------------ Preprocessors --------------------------

class GlareReducer(IFrameProcessor):
    def __init__(self,v_thresh=0.9,s_thresh=0.4,knee_tau=0.75,knee_strength=3.0,dilate_px=2):
        self.v_thresh=v_thresh; self.s_thresh=s_thresh; self.knee_tau=knee_tau; self.knee_strength=knee_strength; self.dilate_px=dilate_px
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)/255.0
        h,s,v=cv2.split(hsv)
        mask=(v>self.v_thresh)&(s<self.s_thresh)
        mask=mask.astype(np.uint8)*255
        if self.dilate_px>0:
            kernel=np.ones((self.dilate_px,self.dilate_px),np.uint8)
            mask=cv2.dilate(mask,kernel)
        over=v>self.knee_tau
        v[over]=self.knee_tau+(v[over]-self.knee_tau)/(1+self.knee_strength*(v[over]-self.knee_tau))
        hsv=cv2.merge([h,s,v])
        return cv2.cvtColor((hsv*255).astype(np.uint8), cv2.COLOR_HSV2BGR)

class Denoiser(IFrameProcessor):
    def __init__(self,h_luma=7,h_chroma=5,template=7,search=21):
        self.h_luma=h_luma; self.h_chroma=h_chroma; self.template=template; self.search=search
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame_bgr,None,h=self.h_luma,hColor=self.h_chroma,templateWindowSize=self.template,searchWindowSize=self.search)

class Deblurrer(IFrameProcessor):
    def __init__(self,alpha=0.6,sigma=1.2):
        self.alpha=alpha; self.sigma=sigma
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        blur=cv2.GaussianBlur(frame_bgr,(0,0),self.sigma)
        return cv2.addWeighted(frame_bgr,1+self.alpha,blur,-self.alpha,0)

class FrameProcessingPipeline(IFrameProcessor):
    def __init__(self,processors:List[IFrameProcessor])->None:
        self._processors=processors
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        out=frame_bgr
        for p in self._processors: out=p.process(out)
        return out

def build_preprocessor(cfg:PipelineConfig)->Optional[FrameProcessingPipeline]:
    procs=[]
    if cfg.enable_glare:
        procs.append(GlareReducer(cfg.glare_v_thresh,cfg.glare_s_thresh,cfg.glare_knee_tau,cfg.glare_knee_strength,cfg.glare_dilate_px))
    if cfg.enable_denoise:
        procs.append(Denoiser(cfg.denoise_h_luma,cfg.denoise_h_chroma,cfg.denoise_template,cfg.denoise_search))
    if cfg.enable_deblur:
        procs.append(Deblurrer(cfg.deblur_alpha,cfg.deblur_sigma))
    return FrameProcessingPipeline(procs) if procs else None

# ------------------------------ SuperSamplers --------------------------

class BicubicSuperSampler(ISuperSampler):
    def __init__(self, scale:int): self._scale=scale
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        h,w=img_bgr.shape[:2]
        return cv2.resize(img_bgr,(w*self._scale,h*self._scale),interpolation=cv2.INTER_CUBIC)

class DNNSuperSampler(ISuperSampler):
    def __init__(self, algo:SuperResAlgo, scale:int, model_dir:Path)->None:
        try:
            from cv2.dnn_superres import DnnSuperResImpl_create
        except Exception as ex:
            raise RuntimeError("OpenCV contrib (dnn_superres) not available. Install opencv-contrib-python.") from ex
        self._sr=DnnSuperResImpl_create(); algo_name=algo.value.lower(); self._sr.setModel(algo_name,scale)
        model_filename=f"{algo_name.upper()}_x{scale}.pb"
        model_path=(model_dir/model_filename).resolve()
        if not model_path.exists(): raise FileNotFoundError(f"Missing model {model_path}")
        self._sr.readModel(str(model_path))
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray: return self._sr.upsample(img_bgr)

def build_supersampler(cfg:PipelineConfig)->ISuperSampler:
    if cfg.algo==SuperResAlgo.BICUBIC: return BicubicSuperSampler(cfg.upscale_factor)
    if cfg.model_dir is None:
        if cfg.fail_on_missing_superres:
            raise ValueError("model_dir must be set for DNN superres.")
        return BicubicSuperSampler(cfg.upscale_factor)
    try: return DNNSuperSampler(cfg.algo,cfg.upscale_factor,cfg.model_dir)
    except Exception as ex:
        if cfg.fail_on_missing_superres: raise
        print(f"[WARN] Superres failed ({ex}), fallback bicubic", file=sys.stderr)
        return BicubicSuperSampler(cfg.upscale_factor)

# -------------------------------- Image Sink ----------------------------

class DiskImageSink(IImageSink):
    def __init__(self,out_dir:Path,image_format="png",jpg_quality=95,write_full_size_only=True):
        self._out=out_dir; self._out.mkdir(parents=True,exist_ok=True)
        self._fmt=image_format.lower(); self._jpg_quality=jpg_quality; self._write_full_only=write_full_size_only; self._rows=[]
    def _imwrite(self,path:Path,bgr:np.ndarray)->None:
        if self._fmt in("jpg","jpeg"): cv2.imwrite(str(path),bgr,[int(cv2.IMWRITE_JPEG_QUALITY),self._jpg_quality])
        else: cv2.imwrite(str(path),bgr)
    def write(self,frame_idx:int,ts_sec:float,original_bgr:np.ndarray,upscaled_bgr:np.ndarray,metadata:Dict[str,float])->None:
        stem=f"frame_{frame_idx:08d}_t{ts_sec:010.3f}"
        up_path=self._out/f"{stem}_SR.{self._fmt}"; self._imwrite(up_path,upscaled_bgr)
        if not self._write_full_only:
            orig_path=self._out/f"{stem}_ORIG.{self._fmt}"; self._imwrite(orig_path,original_bgr)
        row={"frame_idx":frame_idx,"timestamp_s":ts_sec,
             "orig_w":int(original_bgr.shape[1]),"orig_h":int(original_bgr.shape[0]),
             "sr_w":int(upscaled_bgr.shape[1]),"sr_h":int(upscaled_bgr.shape[0])}
        row.update(metadata); self._rows.append(row)
    def close(self)->None:
        if not self._rows: return
        df=pd.DataFrame(self._rows); df.sort_values(by=["frame_idx"],inplace=True)
        df.to_csv(self._out/"frames_metadata.csv",index=False)

# --------------------------- Pipeline Orchestrator ----------------------

class SuperSamplingPipeline:
    def __init__(self,reader:IVideoReader,sampler:IFrameSampler,supersampler:ISuperSampler,sink:IImageSink,max_frames:Optional[int]=None,preprocessor:Optional[IFrameProcessor]=None)->None:
        self._reader=reader; self._sampler=sampler; self._super=supersampler; self._sink=sink; self._max_frames=max_frames; self._pre=preprocessor
    def run(self)->Dict[str,float]:
        info=self._reader.info(); processed=0; kept=0; t0=time.time()
        for idx, ts, frame in self._reader.frames():
            processed+=1
            if self._max_frames is not None and kept>=self._max_frames: break
            if self._sampler.should_keep(idx, ts, frame):
                work=self._pre.process(frame) if self._pre is not None else frame
                up=self._super.upscale(work)
                self._sink.write(idx,ts,frame,up,{"fps":info.get("fps",0.0)})
                kept+=1
        self._sink.close(); elapsed=time.time()-t0
        return {"frames_read":float(processed),"frames_kept":float(kept),"elapsed_s":float(elapsed),"fps_input":float(info.get("fps",0.0))}

# -------------------------------- Factories -----------------------------

def build_sampler(cfg:PipelineConfig)->IFrameSampler:
    parts=[NthFrameSampler(cfg.sample_every_n_frames)]
    if cfg.enable_scene_change_sampling:
        parts.append(SceneChangeSampler(cfg.scene_hist_threshold,cfg.scene_min_gap_frames))
    return CompositeSampler(parts)

# -------------------------------- CLI Entry -----------------------------

def main(argv:Optional[List[str]]=None)->None:
    import argparse
    p=argparse.ArgumentParser(description="Nest night video pipeline")
    p.add_argument("input",type=Path)
    p.add_argument("-o","--out",type=Path,required=True)
    p.add_argument("--n",type=int,default=15)
    p.add_argument("--scene",action="store_true")
    p.add_argument("--scene-thresh",type=float,default=0.30)
    p.add_argument("--scene-gap",type=int,default=20)
    p.add_argument("--algo",type=str,default="edsr",choices=[a.value for a in SuperResAlgo])
    p.add_argument("--scale",type=int,default=2)
    p.add_argument("--models",type=Path,default=None)
    p.add_argument("--fmt",type=str,default="png",choices=["png","jpg"])
    p.add_argument("--jpgq",type=int,default=95)
    p.add_argument("--orig-too",action="store_true")
    p.add_argument("--max",type=int,default=None)
    p.add_argument("--strict-dnn",action="store_true")

    # Preprocessing flags
    p.add_argument("--glare",action="store_true")
    p.add_argument("--denoise",action="store_true")
    p.add_argument("--deblur",action="store_true")
    p.add_argument("--glare-v",type=float,default=0.90)
    p.add_argument("--glare-s",type=float,default=0.40)
    p.add_argument("--glare-tau",type=float,default=0.75)
    p.add_argument("--glare-k",type=float,default=3.0)
    p.add_argument("--glare-dilate",type=int,default=2)
    p.add_argument("--dn-hl",type=int,default=7)
    p.add_argument("--dn-hc",type=int,default=5)
    p.add_argument("--dn-t",type=int,default=7)
    p.add_argument("--dn-s",type=int,default=21)
    p.add_argument("--db-alpha",type=float,default=0.6)
    p.add_argument("--db-sigma",type=float,default=1.2)

    args=p.parse_args(argv)

    cfg=PipelineConfig(
        input_video=args.input,output_dir=args.out,sample_every_n_frames=args.n,
        enable_scene_change_sampling=args.scene,scene_hist_threshold=args.scene_thresh,scene_min_gap_frames=args.scene_gap,
        upscale_factor=args.scale,algo=SuperResAlgo(args.algo),model_dir=args.models,
        image_format=args.fmt,jpg_quality=args.jpgq,write_full_size_only=not bool(args.orig_too),
        max_frames=args.max,fail_on_missing_superres=args.strict_dnn,
        enable_glare=args.glare,enable_denoise=args.denoise,enable_deblur=args.deblur,
        glare_v_thresh=args.glare_v,glare_s_thresh=args.glare_s,glare_knee_tau=args.glare_tau,glare_knee_strength=args.glare_k,glare_dilate_px=args.glare_dilate,
        denoise_h_luma=args.dn_hl,denoise_h_chroma=args.dn_hc,denoise_template=args.dn_t,denoise_search=args.dn_s,
        deblur_alpha=args.db_alpha,deblur_sigma=args.db_sigma
    )

    reader=OpenCVVideoReader(cfg.input_video)
    sampler=build_sampler(cfg)
    supers=build_supersampler(cfg)
    sink=DiskImageSink(cfg.output_dir,cfg.image_format,cfg.jpg_quality,cfg.write_full_size_only)
    pre=build_preprocessor(cfg)

    pipeline=SuperSamplingPipeline(reader,sampler,supers,sink,cfg.max_frames,preprocessor=pre)
    stats=pipeline.run()
    print(f"Done. Stats: {stats}")

if __name__=="__main__":
    main()
