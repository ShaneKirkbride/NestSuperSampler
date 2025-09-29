# NestSuperSampler

SOLID Python pipeline:
- Read Nest videos
- Optional: glare suppression → denoise → deblur (night-video)
- Super-resolution via OpenCV dnn_superres (EDSR/ESPCN/FSRCNN/LapSRN) or bicubic fallback
- Export upscaled frames + frames_metadata.csv

## Quickstart (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Bicubic fallback (no models needed)
python .\src\nest_super_sampler.py .\night_clip.mp4 -o .\out --n 12 --scene --algo bicubic --scale 2 --fmt png --max 150

# DNN SR (EDSR x4) with night preprocessing
python .\NestSuperSampler.py .\videos\car2.mp4 -o .\out --n 10 --scene --algo edsr --scale 4 --models .\models --fmt jpg --max 150 --glare --glare-v 0.9 --glare-s 0.4 --glare-tau 0.75 --glare-k 3 --glare-dilate 2 --denoise --dn-hl 7 --dn-hc 5 --dn-t 7 --dn-s 21 --deblur --db-alpha 0.6 --db-sigma 1.2
```
