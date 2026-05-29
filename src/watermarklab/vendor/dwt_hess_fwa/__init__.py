"""Paper-faithful reproduction scaffold for Gaata et al. (2022) DWT + Hessenberg + FWA image watermarking."""

from .metrics import mse, psnr, ber, nc, ncc
from .watermark import WatermarkConfig, embed_watermark, extract_watermark
