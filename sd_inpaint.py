"""
sd_inpaint.py — Stable Diffusion Inpainting Engine (Kimi AI v1)
Integrado ao MP Studio v6.0
Cache em D:/hf_cache (nunca no C:)
Lazy loading: pipeline carregado somente na primeira mascara detectada.
Sequential CPU offload: essencial para RTX 3050 6GB com outros modelos carregados.
"""

import os
import gc
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple

# =============================================================
# CACHE EM D: — define ANTES de importar diffusers/transformers
# =============================================================
_HF_CACHE = "D:/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE
os.environ["DIFFUSERS_CACHE"] = _HF_CACHE

_MODEL_ID = "runwayml/stable-diffusion-inpainting"
_MAX_DIM = 768          # limite para evitar OOM (múltiplo de 8)
_pipe = None            # singleton global — reutilizado entre chamadas
_pipe_attempted = False # evita tentar carregar de novo após falha definitiva

# =============================================================
# PROMPTS OTIMIZADOS — pele hiperrealista sem efeito artificial
# =============================================================
_PROMPT = (
    "professional portrait photography, hyperrealistic natural skin, "
    "seamless skin texture, subtle skin pores, soft diffused studio lighting, "
    "realistic complexion, natural skin tone, photorealistic, "
    "8k uhd, highly detailed, sharp focus"
)
_NEGATIVE = (
    "doll-like skin, plastic skin, waxy skin, artificial smooth skin, "
    "overprocessed, oversaturated, blurry, painted, cartoon, anime, "
    "3d render, cgi, plastic surgery, uncanny valley, makeup artifacts, "
    "watermark, text, logo, synthetic texture, filter"
)


def _load_pipe(logs: list) -> bool:
    """Carrega o pipeline SD lazily. Retorna True se bem-sucedido."""
    global _pipe, _pipe_attempted
    if _pipe is not None:
        return True
    if _pipe_attempted:
        return False

    _pipe_attempted = True
    try:
        from diffusers import StableDiffusionInpaintPipeline

        logs.append("🤖 SD Inpainting: carregando modelo neural (1ª vez pode baixar ~4GB)...")

        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=_HF_CACHE,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # CRÍTICO para 6GB VRAM: componentes ficam na CPU, só o ativo vai pra GPU
        _pipe.enable_sequential_cpu_offload()
        _pipe.enable_attention_slicing(1)
        _pipe.enable_vae_slicing()
        _pipe.set_progress_bar_config(disable=True)

        # Tenta xformers (mais rápido, opcional)
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        logs.append("✅ SD Inpainting: modelo pronto (CPU offload ativo para RTX 3050 6GB).")
        return True

    except Exception as e:
        logs.append(f"⚠️ SD Inpainting: falha ao carregar ({e}). Usando fallback OpenCV.")
        _pipe = None
        return False


def _unload_pipe():
    """Descarrega o pipeline e libera VRAM após OOM."""
    global _pipe, _pipe_attempted
    if _pipe is not None:
        del _pipe
        _pipe = None
    _pipe_attempted = False   # permite nova tentativa no próximo request
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _safe_resize(img_bgr: np.ndarray,
                 mask_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Redimensiona para <= _MAX_DIM e garante múltiplo de 8 (exigido pelo VAE)."""
    oh, ow = img_bgr.shape[:2]
    scale = min(1.0, _MAX_DIM / max(oh, ow))
    nw = max(8, (int(ow * scale) // 8) * 8)
    nh = max(8, (int(oh * scale) // 8) * 8)

    img_r  = cv2.resize(img_bgr,  (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask_gray, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r, (oh, ow)


def process_inpaint_sd(img_bgr: np.ndarray,
                       mask_gray: np.ndarray,
                       logs: list,
                       strength: float = 0.25) -> Optional[np.ndarray]:
    """
    Executa Stable Diffusion Inpainting na área mascarada.

    Parâmetros:
        img_bgr   : numpy BGR (imagem original)
        mask_gray : numpy uint8 grayscale (255 = área a processar)
        logs      : lista de strings para o painel de feedback
        strength  : 0.20–0.35 (baixo = preserva textura, alto = mais criativo)

    Retorna numpy BGR com inpainting aplicado, ou None para fallback OpenCV.
    """
    if np.sum(mask_gray > 0) == 0:
        logs.append("⚠️ Máscara vazia — nenhum pixel para processar.")
        return None

    if not _load_pipe(logs):
        return None

    oh, ow = img_bgr.shape[:2]
    img_r, mask_r, (orig_h, orig_w) = _safe_resize(img_bgr, mask_gray)

    img_pil  = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray((mask_r > 127).astype(np.uint8) * 255)

    logs.append(f"🎨 SD Inpainting: processando {img_pil.width}×{img_pil.height}px (strength={strength})...")

    try:
        generator = torch.Generator(device="cpu").manual_seed(42)

        result = _pipe(
            prompt=_PROMPT,
            negative_prompt=_NEGATIVE,
            image=img_pil,
            mask_image=mask_pil,
            strength=strength,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        # Volta para resolução original
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if result_bgr.shape[:2] != (orig_h, orig_w):
            result_bgr = cv2.resize(result_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        logs.append("✨ SD Inpainting: textura neural aplicada com sucesso.")
        return result_bgr

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower():
            logs.append("⚠️ SD Inpainting: OOM detectado — liberando VRAM, usando fallback.")
            _unload_pipe()
        else:
            logs.append(f"⚠️ SD Inpainting: erro ({e}). Usando fallback.")
        return None

    except Exception as e:
        logs.append(f"⚠️ SD Inpainting: erro inesperado ({e}). Usando fallback.")
        return None
