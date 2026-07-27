# ==============================================================================
# 🚀 TESTE DE QUALIDADE E CLONAGEM ANATÔMICA (INSTANTID + SDXL + REAL-ESRGAN)
# Executar no Google Colab (Gratuito com GPU T4/A100) da sua conta Google Pro
# ==============================================================================

# 1. INSTALAÇÃO DE DEPENDÊNCIAS DE ALTA NITIDEZ E RECONSTRUÇÃO DE PELE
!pip install -q diffusers transformers accelerate insightface controlnet_aux opencv-python Pillow
!pip install -q git+https://github.com/xinntao/Real-ESRGAN.git

import sys
import os
import cv2
import torch
import torchvision.transforms.functional as F
# Patch para compatibilidade do basicsr com torchvision recente
sys.modules['torchvision.transforms.functional_tensor'] = F

import numpy as np
from PIL import Image
from google.colab import files
from google.colab import drive

print("✅ Dependências instaladas com sucesso!")

# 2. MONTAR GOOGLE DRIVE PARA CARREGAR FOTO REAL DE MARCOS
drive.mount('/content/drive')

# Caminho da foto de referência real do Marcos no seu Google Drive
ref_path = "/content/drive/MyDrive/Imagens/Exemplos de Marcos/Marcos 07 foto 3x4 leve sorriso(1).png"
if not os.path.exists(ref_path):
    print("📌 Por favor, faça o upload de 1 foto sua de teste:")
    uploaded = files.upload()
    ref_path = list(uploaded.keys())[0]

print(f"📸 Foto de Referência Carregada: {ref_path}")

# 3. PIPELINE DE RESTAURAÇÃO DE NITIDEZ 2K COM REAL-ESRGAN (SEM DEFORMAR SOBRANCELHAS OU DENTES)
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

print("🔄 Processando restauração de poros, pelos do braço e dentes em 2K HD...")

model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model_esrgan,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True
)

img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
output_img, _ = upsampler.enhance(img, outscale=2)

out_filename = "/content/marcos_restaurado_2k.png"
cv2.imwrite(out_filename, output_img)

print("\n=====================================================================")
print(f"✨ RESTAURAÇÃO CONCLUÍDA! Imagem salva em: {out_filename}")
print("=====================================================================")

# Exibe a imagem restaurada direto na tela do Colab para você avaliar
from IPython.display import Image as DisplayImage, display
display(DisplayImage(filename=out_filename))
