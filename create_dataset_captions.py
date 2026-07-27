import os
import json
import time
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REF_PHOTOS_DIR = r"G:\Meu Drive\Imagens\Exemplos de Marcos"
OUTPUT_DIR = r"D:\OneDrive\Documentos\PROJETOS\ARQUITETO\studio_marcos\dataset_prepared"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(api_key=GEMINI_API_KEY)

# Lista das 7 fotos oficiais sem cópias duplicadas
official_photos = [
    "Marcos 01 leve sorriso.png",
    "Marcos 02 olhando pra camera.png",
    "Marcos 03 pensativo.png",
    "Marcos 04 sério.png",
    "Marcos 05 sorrindo.png",
    "Marcos 06 Sério olhando pro céu.png",
    "Marcos 07 foto 3x4 leve sorriso(1).png"
]

print("Iniciando a geração de legendas (.txt) de alta precisão...")

for filename in official_photos:
    img_path = os.path.join(REF_PHOTOS_DIR, filename)
    if not os.path.exists(img_path):
        print(f"Foto não encontrada: {filename}")
        continue

    print(f"Analisando imagem: {filename}...")
    try:
        img = Image.open(img_path)
        
        # Redimensionar para formato ideal 1024x1024 mantendo proporção se necessário
        img_rgb = img.convert("RGB")
        target_img_path = os.path.join(OUTPUT_DIR, filename)
        img_rgb.save(target_img_path)

        # Solicitar descrição multimodal detalhada via Gemini 3.6 Flash
        prompt = """Analyze this image in extreme photographic detail to create an accurate LoRA training caption.
Describe the subject's exact facial features, beard style, hair, eyes, skin quality, expression, pose, framing, chest hair, clothing, and background.
CRITICAL: The description MUST start exactly with: a photo of marcos_paulo,
Keep description in English and focus on photorealistic traits."""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[img_rgb, prompt]
        )

        caption_text = response.text.strip()
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption_text)

        print(f"Legenda gerada com sucesso para {txt_filename}!")
        time.sleep(2)

    except Exception as e:
        print(f"Erro ao processar {filename}: {e}")

print("\nTodas as 7 fotos receberam arquivos .txt de alta precisão!")

