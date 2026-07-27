import os
import json
import time
import io
from dotenv import load_dotenv

from PIL import Image, ImageDraw
from google import genai
from google.genai import types

load_dotenv()

# 1. Carregar o catálogo JSON de fotos de referência do Marcos
with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Selecionar a foto de referência real do Marcos (ex: foto 01 frontal)
ref_info = catalog["references"][0]
print(f"Foto de referência selecionada: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

# 2. Prepara a imagem base e a máscara
base_img = Image.open(orig_img_path).convert("RGB")
base_img.thumbnail((1024, 1024))
w, h = base_img.size

mask_img = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask_img)
# Mascarar apenas a roupa do pescoço para baixo
draw.rectangle([0, int(h * 0.68), w, h], fill=255)

base_img.save("temp_base.png")
mask_img.save("temp_mask.png")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = (
    "Inpaint edit: Change the subject's shirt in the masked lower region to a stylish navy blue blazer with a white shirt underneath. "
    "Keep 100% of the subject's face, beard, skin texture, head hair, background, and lighting completely identical to the source photo."
)

print("Disparando geração de imagem via Imagen 3 (Reference Inpainting)...")
try:
    result = client.models.generate_images(
        model="gemini-3.1-flash-image",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            output_mime_type="image/png"
        )
    )


    
    for i, generated_image in enumerate(result.generated_images):
        image = Image.open(io.BytesIO(generated_image.image.image_bytes))
        out_path = f"resultado_zero_shot_{i+1}.png"
        image.save(out_path)
        print(f"✅ IMAGEM SALVA COM SUCESSO EM: {out_path}")

except Exception as e:
    print(f"Erro na geração de imagem: {e}")
