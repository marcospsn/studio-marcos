import os
import json
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from google import genai
import requests

load_dotenv()

# 1. Carregar o catálogo JSON de fotos de referência reais
with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Selecionar a foto de referência real do Marcos (ex: foto 01 frontal)
ref_info = catalog["references"][0]
print(f"Foto de referência selecionada: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

# 2. Abrir a foto original real
base_img = Image.open(orig_img_path).convert("RGB")

# Redimensionar mantendo proporção para o teste (max 1024)
base_img.thumbnail((1024, 1024))
w, h = base_img.size

# 3. Criar uma máscara de exemplo (vamos mascarar a roupa/camisa para alterar o terno/estilo)
# Isso demonstra a alteração mantendo 100% o seu rosto, pele, barba e cabelos congelados.
mask_img = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask_img)

# Mascarar do pescoço para baixo (área da camisa/corpo)
draw.rectangle([0, int(h * 0.70), w, h], fill=255)

# Salvar as imagens de teste temporárias
base_img.save("temp_base.png")
mask_img.save("temp_mask.png")

print("Imagens temporárias preparadas.")
print("Disparando edição multimodal sem treinamento com o Gemini 3.6 Flash / Imagen...")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = (
    "Using the provided reference photo of Marcos as the exact ground truth for his face, skin texture, beard, and eyes, "
    "perform an inpainting modification on the masked region (torso/clothing). "
    "Change his current shirt to an elegant navy blue blazer over a crisp white shirt. "
    "Keep 100% of his actual face, skin pores, beard, head hair, background, and lighting completely identical and realistic."
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        base_img,
        mask_img,
        prompt
    ]
)

print("\n--- RESPOSTA DA EDIÇÃO MULTIMODAL SEM TREINAMENTO ---")
print(response.text)
