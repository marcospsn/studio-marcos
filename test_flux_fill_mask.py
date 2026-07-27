import os
import json
import time
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import replicate

load_dotenv()

with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

ref_info = catalog["references"][0]
print(f"Foto de referência real do Marcos: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

# Criar a máscara para a área da roupa
base_img = Image.open(orig_img_path).convert("RGB")
base_img.thumbnail((1024, 1024))
w, h = base_img.size

mask_img = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask_img)
draw.rectangle([0, int(h * 0.68), w, h], fill=255)

base_path = "temp_ref_base.png"
mask_path = "temp_ref_mask.png"
base_img.save(base_path)
mask_img.save(mask_path)

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

print("Enviando foto real e máscara para a nuvem...")
base_file = client.files.create(open(base_path, "rb"))
mask_file = client.files.create(open(mask_path, "rb"))

print("Disparando inpainting com a sua FOTO REAL de referência (Sem treino)...")
pred = client.predictions.create(
    model="black-forest-labs/flux-fill-dev",
    input={
        "image": base_file.urls["get"],
        "mask": mask_file.urls["get"],
        "prompt": "an elegant navy blue blazer over a crisp white shirt, high detail realistic fabric texture",
        "guidance_scale": 3.5,
        "num_inference_steps": 28
    }
)

print(f"Predição iniciada! ID: {pred.id}")
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\nIMAGEM GERADA COM SUCESSO SEM TREINAMENTO:")
    print("URL:", pred.output)
else:
    print("\nERRO:", pred.error)
