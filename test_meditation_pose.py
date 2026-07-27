import os
import json
import time
from dotenv import load_dotenv
import replicate

load_dotenv()

with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Usamos a foto de referência real do Marcos como base de fisionomia (foto 01)
ref_info = catalog["references"][0]
print(f"Foto de referência real selecionada: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

print("Enviando foto real do Marcos para a nuvem do Replicate...")
file_obj = client.files.create(open(orig_img_path, "rb"))
real_photo_url = file_obj.urls["get"]

print("\nDisparando geração sem treino: Marcos meditando em posição de lótus de olhos fechados...")

# Usamos PuLID / FLUX para transferir a identidade real do Marcos para a nova pose completa
pred = client.predictions.create(
    version="bytedance/flux-pulid:8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",


    input={
        "main_face_image": real_photo_url,
        "prompt": "a full body photo of marcos_paulo, a 35yo man with short dark hair and short beard, sitting in lotus meditation pose with eyes closed, hands gently resting on knees with fingertips touching, calm serene atmosphere, natural lighting, realistic human skin texture with pores",
        "identity_scale": 0.85,
        "num_inference_steps": 30,
        "guidance_scale": 3.5
    }
)

print(f"Predição iniciada! ID: {pred.id}")
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\nIMAGEM DO MARCOS MEDITANDO GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\nERRO:", pred.error)
