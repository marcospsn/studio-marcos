import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
lora_url = "https://replicate.delivery/xezq/UnzS2SMIWPbBG52QBCBPieq3GKuZBfek30whwkBcLvb7YN4tA/flux-lora.tar"

print("Disparando geração de teste direto...")
pred = client.predictions.create(
    model="black-forest-labs/flux-dev",
    input={
        "prompt": "a photograph of marcos_paulo, professional studio portrait, natural skin texture, visible pores, sharp focus, 35mm lens",
        "lora_weights": lora_url,
        "lora_scale": 0.65,
        "guidance_scale": 3.5
    }
)

print("ID:", pred.id)
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\n✅ IMAGEM GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\n❌ ERRO:", pred.error)
