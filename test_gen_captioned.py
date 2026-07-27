import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Novo LoRA treinado com os arquivos .txt individuais do Gemini
lora_url = "https://replicate.delivery/xezq/f8HYG0LsewtTyksmSoOVMqWtyX5elfc9N720eajqbndItfBvF/flux-lora.tar"

print("Disparando teste de geração com o novo LoRA legendado...")
pred = client.predictions.create(
    model="black-forest-labs/flux-dev",
    input={
        "prompt": "a photo of marcos_paulo, studio portrait, dark hair, bearded, muscular build, natural skin texture with visible pores, 35mm photograph",
        "lora_weights": lora_url,
        "lora_scale": 0.8,
        "guidance_scale": 3.5
    }
)

print("ID da Predição:", pred.id)
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
