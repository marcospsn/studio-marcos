import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# URL do dataset hospedado no Replicate Files
ZIP_URL = "https://api.replicate.com/v1/files/YTkzZDdhM2ItZTI2My00ODNmLWFlYzQtYjE4NDNiYzFiMjAz.zip"

print("Disparando treinamento profissional no Replicate com Ostris AI-Toolkit oficial...")
training = client.trainings.create(
    version="ostris/flux-dev-lora-trainer:26dce37af90b9d997eeb970d92e47de3064d46c300504ae376c75bef6a9022d2",
    input={
        "input_images": ZIP_URL,
        "trigger_word": "marcos_paulo",
        "steps": 1000,
        "lora_rank": 16,
        "optimizer": "adamw8bit",
        "learning_rate": 0.0001,
        "resolution": "1024",
        "autocaption": False,  # Preserva 100% dos .txt gerados no Gemini
        "batch_size": 1
    },
    destination="marcospsn/marcos-paulo-lora-hd"
)

print(f"Treinamento iniciado! ID: {training.id}")
start = time.time()
while training.status not in ["succeeded", "failed", "canceled"]:
    time.sleep(10)
    training.reload()
    print(f"Status do Treino ({int(time.time() - start)}s): {training.status}")

if training.status == "succeeded":
    print("\nTREINAMENTO CONCLUIDO COM SUCESSO!")
    print("Pesos gerados:", training.output)
else:
    print("\nERRO NO TREINAMENTO:", training.error)
