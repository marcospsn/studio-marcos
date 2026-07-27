# Treinamento de LoRA FLUX.1 Quantizado (Otimizado para GPU T4 do Colab Grátis)

import os

print("=== INICIANDO CONFIGURAÇÃO OTIMIZADA PARA GPU T4 ===")

# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Instalar AI-Toolkit com otimização de memória (bitsandbytes e quanto)
!git clone https://github.com/ostris/ai-toolkit.git
%cd ai-toolkit
!git submodule update --init --recursive
!pip install -q -r requirements.txt
!pip install -q bitsandbytes optimum-quanto lycoris_lora diffusers transformers ftfy accelerate

# 3. Descompactar dataset legendado
!mkdir -p /content/dataset_marcos
!unzip -o "/content/drive/MyDrive/Imagens/Exemplos de Marcos/marcos_dataset_captioned.zip" -d /content/dataset_marcos

# 4. Configuração com Quantização em 4-bit/8-bit para evitar queda por falta de VRAM
config_yaml = """
job: extension
config:
  name: "marcos_paulo_lora_t4"
  process:
    - type: 'sd_trainer'
      training_folder: "/content/drive/MyDrive/LoRA_Marcos_Output"
      device: cuda:0
      trigger_word: "marcos_paulo"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves: 4
      datasets:
        - folder_path: "/content/dataset_marcos"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 800
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        low_vram: true
"""

with open("/content/ai-toolkit/config_marcos.yaml", "w") as f:
    f.write(config_yaml)

print("\n Setup otimizado para a GPU T4 do Colab concluído com sucesso!")
print("Execute a célula abaixo para iniciar:")
print("!python run.py config_marcos.yaml")
