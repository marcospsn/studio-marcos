import os
import json
import io
import time
import sys
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

print("=====================================================================")
print("TESTE DE ANALISADOR DE POSE E SELECAO DE REFERENCIAS")
print("=====================================================================")

def analyze_and_select_references(target_image_path: str):
    if not os.path.exists(target_image_path):
        print(f"❌ Arquivo não encontrado: {target_image_path}")
        return

    print(f"\n📸 Analisando imagem de teste: {target_image_path}")
    img_pil = Image.open(target_image_path).convert("RGB")

    # Prompt leve para o modelo classificar o enquadramento, ângulo do rosto e expressão
    analysis_prompt = (
        "Analyze this image of a person and respond strictly with a short JSON describing: "
        "1. 'pose': choose from ['frontal', 'perfil_3_4', 'inclinado_cima', 'corpo_inteiro_frontal']. "
        "2. 'expression': choose from ['sorriso_aberto', 'leve_sorriso', 'serio_neutro']. "
        "3. 'teeth_visible': boolean (true if teeth are visible, false otherwise). "
        "4. 'reasoning': short 1-sentence description of what you observed (in Portuguese)."
    )

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[img_pil, analysis_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        t1 = time.time()

        analysis = json.loads(response.text)
        print(f"⏱️ Análise concluída em {round(t1 - t0, 2)}s!")
        print(f"  🔍 Visão da IA: {analysis.get('reasoning')}")
        print(f"  📐 Pose detectada: {analysis.get('pose')}")
        print(f"  😊 Expressão: {analysis.get('expression')} (Dentes visíveis: {analysis.get('teeth_visible')})")

        # Regra de Seleção Automática de Referências do Catálogo
        selected = []
        
        # 1. Se tem dentes visíveis, prioriza Marcos 05 (sorrindo)
        if analysis.get('teeth_visible'):
            selected.append("marcos_05_sorrindo")

        # 2. Seleciona pela pose
        pose = analysis.get('pose')
        if pose == 'perfil_3_4':
            selected.append("marcos_03_pensativo")
        elif pose == 'inclinado_cima':
            selected.append("marcos_06_olhando_ceu")
        elif pose == 'corpo_inteiro_frontal':
            selected.append("marcos_04_serio")
            selected.append("marcos_01_frontal_sorriso")
        else: # frontal
            selected.append("marcos_07_foto_3x4")
            selected.append("marcos_01_frontal_sorriso")

        # Garante a foto 3x4 (marcos_07) para detalhes de olho/sobrancelha se não estiver na lista
        if "marcos_07_foto_3x4" not in selected and len(selected) < 3:
            selected.append("marcos_07_foto_3x4")

        # Remove duplicados mantendo a ordem
        final_selected = list(dict.fromkeys(selected))[:3]

        print("  🎯 Fotos do Banco Escolhidas Automaticamente:")
        for ref_id in final_selected:
            for item in catalog["references"]:
                if item["id"] == ref_id:
                    print(f"     ✅ [{item['id']}] - {item['file_name']} (Motivo: {item['pose']}, {item['expression']})")

    except Exception as e:
        print(f"❌ Erro na análise: {e}")

# Executa o teste em todas as fotos de exemplo registradas no catálogo + a foto temp_main.png do teste atual
test_paths = [ref["path"] for ref in catalog["references"]]
if os.path.exists("temp_main.png"):
    test_paths.append("temp_main.png")

for path in test_paths:
    analyze_and_select_references(path)
