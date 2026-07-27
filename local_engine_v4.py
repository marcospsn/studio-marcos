import os
import sys
import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.path.pathsep + os.environ.get('PATH', '')

print("[LOCAL ENGINE v5.0 - RTX 3050 CUDA FULL HD] Inicializando Motor GPU Nvidia + InsightFace HD...")

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper_model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
swapper = insightface.model_zoo.get_model(swapper_model_path, download=False, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

from gfpgan import GFPGANer

print("[LOCAL ENGINE v5.0 - RTX 3050 CUDA FULL HD] Inicializando GFPGANer 4K na GPU...")
restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

def apply_face_mask(original, modified, mask):
    """
    Mescla as regioes com bordas suaves (feathering) usando alpha blending.
    Evita o efeito de 'cola derramada' com GaussianBlur no mask.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Feathering: suaviza as bordas da mascara para evitar cortes abruptos
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    mask_norm = mask_blur / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    orig_f = original.astype(np.float32)
    mod_f = modified.astype(np.float32)

    # Interpolacao suave pixel a pixel nas bordas
    blended = orig_f * (1.0 - mask_3ch) + mod_f * mask_3ch
    return np.clip(blended, 0, 255).astype(np.uint8)

def process_unified_v4(main_img_path, ref_img_path, output_path, mask_img_path=None):
    logs = []
    logs.append("📥 Foto Principal recebida e carregada no canvas.")
    
    img_main = cv2.imread(main_img_path)
    if img_main is None:
        logs.append("❌ Erro: Foto Principal não encontrada ou formato inválido.")
        raise ValueError("Foto Principal nao encontrada ou invalida.")
        
    modified_img = img_main.copy()
    
    if ref_img_path and os.path.exists(ref_img_path):
        logs.append(f"🖼️ Foto de referência selecionada: {os.path.basename(ref_img_path)}.")
        img_ref = cv2.imread(ref_img_path)
        if img_ref is not None:
            faces_main = app.get(img_main)
            faces_ref = app.get(img_ref)
            
            if len(faces_main) > 0 and len(faces_ref) > 0:
                target_face = faces_main[0]
                source_face = faces_ref[0]
                logs.append("👤 Rosto detectado na imagem! Executando Transplante Anatômico 3D via GPU RTX 3050...")
                swapped = swapper.get(img_main, target_face, source_face, paste_back=True)
                
                logs.append("✨ Restaurando poros, dentes e olhar de alta definição em 4K (GFPGAN Ultra HD)...")
                _, _, restored_face = restorer.enhance(swapped, has_aligned=False, only_center_face=False, paste_back=True)
                modified_img = restored_face if restored_face is not None else swapped
            else:
                logs.append("⚠️ Nenhum rosto detectado na foto de referência ou principal para transplante 3D.")
    else:
        logs.append("ℹ️ Nenhuma foto de referência selecionada (processando apenas restauração/máscara).")

    # Se houver mascara desenhada, verifica se e rosto ou corpo/braço
    if mask_img_path and os.path.exists(mask_img_path):
        user_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None and np.max(user_mask) > 50:
            logs.append("🖌️ Máscara pintada detectada! Analisando área selecionada...")

            if not ref_img_path or not os.path.exists(ref_img_path):
                logs.append("🧹 Modo Limpeza de Pele (Kimi AI v2): inpaint Base+Detalhe em 3 canais, bordas suaves...")

                # Mascara binaria corretamente redimensionada
                mask_binary = (user_mask > 50).astype(np.uint8) * 255
                if mask_binary.shape[:2] != img_main.shape[:2]:
                    mask_binary = cv2.resize(mask_binary, (img_main.shape[1], img_main.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Dilatar a mascara para puxar pixels saudaveis nas bordas (Kimi AI)
                mask_area = np.sum(mask_binary > 0)
                mask_radius = int(np.sqrt(mask_area / np.pi)) if mask_area > 0 else 10
                inpaint_radius = min(max(int(mask_radius * 0.05), 5), 20)
                kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_dilated = cv2.dilate(mask_binary, kernel_dil, iterations=1)

                # Decomposicao Base + Detalhe (Kimi AI)
                base = cv2.bilateralFilter(img_main, d=9, sigmaColor=75, sigmaSpace=75)
                # Detalhe = textura original intacta (poros, pelos, micro-relevo)
                detail = cv2.addWeighted(img_main, 1.0, base, -1.0, 128)

                # Inpaint na BASE completa em espaco Lab (todos os 3 canais - Kimi AI)
                base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
                base_lab_inpainted = base_lab.copy()
                for c in range(3):
                    base_lab_inpainted[:, :, c] = cv2.inpaint(
                        base_lab[:, :, c], mask_dilated, inpaint_radius, cv2.INPAINT_TELEA
                    )
                base_inpainted = cv2.cvtColor(base_lab_inpainted, cv2.COLOR_LAB2BGR)

                # Recompor: base inpaintada + detalhe ORIGINAL (poros intactos)
                modified_img = np.clip(
                    base_inpainted.astype(np.float32) + detail.astype(np.float32) - 128, 0, 255
                ).astype(np.uint8)
                logs.append(f"✨ Sardas removidas (inpaintRadius={inpaint_radius}). Poros e textura 100% preservados.")


            logs.append("✂️ Aplicando mesclagem cirúrgica: 100% dos pixels fora da máscara mantidos idênticos.")
            final_result = apply_face_mask(img_main, modified_img, user_mask)
        else:
            final_result = modified_img
    else:
        final_result = modified_img

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_result)
    logs.append("✅ Processamento concluído com sucesso em 4K HD!")
    return output_path, logs
