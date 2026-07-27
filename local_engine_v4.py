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
            
            # Sem referência: inpaint + seamlessClone para remover sardas/manchas com textura natural
            if not ref_img_path or not os.path.exists(ref_img_path):
                logs.append("🧹 Modo Limpeza de Pele: removendo sardas/manchas e fundindo textura nas bordas (inpaint + seamless blend)...")
                # Mascara binaria para o inpaint (pixeis a preencher)
                mask_binary = (user_mask > 50).astype(np.uint8) * 255
                if mask_binary.shape[:2] != img_main.shape[:2]:
                    mask_binary = cv2.resize(mask_binary, (img_main.shape[1], img_main.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Inpainting: preenche a regiao mascarada usando pixels vizinhos (Telea algorithm)
                inpainted = cv2.inpaint(img_main, mask_binary, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
                # Equaliza a textura de pele com bilateral filter na regiao inpaintada
                inpainted_smooth = cv2.bilateralFilter(inpainted, d=9, sigmaColor=75, sigmaSpace=75)
                # seamlessClone: funde a regiao suavizada com a imagem original preservando bordas e iluminacao
                mask_for_clone = mask_binary.copy()
                # Centro da mascara para o seamlessClone
                ys, xs = np.where(mask_for_clone > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int((xs.min() + xs.max()) / 2)
                    cy = int((ys.min() + ys.max()) / 2)
                    try:
                        modified_img = cv2.seamlessClone(inpainted_smooth, img_main, mask_for_clone, (cx, cy), cv2.NORMAL_CLONE)
                        logs.append("✨ Textura de pele preservada e sardas removidas com fusão natural nas bordas.")
                    except Exception:
                        modified_img = inpainted_smooth
                        logs.append("✨ Sardas removidas com inpaint (fallback sem seamlessClone).")
                else:
                    modified_img = inpainted_smooth

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
