%%writefile /kaggle/working/InstantMesh2/app.py
import os
import time
import torch
import numpy as np
import rembg
import tempfile
import gc
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import gradio as gr
import sys

# InstantMesh bağımlılıkları (varsayılan olarak kurulu kabul edilmiştir)
try:
    from src.utils.train_util import instantiate_from_config
    from src.utils.camera_util import get_zero123plus_input_cameras
    from src.utils.mesh_util import save_obj, save_glb
except ImportError as e:
    print(f"[FATAL ERROR] InstantMesh source code (src) imports failed. Ensure the InstantMesh repository structure is correct. Error: {e}")
    sys.exit(1)


# -------------------- GPU / CPU & Konfigürasyon --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# VRAM fragmentasyonunu azaltma (CUDA için)
if device.type == 'cuda':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] PYTORCH_CUDA_ALLOC_CONF set for VRAM efficiency.")

# -------------------- Seed --------------------
SEED = 42
seed_everything(SEED)
print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Seed set to {SEED}.")

# -------------------- Config --------------------
config_path = "configs/instant-mesh-base.yaml"
try:
    print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    model_config = config.model_config
    infer_config = config.infer_config
    print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Config loaded successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Config file loading failed: {e}")
    sys.exit(1)


# -------------------- Model Yükleme: Rekonstrüksiyon Modeli (BASE) --------------------
print(f"[{time.strftime('%H:%M:%S')}] [MODEL] Loading BASE Reconstruction Model...")
try:
    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_base.ckpt",
        repo_type="model",
        cache_dir="./ckpts/"
    )
    model = instantiate_from_config(model_config)
    state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    
    # Prefix'i kaldır ve sadece lrm_generator ile başlayan ve source_camera içermeyen key'leri al
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cpu").eval()
    print(f"[{time.strftime('%H:%M:%S')}] [MODEL] BASE Reconstruction Model loaded and set to CPU.")
except Exception as e:
    print(f"[FATAL ERROR] Reconstruction Model loading failed: {e}")
    sys.exit(1)


# -------------------- Agresif VRAM Temizliği --------------------
def aggressive_cleanup():
    """Çöp toplama ve CUDA belleğini temizleme."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    if torch.cuda.is_available():
        print(f"[{time.strftime('%H:%M:%S')}] [VRAM] Cleanup done. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [VRAM] Cleanup done (CPU mode).")

aggressive_cleanup()

# -------------------- Ön İşleme (Preprocess) --------------------
def preprocess(input_image, remove_bg):
    """Giriş görüntüsünü hazırlar: opsiyonel arka plan kaldırma, RGB dönüştürme ve yeniden boyutlandırma."""
    print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Starting preprocessing...")
    if remove_bg:
        print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Attempting to remove background with Rembg...")
        try:
            session = rembg.new_session()
            input_image = rembg.remove(input_image, session=session)
            print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Background removal successful.")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [WARNING] Rembg failed: {e}. Proceeding without removal.")
            gr.Warning("Rembg başarısız oldu. Lütfen arka planı kaldırılmış bir resim yüklemeyi deneyin.")

    input_image = input_image.convert("RGB")
    input_image = input_image.resize((320, 320))
    print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Image converted to RGB and resized to 320x320.")
    return input_image

# -------------------- OBJ/GLB Üretimi --------------------
def generate_obj_glb(input_image, steps=30, seed=42):
    """2D görüntüyü alıp, Diffusion ve Reconstruction ile 3D model üretir."""
    global device, model
    
    print(f"[{time.strftime('%H:%M:%S')}] [GEN] Starting 3D generation. Steps: {steps}, Seed: {seed}")
    
    seed_everything(seed)
    aggressive_cleanup()

    # --- AŞAMA 1: Diffusion (Zero123Plus) ---
    print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Loading Diffusion Pipeline (Zero123Plus)...")
    pipeline = None
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
            cache_dir="./ckpts/"
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        pipeline.to(device)
        print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Pipeline loaded and moved to {device}.")

        print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Running diffusion for multi-view image generation...")
        start_time = time.time()
        with torch.autocast(device.type if device.type != 'cpu' else 'cpu', dtype=torch.float16 if device.type != 'cpu' else torch.float32):
            # float16 sadece CUDA'da çalışır, CPU'da float32 kullan
            z_image = pipeline(input_image, num_inference_steps=steps).images[0]
        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Diffusion completed in {end_time - start_time:.2f} seconds.")

        z_image_np = np.array(z_image).copy()
        del z_image
        z_image = Image.fromarray(z_image_np)
        print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Multi-view image prepared for Reconstruction.")

    except Exception as e:
        error_msg = f"Diffusion Hatası ({type(e).__name__}): {str(e)}"
        print(f"[{time.strftime('%H:%M:%S')}] [FATAL ERROR] {error_msg}")
        if "CUDA capability sm_60" in str(e):
            raise gr.Error("GPU Uyumluluk Hatası: Tesla P100 (sm_60) veya benzeri uyumsuz bir GPU kullanıyorsunuz. Modeli daha uyumlu bir ortamda çalıştırın.")
        raise gr.Error(error_msg)
    finally:
        # Diffusion pipeline'ı temizle
        if pipeline is not None:
            del pipeline
        aggressive_cleanup()

    # --- AŞAMA 2: Rekonstrüksiyon (InstantMesh) ---
    print(f"[{time.strftime('%H:%M:%S')}] [RECON] Starting Reconstruction...")
    
    # Giriş tensörlerini hazırla
    img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

    # Modeli ve tensörleri cihaza taşı
    model.to(device, dtype=torch.float32)
    img_tensor = img_tensor.to(device)
    input_cameras = input_cameras.to(device)
    print(f"[{time.strftime('%H:%M:%S')}] [RECON] Model and Tensors moved to {device}.")

    # Flexicubes geometrisini başlat
    model.init_flexicubes_geometry(device, fovy=30.0)
    print(f"[{time.strftime('%H:%M:%S')}] [RECON] Flexicubes geometry initialized.")

    # Çıktı yollarını hazırla
    output_dir = "/content/outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    mesh_basename = f"mesh_{timestamp}"
    mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
    mesh_glb_fpath = os.path.join(output_dir, f"{mesh_basename}.glb")

    try:
        print(f"[{time.strftime('%H:%M:%S')}] [RECON] Running forward_planes...")
        with torch.no_grad():
            planes = model.forward_planes(img_tensor, input_cameras)
            del img_tensor, input_cameras
            aggressive_cleanup()
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] Planes generated. Extracting mesh...")

            # Extract Mesh
            # NOTE: model.extract_mesh'in çıktısının PyTorch Tensor olduğu varsayılmıştır.
            vertices, faces, vertex_colors = model.extract_mesh(planes, use_texture_map=False, **infer_config)
            del planes
            aggressive_cleanup()
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] Mesh extracted.")

            # Hata Düzeltme Alanı: Eğer extract_mesh NumPy dizisi döndürdüyse, tensöre çevir.
            # InstantMesh'in normal davranışına göre bu çıktılar zaten tensör olmalıdır.
            # Eğer hala hata alıyorsanız, aşağıda ki kontrolü açın.
            
            # if isinstance(vertices, np.ndarray):
            #     print(f"[{time.strftime('%H:%M:%S')}] [RECON] WARNING: Output is NumPy array, converting to Tensor.")
            #     vertices = torch.from_numpy(vertices).to(device)
            #     faces = torch.from_numpy(faces).to(device)
            #     vertex_colors = torch.from_numpy(vertex_colors).to(device)
            
            # Koordinat sistemini InstantMesh'e göre düzenle
            vertices = vertices[:, [1,2,0]]

            # CPU'ya taşı ve tip dönüşümü yap
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] Moving mesh data to CPU...")
            vertices_cpu = vertices.cpu().float()
            faces_cpu = faces.cpu()
            vertex_colors_cpu = vertex_colors.cpu().float()
            
            del vertices, faces, vertex_colors
            aggressive_cleanup()
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] CPU data ready. Saving files...")

            # OBJ ve GLB kaydet
            save_obj(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath)
            save_glb(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath)
            
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] OBJ saved: {mesh_fpath}")
            print(f"[{time.strftime('%H:%M:%S')}] [RECON] GLB saved: {mesh_glb_fpath}")
            print(f"[{time.strftime('%H:%M:%S')}] [GEN] Generation completed successfully.")

    except RuntimeError as e:
        error_msg = f"Rekonstrüksiyon Hatası: {e}"
        print(f"[{time.strftime('%H:%M:%S')}] [FATAL ERROR] {error_msg}")
        raise gr.Error(error_msg)
    finally:
        # Modeli CPU'ya geri taşı
        model.to("cpu")
        print(f"[{time.strftime('%H:%M:%S')}] [MODEL] Reconstruction Model moved back to CPU.")
        aggressive_cleanup()

    return mesh_fpath, mesh_glb_fpath

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("<h2><b>InstantMesh 3D Generator - BASE Model (Tesla P100)</b></h2>")
    gr.Markdown("""
**BASE model** kullanılır. (Large model yüksek VRAM gerektirir).
- Üretim adımları (**Diffusion Adımları**) arttıkça detay ve süre artar.
- GPU uyumluluğuna dikkat edin.
""")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Giriş Resmi (Objektif Odaklı, Temiz Arka Plan Önerilir)", type="pil")
            remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
            steps = gr.Slider(label="Diffusion Adımları (Daha fazla adım = Daha iyi/daha uzun)", minimum=20, maximum=50, value=30, step=5)
            seed = gr.Number(value=SEED, label="Seed", precision=0)
            generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
        with gr.Column():
            output_obj = gr.File(label="OBJ Dosyası (İndir)")
            output_glb = gr.File(label="GLB Dosyası (İndir)")

    # Hata düzeltme sonrasında generate_obj_glb fonksiyonu çağrılıyor
    # Fonksiyon çağrısı içinde ön işleme yapılıyor
    generate_btn.click(
        fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
        inputs=[input_image, remove_bg, steps, seed],
        outputs=[output_obj, output_glb]
    )

print(f"[{time.strftime('%H:%M:%S')}] [UI] Gradio UI defined.")
demo.queue(max_size=1)
demo.launch(server_name="0.0.0.0", share=True)
print("app.py başarıyla overwrite edildi!")