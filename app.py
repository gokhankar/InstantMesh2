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

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_glb

# -------------------- GPU / CPU --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# VRAM fragmentasyonunu azalt
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# -------------------- Seed --------------------
seed_everything(42)

# -------------------- Config --------------------
config_path = "configs/instant-mesh-base.yaml"
config = OmegaConf.load(config_path)
model_config = config.model_config
infer_config = config.infer_config

# -------------------- Reconstruction Model --------------------
print("[DEBUG] Loading BASE Reconstruction Model...")
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_base.ckpt",
    repo_type="model",
    cache_dir="./ckpts/"
)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
model.load_state_dict(state_dict, strict=False)
model = model.to("cpu").eval()
print("[DEBUG] BASE Reconstruction Model loaded.")

# -------------------- Aggressive VRAM Cleanup --------------------
def aggressive_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    gc.collect()
    print(f"[DEBUG] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

# -------------------- Preprocess --------------------
def preprocess(input_image, remove_bg):
    if remove_bg:
        try:
            session = rembg.new_session()
            input_image = rembg.remove(input_image, session=session)
        except Exception as e:
            print(f"UYARI: Arka plan kaldırma (Rembg) başarısız oldu: {e}")
            gr.Warning("Rembg başarısız oldu. Lütfen arka planı kaldırılmış bir resim yükleyin.")

    input_image = input_image.convert("RGB")
    input_image = input_image.resize((320, 320))
    return input_image

# -------------------- OBJ/GLB Generation --------------------
def generate_obj_glb(input_image, steps=30, seed=42):
    global device, model
    seed_everything(seed)
    aggressive_cleanup()

    pipeline = None
    try:
        print("[DEBUG] Loading Diffusion Pipeline...")
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

        print("[DEBUG] Running diffusion...")
        with torch.autocast(device.type, dtype=torch.float16):
            z_image = pipeline(input_image, num_inference_steps=steps).images[0]

        z_image_np = np.array(z_image).copy()
        del z_image
        z_image = Image.fromarray(z_image_np)

    except (ImportError, AttributeError) as e:
        error_msg = f"Bağımlılık Hatası ({type(e).__name__}): diffusers Zero123PlusPipeline bulunamadı veya kütüphane uyumsuzluğu var."
        print(f"[HATA] {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        error_msg = f"Diffusion Hatası: {type(e).__name__}: {str(e)}"
        print(f"[HATA] {error_msg}")
        if "CUDA capability sm_60" in str(e):
            raise gr.Error("GPU Uyumluluk Hatası: Tesla P100 (sm_60) uyumsuz.")
        raise gr.Error(error_msg)
    finally:
        if pipeline is not None:
            del pipeline
        aggressive_cleanup()

    img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

    model.to(device, dtype=torch.float32)
    img_tensor = img_tensor.to(device)
    input_cameras = input_cameras.to(device)

    model.init_flexicubes_geometry(device, fovy=30.0)

    output_dir = "/content/outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    mesh_basename = f"mesh_{timestamp}"
    mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
    mesh_glb_fpath = os.path.join(output_dir, f"{mesh_basename}.glb")

    print("[DEBUG] Extracting mesh...")
    try:
        with torch.no_grad():
            planes = model.forward_planes(img_tensor, input_cameras)
            del img_tensor, input_cameras
            aggressive_cleanup()

            vertices, faces, vertex_colors = model.extract_mesh(planes, use_texture_map=False, **infer_config)
            del planes
            aggressive_cleanup()

            vertices = vertices[:, [1,2,0]]

            vertices_cpu = vertices.cpu().float()
            faces_cpu = faces.cpu()
            vertex_colors_cpu = vertex_colors.cpu().float()
            del vertices, faces, vertex_colors
            aggressive_cleanup()

            save_obj(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath)
            save_glb(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath)

        print(f"[DEBUG] OBJ: {mesh_fpath}, GLB: {mesh_glb_fpath}")
    except RuntimeError as e:
        raise gr.Error(f"Rekonstrüksiyon Hatası: {e}")
    finally:
        model.to("cpu")
        aggressive_cleanup()

    return mesh_fpath, mesh_glb_fpath

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("<h2><b>InstantMesh 3D Generator - BASE Model (Tesla P100)</b></h2>")
    gr.Markdown("""
**BASE model** kullanır (Large model P100 GPU’da çalışmaz).
- Kütüphane uyumsuzluklarına dikkat edin (`diffusers`, `accelerate`, `peft`).
- Detay seviyesi için steps değerini artırabilirsiniz.
""")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Giriş Resmi", type="pil")
            remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
            steps = gr.Slider(label="Diffusion Adımları", minimum=20, maximum=50, value=30, step=5)
            seed = gr.Number(value=42, label="Seed", precision=0)
            generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
        with gr.Column():
            output_obj = gr.File(label="OBJ Dosyası")
            output_glb = gr.File(label="GLB Dosyası")

    generate_btn.click(
        fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
        inputs=[input_image, remove_bg, steps, seed],
        outputs=[output_obj, output_glb]
    )

demo.queue(max_size=1)
demo.launch(server_name="0.0.0.0", share=True)