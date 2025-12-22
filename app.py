# Burada büyük obj ve glb üretenler calisti
import time

import os
import time
import torch
import numpy as np
import rembg
import tempfile
import gc
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import gradio as gr
import uuid

# src klasöründeki yardımcı dosyaları dahil edin. 
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras

# -------------------- Mesh Utility Fonksiyonları (Doku Desteğiyle) --------------------
# Köşe rengi boyutunu güvenli bir şekilde ele almak için güncellendi.

def safe_vertex_colors(vertex_colors):
    """
    Köşe renklerinin [N, 3] boyutunda (RGB) olmasını sağlar. 
    Eğer [N, 2] gelirse, 3. kanalı ekler.
    """
    if vertex_colors.shape[1] == 2:
        # Eğer renkler [N, 2] boyutundaysa (hataya neden olan durum), 
        # üçüncü kanalı (örneğin mavi) 1.0 (beyaz) olarak ekleyelim.
        print(f"UYARI: Köşe renkleri boyutu {vertex_colors.shape}. [N, 3] bekleniyordu. 3. kanal 1.0 olarak ekleniyor.")
        ones = torch.ones(vertex_colors.shape[0], 1, device=vertex_colors.device, dtype=vertex_colors.dtype)
        vertex_colors = torch.cat([vertex_colors, ones], dim=1)
    
    # Eğer [N, 4] (RGBA) ise, [N, 3] (RGB) kısmını alın.
    if vertex_colors.shape[1] > 3:
        vertex_colors = vertex_colors[:, :3]
        
    return vertex_colors

def save_obj(vertices, faces, vertex_colors, mesh_fpath, uvs=None, face_uvs=None, texture_map=None):
    """
    Yüksek kaliteli doku haritası kaydını destekleyen güncel save_obj implementasyonu.
    """
    
    # KÖŞE RENGİ KONTROLÜ VE DÜZELTMESİ
    vertex_colors = safe_vertex_colors(vertex_colors)
    
    mesh_dir = os.path.dirname(mesh_fpath)
    mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
    
    mtl_fpath = os.path.join(mesh_dir, f"{mesh_name}.mtl")
    texture_fpath = os.path.join(mesh_dir, f"{mesh_name}.png")

    # Doku haritasını kaydet (Tensor'den PIL Image'a dönüşüm kritik!)
    if texture_map is not None:
        tex_np = texture_map.cpu().numpy()
        
        if tex_np.ndim == 4 and tex_np.shape[0] == 1:
            tex_np = tex_np[0]
        
        # HATA DÜZELTME: NumPy dizisini (C, H, W)'den (H, W, C)'ye dönüştür
        if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
             tex_np = np.transpose(tex_np, (1, 2, 0))
            
        tex_np = (tex_np * 255).astype(np.uint8)
        
        try:
            tex_image = Image.fromarray(tex_np)
        except Exception as e:
            print(f"HATA: PIL Image oluşturma başarısız oldu. NumPy Dizisi Şekli: {tex_np.shape}, Hata: {e}")
            raise Exception(f"Doku haritası çıkarılamadı veya geçersiz boyutta ({tex_np.shape}). Lütfen farklı bir resim deneyin.") from e
            
        tex_image.save(texture_fpath)
        
        with open(mtl_fpath, "w") as f:
            f.write("newmtl material_0\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("illum 1\n")
            f.write(f"map_Kd {mesh_name}.png\n")

    # OBJ dosyasını kaydet
    with open(mesh_fpath, "w") as f:
        if texture_map is not None:
            f.write(f"mtllib {mesh_name}.mtl\n")
            f.write(f"usemtl material_0\n")

        # Vertices (Köşe noktaları) 
        for i in range(vertices.shape[0]):
            v = vertices[i]
            vc = vertex_colors[i]
            # vc'nin 3 boyutu olduğunu varsayıyoruz (safe_vertex_colors sayesinde)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vc[0]:.6f} {vc[1]:.6f} {vc[2]:.6f}\n")
        
        # UVs (Doku koordinatları)
        if uvs is not None:
            for i in range(uvs.shape[0]):
                uv = uvs[i]
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Faces (Yüzeyler)
        for i in range(faces.shape[0]):
            face = faces[i]
            if uvs is None or texture_map is None:
                # Sadece köşe indeksleri
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            else:
                # Köşe/UV indeksleri
                face_uv = face_uvs[i]
                f.write(f"f {face[0] + 1}/{face_uv[0] + 1} {face[1] + 1}/{face_uv[1] + 1} {face[2] + 1}/{face_uv[2] + 1}\n")


def save_glb(vertices, faces, vertex_colors, glb_fpath, uvs=None, face_uvs=None, texture_map=None):
    """
    Yüksek kaliteli doku haritası kaydını destekleyen güncel save_glb implementasyonu (trimesh kullanır).
    """
    
    # KÖŞE RENGİ KONTROLÜ VE DÜZELTMESİ
    vertex_colors = safe_vertex_colors(vertex_colors)
    
    try:
        if texture_map is not None:
            # TENSOR'Ü PIL IMAGE'A DÖNÜŞTÜRME (Güvenli Dönüşüm)
            tex_np = texture_map.cpu().numpy()
            
            if tex_np.ndim == 4 and tex_np.shape[0] == 1:
                tex_np = tex_np[0]

            # HATA DÜZELTME: NumPy dizisini (C, H, W)'den (H, W, C)'ye dönüştür
            if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
                 tex_np = np.transpose(tex_np, (1, 2, 0))

            tex_np = (tex_np * 255).astype(np.uint8)

            try:
                tex_image = Image.fromarray(tex_np)
            except Exception as e:
                print(f"HATA: GLB için PIL Image oluşturma başarısız oldu. NumPy Dizisi Şekli: {tex_np.shape}, Hata: {e}")
                raise Exception(f"GLB için doku haritası oluşturma başarısız oldu veya geçersiz boyutta ({tex_np.shape}). Lütfen farklı bir resim deneyin.") from e

            
            # Doku haritalı mesh oluştur
            # Trimesh'e renkleri (float [0,1] yerine uint8 [0,255] olarak) RGB veya RGBA formatında veriyoruz.
            material = trimesh.visual.TextureVisuals(image=tex_image)
            mesh = trimesh.Trimesh(
                vertices=vertices.numpy(),
                faces=faces.numpy(),
                visual=material,
                uv=uvs.numpy(),
                faces_uv=face_uvs.numpy()
            )
        else:
            # Sadece köşe renkleriyle mesh oluştur
            # Köşe renklerini 0-255 aralığına ölçekleyip uint8 formatına dönüştür.
            # safe_vertex_colors() ile boyut [N, 3] olarak ayarlandı.
            colors_255 = (vertex_colors.numpy() * 255).astype(np.uint8)
            mesh = trimesh.Trimesh(
                vertices=vertices.numpy(),
                faces=faces.numpy(),
                vertex_colors=colors_255 # Trimesh için 0-255 uint8 renkler beklenir
            )
        
        # GLB olarak kaydet
        mesh.export(glb_fpath, file_type='glb')

    except Exception as e:
        print(f"UYARI: GLB kaydı trimesh ile başarısız oldu: {e}")
        raise gr.Error(f"GLB dosyası kaydı başarısız oldu: {e}. OBJ dosyası hala mevcut olabilir.")

# -------------------- GPU / CPU --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# VRAM fragmentasyonunu azalt
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# -------------------- Seed --------------------
seed_everything(42)

# -------------------- Config (BASE Model) --------------------
# LARGE model hafıza hatası verdiği için BASE modele geri dönülüyor
config_path = "configs/instant-mesh-large.yaml" 
config = OmegaConf.load(config_path)
model_config = config.model_config
infer_config = config.infer_config

# YÜKSEK KALİTE AYARI: Doku Çözünürlüğünü (texture_size) 1024'e Yükseltme
# Bu, BASE modelinden daha keskin dokular elde etmeyi amaçlar.
# Orijinal BASE config'de 512 veya 256 olabilir.
infer_config.texture_size = 2048
print(f"[DEBUG] Doku Çözünürlüğü (texture_size) {infer_config.texture_size} olarak ayarlandı.")


# -------------------- Reconstruction Model (BASE) --------------------
print("[DEBUG] Loading LARGE Reconstruction Model...")
# BASE model checkpoint dosyası indiriliyor
model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_large.ckpt", # BASE model dosyası
    repo_type="model",
    cache_dir="./ckpts/"
)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
model.load_state_dict(state_dict, strict=False)
model = model.to("cpu").eval()
print("[DEBUG] LARGE Reconstruction Model loaded.")

# -------------------- Aggressive VRAM Cleanup --------------------
def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    if torch.cuda.is_available():
        print(f"[DEBUG] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
    else:
        print(f"[DEBUG] VRAM: 0.00 GB allocated, 0.00 GB reserved (CPU mode)")

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
def generate_obj_glb(input_image, steps=50, seed=None): # steps varsayılanı 50
    global device, model
    if seed is None:


        seed = int(time.time())

    seed_everything(seed)

    print(f"[DEBUG] Random seed: {seed}")
    aggressive_cleanup()

    pipeline = None
    try:
        print("[DEBUG] Loading Diffusion Pipeline...")
        # zero123plus-v1.2 (Diffusion Pipeline) BASE ve LARGE modeller için ortaktır
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
        # torch.autocast, hafıza verimliliği için kullanılıyor
        with torch.autocast(device.type if device.type != 'cpu' else 'cpu', dtype=torch.float16 if device.type != 'cpu' else torch.float32):
            z_image = pipeline(input_image, num_inference_steps=steps).images[0]

        z_image_np = np.array(z_image).copy()
        del z_image
        z_image = Image.fromarray(z_image_np)

    except (ImportError, AttributeError, KeyError) as e:
        error_msg = f"Bağımlılık Hatası ({type(e).__name__}): diffusers Zero123PlusPipeline veya modeli bulunamadı."
        print(f"[HATA] {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        error_msg = f"Diffusion Hatası: {type(e).__name__}: {str(e)}"
        print(f"[HATA] {error_msg}")
        if "CUDA capability sm_60" in str(e):
            raise gr.Error("GPU Uyumluluk Hatası: Tesla P100 (sm_60) uyumsuz. Kodu CPU'da çalıştırmayı deneyin.")
        raise gr.Error(error_msg)
    finally:
        if pipeline is not None:
            del pipeline
        aggressive_cleanup()

    img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

    # Modelin doğru cihazda ve veri tipinde olduğundan emin olun.
    model.to(device, dtype=torch.float32)
    img_tensor = img_tensor.to(device)
    input_cameras = input_cameras.to(device)

    model.init_flexicubes_geometry(device, fovy=30.0)

    # Output directory handling - flexible for local and server
    base_output_dir = "outputs"
    if os.path.exists("/root/InstantMesh2"):
        base_output_dir = "/root/InstantMesh2/outputs"
    else:
        base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

    os.makedirs(base_output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    mesh_basename = f"mesh_{timestamp}_{unique_id}"
    mesh_fpath = os.path.join(base_output_dir, f"{mesh_basename}.obj")
    mesh_glb_fpath = os.path.join(base_output_dir, f"{mesh_basename}.glb")

    # Debug Image Stats to verify new input
    if isinstance(input_image, Image.Image):
        # Calculate a simple hash/stat of the image to verify it changed
        img_stat = np.array(input_image).mean()
        print(f"[DEBUG] Processing Image. Size: {input_image.size}, Mean Color: {img_stat:.2f}, ID: {unique_id}")
    elif isinstance(input_image, torch.Tensor):
         print(f"[DEBUG] Processing Tensor Image. Shape: {input_image.shape}, ID: {unique_id}")

    print("[DEBUG] Extracting mesh...")
    try:
        with torch.no_grad():
            planes = model.forward_planes(img_tensor, input_cameras)
            del img_tensor, input_cameras
            aggressive_cleanup()

            # YÜKSEK DOKU ÇÖZÜNÜRLÜĞÜ (texture_size: 1024) infer_config'den alınır.
            vertices, faces, vertex_colors, uvs, texture_map = model.extract_mesh(planes, use_texture_map=True, **infer_config)
            del planes
            aggressive_cleanup()
            
            face_uvs = faces.clone()
            
            # NumPy dizisi hatasını önlemek için kontrol ve dönüştürme
            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices).to(device)
                faces = torch.from_numpy(faces).to(device)
                vertex_colors = torch.from_numpy(vertex_colors).to(device)
                uvs = torch.from_numpy(uvs).to(device)
                face_uvs = faces.clone() 

            # Koordinat sistemini dönüştür (y-z değişimi)
            vertices = vertices[:, [1,2,0]]

            # Kaydetmek için CPU'ya taşı ve uygun veri tipine dönüştür
            vertices_cpu = vertices.cpu().float()
            faces_cpu = faces.cpu()
            # ÖNEMLİ: Renkler [0, 1] aralığında ve float olarak kalmalı, düzeltme save_obj içinde yapılır.
            vertex_colors_cpu = vertex_colors.cpu().float() 
            uvs_cpu = uvs.cpu().float()
            face_uvs_cpu = face_uvs.cpu()
            
            del vertices, faces, vertex_colors, uvs, face_uvs 
            aggressive_cleanup()

            # Yerel olarak tanımlanan fonksiyonları kullanarak kaydet
            print(f"[DEBUG] OBJ Kaydediliyor: {mesh_fpath}")
            save_obj(
                vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath,
                uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
            )
            
            print(f"[DEBUG] GLB Kaydediliyor: {mesh_glb_fpath}")
            save_glb(
                vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath,
                uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
            )

        print(f"[DEBUG] OBJ: {mesh_fpath}, GLB: {mesh_glb_fpath}")
    except RuntimeError as e:
        # Hata mesajı özelleştirildi
        raise gr.Error(f"Rekonstrüksiyon Hatası: {e}. Muhtemelen GPU hafızası (VRAM) yetersiz kaldı. Model çıkarma sırasında bir sorun oluştu.")
    except Exception as e:
         # Diğer genel hataları yakala
        raise gr.Error(f"Genel Hata: {e}")
    finally:
        model.to("cpu")
        aggressive_cleanup()

    return mesh_fpath, mesh_glb_fpath

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("<h2><b>InstantMesh 3D Generator - BASE Model (MAKSİMUM KALİTE)</b></h2>")
    gr.Markdown("""
**BASE Model** kullanılıyor. LARGE model GPU kısıtlamaları nedeniyle çalıştırılamıyor.
Şu anda BASE modelinin elde edebileceği **en yüksek ayarları** kullanıyoruz:

1.  **Diffusion Adımları (Steps)**: 50 (Maksimum)
2.  **Doku Çözünürlüğü (Texture Size)**: 1024x1024 (Yüksek Detay)

- **Giriş**: Arka planı basit, objesi ortalanmış bir resim yükleyin.
""")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Giriş Resmi", type="pil")
            remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
            # Varsayılan değeri 50'ye yükseltiyoruz
            steps = gr.Slider(label="Diffusion Adımları (Maksimum 50)", minimum=20, maximum=100, value=75, step=5) 
            seed = gr.Number(value=42, label="Seed", precision=0)
            generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
        with gr.Column():
            output_obj = gr.File(label="OBJ Dosyası (Yüksek Çözünürlüklü Doku)")
            output_glb = gr.File(label="GLB Dosyası (Yüksek Çözünürlüklü Doku)")

    generate_btn.click(
        fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
        inputs=[input_image, remove_bg, steps, seed],
        outputs=[output_obj, output_glb]
    )

demo.queue(max_size=1)
demo.launch(server_name="0.0.0.0", share=True)




#Kaggle da calisan düsük kaliteli cikti veren dosya

# import os
# import time
# import torch
# import numpy as np
# import rembg
# import tempfile
# import gc
# from PIL import Image
# from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything
# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# from huggingface_hub import hf_hub_download
# import gradio as gr
# import sys

# # InstantMesh bağımlılıkları (varsayılan olarak kurulu kabul edilmiştir)
# try:
#     from src.utils.train_util import instantiate_from_config
#     from src.utils.camera_util import get_zero123plus_input_cameras
#     from src.utils.mesh_util import save_obj, save_glb
# except ImportError as e:
#     print(f"[FATAL ERROR] InstantMesh source code (src) imports failed. Ensure the InstantMesh repository structure is correct. Error: {e}")
#     sys.exit(1)


# # -------------------- GPU / CPU & Konfigürasyon --------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# # VRAM fragmentasyonunu azaltma (CUDA için)
# if device.type == 'cuda':
#     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
#     print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] PYTORCH_CUDA_ALLOC_CONF set for VRAM efficiency.")

# # -------------------- Seed --------------------
# SEED = 42
# seed_everything(SEED)
# print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Seed set to {SEED}.")

# # -------------------- Config --------------------
# config_path = "configs/instant-mesh-large.yaml"
# try:
#     print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Loading config from {config_path}...")
#     config = OmegaConf.load(config_path)
#     model_config = config.model_config
#     infer_config = config.infer_config
#     print(f"[{time.strftime('%H:%M:%S')}] [CONFIG] Config loaded successfully.")
# except Exception as e:
#     print(f"[FATAL ERROR] Config file loading failed: {e}")
#     sys.exit(1)


# # -------------------- Model Yükleme: Rekonstrüksiyon Modeli (BASE) --------------------
# print(f"[{time.strftime('%H:%M:%S')}] [MODEL] Loading LARGE Reconstruction Model...")
# try:
#     model_ckpt_path = hf_hub_download(
#         repo_id="TencentARC/InstantMesh",
#         filename="instant_mesh_large.ckpt",
#         repo_type="model",
#         cache_dir="./ckpts/"
#     )
#     model = instantiate_from_config(model_config)
#     state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
    
#     # Prefix'i kaldır ve sadece lrm_generator ile başlayan ve source_camera içermeyen key'leri al
#     state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
    
#     model.load_state_dict(state_dict, strict=False)
#     model = model.to("cpu").eval()
#     print(f"[{time.strftime('%H:%M:%S')}] [MODEL] LARGE Reconstruction Model loaded and set to CPU.")
# except Exception as e:
#     print(f"[FATAL ERROR] Reconstruction Model loading failed: {e}")
#     sys.exit(1)


# # -------------------- Agresif VRAM Temizliği --------------------
# def aggressive_cleanup():
#     """Çöp toplama ve CUDA belleğini temizleme."""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         torch.cuda.ipc_collect()
#     gc.collect()
#     if torch.cuda.is_available():
#         print(f"[{time.strftime('%H:%M:%S')}] [VRAM] Cleanup done. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
#     else:
#         print(f"[{time.strftime('%H:%M:%S')}] [VRAM] Cleanup done (CPU mode).")

# aggressive_cleanup()

# # -------------------- Ön İşleme (Preprocess) --------------------
# def preprocess(input_image, remove_bg):
#     """Giriş görüntüsünü hazırlar: opsiyonel arka plan kaldırma, RGB dönüştürme ve yeniden boyutlandırma."""
#     print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Starting preprocessing...")
#     if remove_bg:
#         print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Attempting to remove background with Rembg...")
#         try:
#             session = rembg.new_session()
#             input_image = rembg.remove(input_image, session=session)
#             print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Background removal successful.")
#         except Exception as e:
#             print(f"[{time.strftime('%H:%M:%S')}] [WARNING] Rembg failed: {e}. Proceeding without removal.")
#             gr.Warning("Rembg başarısız oldu. Lütfen arka planı kaldırılmış bir resim yüklemeyi deneyin.")

#     input_image = input_image.convert("RGB")
#     input_image = input_image.resize((320, 320))
#     print(f"[{time.strftime('%H:%M:%S')}] [PREPROC] Image converted to RGB and resized to 320x320.")
#     return input_image

# # -------------------- OBJ/GLB Üretimi --------------------
# def generate_obj_glb(input_image, steps=30, seed=42):
#     """2D görüntüyü alıp, Diffusion ve Reconstruction ile 3D model üretir."""
#     global device, model
    
#     print(f"[{time.strftime('%H:%M:%S')}] [GEN] Starting 3D generation. Steps: {steps}, Seed: {seed}")
    
#     seed_everything(seed)
#     aggressive_cleanup()

#     # --- AŞAMA 1: Diffusion (Zero123Plus) ---
#     print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Loading Diffusion Pipeline (Zero123Plus)...")
#     pipeline = None
#     try:
#         pipeline = DiffusionPipeline.from_pretrained(
#             "sudo-ai/zero123plus-v1.2",
#             custom_pipeline="zero123plus",
#             torch_dtype=torch.float16,
#             cache_dir="./ckpts/"
#         )
#         pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#             pipeline.scheduler.config, timestep_spacing="trailing"
#         )
#         pipeline.to(device)
#         print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Pipeline loaded and moved to {device}.")

#         print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Running diffusion for multi-view image generation...")
#         start_time = time.time()
#         with torch.autocast(device.type if device.type != 'cpu' else 'cpu', dtype=torch.float16 if device.type != 'cpu' else torch.float32):
#             z_image = pipeline(input_image, num_inference_steps=steps).images[0]
#         end_time = time.time()
#         print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Diffusion completed in {end_time - start_time:.2f} seconds.")

#         z_image_np = np.array(z_image).copy()
#         del z_image
#         z_image = Image.fromarray(z_image_np)
#         print(f"[{time.strftime('%H:%M:%S')}] [DIFFUSION] Multi-view image prepared for Reconstruction.")

#     except Exception as e:
#         error_msg = f"Diffusion Hatası ({type(e).__name__}): {str(e)}"
#         print(f"[{time.strftime('%H:%M:%S')}] [FATAL ERROR] {error_msg}")
#         if "CUDA capability sm_60" in str(e):
#             raise gr.Error("GPU Uyumluluk Hatası: Tesla P100 (sm_60) veya benzeri uyumsuz bir GPU kullanıyorsunuz. Modeli daha uyumlu bir ortamda çalıştırın.")
#         raise gr.Error(error_msg)
#     finally:
#         # Diffusion pipeline'ı temizle
#         if pipeline is not None:
#             del pipeline
#         aggressive_cleanup()

#     # --- AŞAMA 2: Rekonstrüksiyon (InstantMesh) ---
#     print(f"[{time.strftime('%H:%M:%S')}] [RECON] Starting Reconstruction...")
    
#     # Giriş tensörlerini hazırla
#     img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
#     input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

#     # Modeli ve tensörleri cihaza taşı
#     model.to(device, dtype=torch.float32)
#     img_tensor = img_tensor.to(device)
#     input_cameras = input_cameras.to(device)
#     print(f"[{time.strftime('%H:%M:%S')}] [RECON] Model and Tensors moved to {device}.")

#     # Flexicubes geometrisini başlat
#     model.init_flexicubes_geometry(device, fovy=30.0)
#     print(f"[{time.strftime('%H:%M:%S')}] [RECON] Flexicubes geometry initialized.")

#     # Çıktı yollarını hazırla
#     output_dir = "/root/InstantMesh2/outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = int(time.time())
#     mesh_basename = f"mesh_{timestamp}"
#     mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
#     mesh_glb_fpath = os.path.join(output_dir, f"{mesh_basename}.glb")

#     try:
#         print(f"[{time.strftime('%H:%M:%S')}] [RECON] Running forward_planes...")
#         with torch.no_grad():
#             planes = model.forward_planes(img_tensor, input_cameras)
#             del img_tensor, input_cameras
#             aggressive_cleanup()
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] Planes generated. Extracting mesh...")

#             # Extract Mesh
#             vertices, faces, vertex_colors = model.extract_mesh(planes, use_texture_map=False, **infer_config)
#             del planes
#             aggressive_cleanup()
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] Mesh extracted.")

#             # 🔥 HATA DÜZELTME: Output'un NumPy dizisi olarak geldiğini varsayarak tensöre çevir.
#             if isinstance(vertices, np.ndarray):
#                 print(f"[{time.strftime('%H:%M:%S')}] [RECON] !!! HATA DÜZELTME AKTİF: NumPy dizisini PyTorch Tensor'a çeviriliyor. !!!")
#                 vertices = torch.from_numpy(vertices).to(device)
#                 faces = torch.from_numpy(faces).to(device)
#                 vertex_colors = torch.from_numpy(vertex_colors).to(device)

#             # Koordinat sistemini InstantMesh'e göre düzenle
#             vertices = vertices[:, [1,2,0]]

#             # CPU'ya taşı ve tip dönüşümü yap
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] Moving mesh data to CPU...")
#             vertices_cpu = vertices.cpu().float()
#             faces_cpu = faces.cpu()
#             vertex_colors_cpu = vertex_colors.cpu().float()
            
#             # Belleği serbest bırak
#             del vertices, faces, vertex_colors
#             aggressive_cleanup()
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] CPU data ready. Saving files...")

#             # OBJ ve GLB kaydet
#             save_obj(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath)
#             save_glb(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath)
            
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] OBJ saved: {mesh_fpath}")
#             print(f"[{time.strftime('%H:%M:%S')}] [RECON] GLB saved: {mesh_glb_fpath}")
#             print(f"[{time.strftime('%H:%M:%S')}] [GEN] Generation completed successfully.")

#     except RuntimeError as e:
#         error_msg = f"Rekonstrüksiyon Hatası: {e}"
#         print(f"[{time.strftime('%H:%M:%S')}] [FATAL ERROR] {error_msg}")
#         raise gr.Error(error_msg)
#     finally:
#         # Modeli CPU'ya geri taşı
#         model.to("cpu")
#         print(f"[{time.strftime('%H:%M:%S')}] [MODEL] Reconstruction Model moved back to CPU.")
#         aggressive_cleanup()

#     return mesh_fpath, mesh_glb_fpath

# # -------------------- Gradio UI --------------------
# with gr.Blocks() as demo:
#     gr.Markdown("<h2><b>InstantMesh 3D Generator - BASE Model (Tesla P100)</b></h2>")
#     gr.Markdown("""
# **BASE model** kullanılır. (Large model yüksek VRAM gerektirir).
# - Üretim adımları (**Diffusion Adımları**) arttıkça detay ve süre artar.
# - GPU uyumluluğuna dikkat edin.
# """)
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(label="Giriş Resmi (Objektif Odaklı, Temiz Arka Plan Önerilir)", type="pil")
#             remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
#             steps = gr.Slider(label="Diffusion Adımları (Daha fazla adım = Daha iyi/daha uzun)", minimum=20, maximum=100, value=30, step=5)
#             seed = gr.Number(value=SEED, label="Seed", precision=0)
#             generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
#         with gr.Column():
#             output_obj = gr.File(label="OBJ Dosyası (İndir)")
#             output_glb = gr.File(label="GLB Dosyası (İndir)")

#     # Fonksiyon çağrısı içinde ön işleme yapılıyor
#     generate_btn.click(
#         fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
#         inputs=[input_image, remove_bg, steps, seed],
#         outputs=[output_obj, output_glb]
#     )

# print(f"[{time.strftime('%H:%M:%S')}] [UI] Gradio UI defined.")
# demo.queue(max_size=1)
# demo.launch(server_name="0.0.0.0", share=True)


# kaggle da ve colabda free de calismayan large model versiyonu
# import os
# import time
# import torch
# import numpy as np
# import rembg
# import tempfile
# import gc
# import trimesh
# from PIL import Image
# from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything
# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# from huggingface_hub import hf_hub_download
# import gradio as gr

# # Import necessary utilities (assuming they are correctly handled by the environment)
# from src.utils.train_util import instantiate_from_config
# from src.utils.camera_util import get_zero123plus_input_cameras

# # -------------------- Mesh Utility Functions (Doku Desteğiyle) --------------------
# # Köşe rengi boyutunu güvenli bir şekilde ele almak için güncellendi.
# def safe_vertex_colors(vertex_colors):
#     """
#     Ensures vertex colors are in [N, 3] (RGB) dimension. 
#     Handles cases where shape might be [N, 2] or [N, 4].
#     """
#     if vertex_colors.shape[1] == 2:
#         print(f"WARNING: Vertex colors shape is {vertex_colors.shape}. Expected [N, 3]. Adding 3rd channel (blue=1.0).")
#         ones = torch.ones(vertex_colors.shape[0], 1, device=vertex_colors.device, dtype=vertex_colors.dtype)
#         vertex_colors = torch.cat([vertex_colors, ones], dim=1)
    
#     if vertex_colors.shape[1] > 3:
#         vertex_colors = vertex_colors[:, :3]
        
#     return vertex_colors

# def save_obj(vertices, faces, vertex_colors, mesh_fpath, uvs=None, face_uvs=None, texture_map=None):
#     """
#     Saves the mesh as OBJ, supporting high-quality texture map.
#     """
    
#     vertex_colors = safe_vertex_colors(vertex_colors)
    
#     mesh_dir = os.path.dirname(mesh_fpath)
#     mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
    
#     mtl_fpath = os.path.join(mesh_dir, f"{mesh_name}.mtl")
#     texture_fpath = os.path.join(mesh_dir, f"{mesh_name}.png")

#     # Save the texture map
#     if texture_map is not None:
#         tex_np = texture_map.cpu().numpy()
        
#         if tex_np.ndim == 4 and tex_np.shape[0] == 1:
#             tex_np = tex_np[0]
        
#         # FIX: Transpose from (C, H, W) to (H, W, C) for PIL
#         if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
#              tex_np = np.transpose(tex_np, (1, 2, 0))
            
#         tex_np = (tex_np * 255).astype(np.uint8)
        
#         try:
#             tex_image = Image.fromarray(tex_np)
#         except Exception as e:
#             print(f"HATA: PIL Image oluşturma başarısız oldu. NumPy Dizisi Şekli: {tex_np.shape}, Hata: {e}")
#             raise Exception(f"Doku haritası çıkarılamadı veya geçersiz boyutta ({tex_np.shape}). Lütfen farklı bir resim deneyin.") from e
            
#         tex_image.save(texture_fpath)
        
#         # Write MTL file
#         with open(mtl_fpath, "w") as f:
#             f.write("newmtl material_0\n")
#             f.write("Ka 1.000 1.000 1.000\n")
#             f.write("Kd 1.000 1.000 1.000\n")
#             f.write("Ks 0.000 0.000 0.000\n")
#             f.write("illum 1\n")
#             f.write(f"map_Kd {mesh_name}.png\n")

#     # Save the OBJ file
#     with open(mesh_fpath, "w") as f:
#         if texture_map is not None:
#             f.write(f"mtllib {mesh_name}.mtl\n")
#             f.write(f"usemtl material_0\n")

#         # Vertices (Köşe noktaları) 
#         for i in range(vertices.shape[0]):
#             v = vertices[i]
#             vc = vertex_colors[i]
#             f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vc[0]:.6f} {vc[1]:.6f} {vc[2]:.6f}\n")
        
#         # UVs (Doku koordinatları)
#         if uvs is not None:
#             for i in range(uvs.shape[0]):
#                 uv = uvs[i]
#                 f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

#         # Faces (Yüzeyler)
#         for i in range(faces.shape[0]):
#             face = faces[i]
#             if uvs is None or texture_map is None:
#                 # Only vertex indices
#                 f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
#             else:
#                 # Vertex/UV indices
#                 face_uv = face_uvs[i]
#                 f.write(f"f {face[0] + 1}/{face_uv[0] + 1} {face[1] + 1}/{face_uv[1] + 1} {face[2] + 1}/{face_uv[2] + 1}\n")


# def save_glb(vertices, faces, vertex_colors, glb_fpath, uvs=None, face_uvs=None, texture_map=None):
#     """
#     Saves the mesh as GLB, supporting high-quality texture map (using trimesh).
#     """
    
#     vertex_colors = safe_vertex_colors(vertex_colors)
    
#     try:
#         if texture_map is not None:
#             # CONVERT TENSOR TO PIL IMAGE (Safe Conversion)
#             tex_np = texture_map.cpu().numpy()
            
#             if tex_np.ndim == 4 and tex_np.shape[0] == 1:
#                 tex_np = tex_np[0]

#             # FIX: Transpose from (C, H, W) to (H, W, C) for PIL
#             if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
#                  tex_np = np.transpose(tex_np, (1, 2, 0))

#             tex_np = (tex_np * 255).astype(np.uint8)

#             try:
#                 tex_image = Image.fromarray(tex_np)
#             except Exception as e:
#                 print(f"HATA: GLB için PIL Image oluşturma başarısız oldu. NumPy Dizisi Şekli: {tex_np.shape}, Hata: {e}")
#                 raise Exception(f"GLB için doku haritası oluşturma başarısız oldu veya geçersiz boyutta ({tex_np.shape}). Lütfen farklı bir resim deneyin.") from e

            
#             # Create mesh with texture map
#             material = trimesh.visual.TextureVisuals(image=tex_image)
#             mesh = trimesh.Trimesh(
#                 vertices=vertices.numpy(),
#                 faces=faces.numpy(),
#                 visual=material,
#                 uv=uvs.numpy(),
#                 faces_uv=face_uvs.numpy()
#             )
#         else:
#             # Create mesh with vertex colors only
#             colors_255 = (vertex_colors.numpy() * 255).astype(np.uint8)
#             mesh = trimesh.Trimesh(
#                 vertices=vertices.numpy(),
#                 faces=faces.numpy(),
#                 vertex_colors=colors_255
#             )
        
#         # Export as GLB
#         mesh.export(glb_fpath, file_type='glb')

#     except Exception as e:
#         print(f"UYARI: GLB kaydı trimesh ile başarısız oldu: {e}")
#         # Not raising an error here, but OBJ might still be available
#         raise gr.Error(f"GLB dosyası kaydı başarısız oldu: {e}. OBJ dosyası hala mevcut olabilir.")

# # -------------------- GPU / CPU --------------------
# # Check for CUDA availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[DEBUG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# # Minimize VRAM fragmentation
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# # -------------------- Seed --------------------
# seed_everything(42)

# # -------------------- Config (LARGE Model) --------------------
# # Using the LARGE model config for maximum detail (Requires high VRAM!)
# config_path = "configs/instant-mesh-large.yaml" 
# config = OmegaConf.load(config_path)
# model_config = config.model_config
# infer_config = config.infer_config

# # HIGH-QUALITY SETTINGS:
# # 1. Texture Resolution: Set to 2048x2048 for maximum detail (Default for LARGE is often 1024)
# infer_config.texture_size = 2048
# # 2. Grid Size: Increase for higher mesh resolution (Default is often 128)
# # Setting it higher than 128 (e.g., 256) dramatically increases VRAM usage.
# # We stick to the model's base config which is usually 128 or slightly more for LARGE.
# # We will use the default grid setting from the LARGE config to be safe.

# print(f"[DEBUG] LARGE Model yapılandırması yüklendi.")
# print(f"[DEBUG] Doku Çözünürlüğü (texture_size) {infer_config.texture_size} olarak ayarlandı.")


# # -------------------- Reconstruction Model (LARGE) --------------------
# print("[DEBUG] Loading LARGE Reconstruction Model...")
# # Downloading LARGE model checkpoint file
# model_ckpt_path = hf_hub_download(
#     repo_id="TencentARC/InstantMesh",
#     filename="instant_mesh_large.ckpt", # LARGE model file
#     repo_type="model",
#     cache_dir="./ckpts/"
# )
# model = instantiate_from_config(model_config)
# state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
# # Remove the 'lrm_generator.' prefix
# state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
# model.load_state_dict(state_dict, strict=False)
# model = model.to("cpu").eval()
# print("[DEBUG] LARGE Reconstruction Model loaded.")

# # -------------------- Aggressive VRAM Cleanup --------------------
# def aggressive_cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         torch.cuda.ipc_collect()
#     gc.collect()
#     if torch.cuda.is_available():
#         print(f"[DEBUG] VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
#     else:
#         print(f"[DEBUG] VRAM: 0.00 GB allocated, 0.00 GB reserved (CPU mode)")

# # -------------------- Preprocess --------------------
# def preprocess(input_image, remove_bg):
#     if remove_bg:
#         try:
#             session = rembg.new_session()
#             input_image = rembg.remove(input_image, session=session)
#         except Exception as e:
#             print(f"UYARI: Arka plan kaldırma (Rembg) başarısız oldu: {e}")
#             gr.Warning("Rembg başarısız oldu. Lütfen arka planı kaldırılmış bir resim yükleyin.")

#     # Resize input image to 320x320
#     input_image = input_image.convert("RGB")
#     input_image = input_image.resize((320, 320))
#     return input_image

# # -------------------- OBJ/GLB Generation --------------------
# def generate_obj_glb(input_image, steps=75, seed=42): # Increased default steps for LARGE model
#     global device, model
#     seed_everything(seed)
#     aggressive_cleanup()

#     pipeline = None
#     try:
#         print("[DEBUG] Loading Diffusion Pipeline...")
#         # zero123plus-v1.2 (Diffusion Pipeline) is common for BASE and LARGE models
#         pipeline = DiffusionPipeline.from_pretrained(
#             "sudo-ai/zero123plus-v1.2",
#             custom_pipeline="zero123plus",
#             torch_dtype=torch.float16,
#             cache_dir="./ckpts/"
#         )
#         pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#             pipeline.scheduler.config, timestep_spacing="trailing"
#         )
#         # Move pipeline to device for inference
#         pipeline.to(device)

#         print("[DEBUG] Running diffusion...")
#         # Use torch.autocast for memory efficiency
#         with torch.autocast(device.type if device.type != 'cpu' else 'cpu', dtype=torch.float16 if device.type != 'cpu' else torch.float32):
#             # Increased inference steps to 75 (if available) for better views
#             z_image = pipeline(input_image, num_inference_steps=steps).images[0] 

#         z_image_np = np.array(z_image).copy()
#         del z_image
#         z_image = Image.fromarray(z_image_np)

#     except (ImportError, AttributeError, KeyError) as e:
#         error_msg = f"Bağımlılık Hatası ({type(e).__name__}): diffusers Zero123PlusPipeline veya modeli bulunamadı."
#         print(f"[HATA] {error_msg}")
#         raise gr.Error(error_msg)
#     except Exception as e:
#         error_msg = f"Diffusion Hatası: {type(e).__name__}: {str(e)}"
#         print(f"[HATA] {error_msg}")
#         if "CUDA capability sm_60" in str(e):
#             raise gr.Error("GPU Uyumluluk Hatası: Tesla P100 (sm_60) uyumsuz. Kodu CPU'da çalıştırmayı deneyin.")
#         raise gr.Error(error_msg)
#     finally:
#         if pipeline is not None:
#             del pipeline
#         aggressive_cleanup()

#     img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
#     input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

#     # Ensure model is on the correct device and data type
#     model.to(device, dtype=torch.float32)
#     img_tensor = img_tensor.to(device)
#     input_cameras = input_cameras.to(device)

#     model.init_flexicubes_geometry(device, fovy=30.0)

#     output_dir = "/root/InstantMesh2/outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = int(time.time())
#     mesh_basename = f"mesh_{timestamp}"
#     mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
#     mesh_glb_fpath = os.path.join(output_dir, f"{mesh_basename}.glb")

#     print("[DEBUG] Extracting mesh...")
#     try:
#         with torch.no_grad():
#             planes = model.forward_planes(img_tensor, input_cameras)
#             del img_tensor, input_cameras
#             aggressive_cleanup()

#             # Using the high texture resolution (2048) set in infer_config
#             vertices, faces, vertex_colors, uvs, texture_map = model.extract_mesh(planes, use_texture_map=True, **infer_config)
#             del planes
#             aggressive_cleanup()
            
#             face_uvs = faces.clone()
            
#             # Control and conversion to prevent NumPy array errors
#             if isinstance(vertices, np.ndarray):
#                 vertices = torch.from_numpy(vertices).to(device)
#                 faces = torch.from_numpy(faces).to(device)
#                 vertex_colors = torch.from_numpy(vertex_colors).to(device)
#                 uvs = torch.from_numpy(uvs).to(device)
#                 face_uvs = faces.clone() 

#             # Coordinate system transformation (y-z swap)
#             vertices = vertices[:, [1,2,0]]

#             # Move to CPU for saving and convert to appropriate data type
#             vertices_cpu = vertices.cpu().float()
#             faces_cpu = faces.cpu()
#             # IMPORTANT: Colors remain in [0, 1] float range; fix is done in save_obj/save_glb
#             vertex_colors_cpu = vertex_colors.cpu().float() 
#             uvs_cpu = uvs.cpu().float()
#             face_uvs_cpu = face_uvs.cpu()
            
#             del vertices, faces, vertex_colors, uvs, face_uvs 
#             aggressive_cleanup()

#             # Save using locally defined functions
#             print(f"[DEBUG] OBJ Kaydediliyor: {mesh_fpath}")
#             save_obj(
#                 vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath,
#                 uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
#             )
            
#             print(f"[DEBUG] GLB Kaydediliyor: {mesh_glb_fpath}")
#             save_glb(
#                 vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath,
#                 uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
#             )

#         print(f"[DEBUG] OBJ: {mesh_fpath}, GLB: {mesh_glb_fpath}")
#     except RuntimeError as e:
#         # Custom error message for VRAM issue
#         raise gr.Error(f"Rekonstrüksiyon Hatası: {e}. Bu genellikle yetersiz GPU hafızası (VRAM) nedeniyle olur. LARGE model yüksek VRAM gerektirir.")
#     except Exception as e:
#          # Catch other general errors
#         raise gr.Error(f"Genel Hata: {e}")
#     finally:
#         model.to("cpu")
#         aggressive_cleanup()

#     return mesh_fpath, mesh_glb_fpath

# # -------------------- Gradio UI --------------------
# with gr.Blocks() as demo:
#     gr.Markdown("<h2><b>InstantMesh 3D Generator - LARGE Model (Maksimum Kalite Ayarı)</b></h2>")
#     gr.Markdown("""
# Bu uygulama **InstantMesh LARGE Modelini** kullanır ve en yüksek kalite için ayarlanmıştır.

# **UYARI:** Bu modelin çalışması için genellikle **16 GB veya daha fazla VRAM'e** sahip bir GPU gereklidir. Önceki denemelerde karşılaşılan `CUDA out of memory` hatalarını alırsanız, bu model mevcut donanımınız için uygun olmayabilir.

# **Ayarlar:**

# 1.  **Diffusion Adımları (Steps)**: 75 (Daha fazla görünüm ve detay)
# 2.  **Doku Çözünürlüğü (Texture Size)**: 2048x2048 (Maksimum Detay)

# - **Giriş**: Arka planı basit, objesi ortalanmış bir resim yükleyin.
# """)
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(label="Giriş Resmi", type="pil")
#             remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
#             # Increased maximum steps for the LARGE model
#             steps = gr.Slider(label="Diffusion Adımları (Daha İyi Görünüm için 75 Önerilir)", minimum=20, maximum=100, value=75, step=5) 
#             seed = gr.Number(value=42, label="Seed", precision=0)
#             generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
#         with gr.Column():
#             output_obj = gr.File(label="OBJ Dosyası (Yüksek Çözünürlüklü Doku)")
#             output_glb = gr.File(label="GLB Dosyası (Yüksek Çözünürlüklü Doku)")

#     generate_btn.click(
#         fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
#         inputs=[input_image, remove_bg, steps, seed],
#         outputs=[output_obj, output_glb]
#     )

# demo.queue(max_size=1)
# demo.launch(server_name="0.0.0.0", share=True)






# optimize edilmisRTX 4000 Ada (20 GB VRAM, 8 vCPU, 32 GB RAM) için VRAM dostu, hızlı ve temiz
# import os
# import time
# import torch
# import numpy as np
# import rembg
# import gc
# import trimesh
# from PIL import Image
# from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything
# from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# from huggingface_hub import hf_hub_download
# import gradio as gr

# # src klasöründeki yardımcı dosyaları dahil edin.
# from src.utils.train_util import instantiate_from_config
# from src.utils.camera_util import get_zero123plus_input_cameras

# # -------------------- Genel Ayarlar --------------------
# # CUDA prefer edilir (RTX 4000 Ada)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[DEBUG] CUDA Available: {torch.cuda.is_available()}, Device: {device}")

# # Fragmentation azaltma (iyi bir başlangıç)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# # Deterministik davranış
# seed_everything(42)

# # -------------------- Mesh Utility Fonksiyonları --------------------
# def safe_vertex_colors(vertex_colors):
#     if vertex_colors is None:
#         return None
#     # Eğer torch Tensor değilse önce tensor yap
#     if isinstance(vertex_colors, np.ndarray):
#         vertex_colors = torch.from_numpy(vertex_colors)
#     if vertex_colors.ndim == 1:
#         vertex_colors = vertex_colors.unsqueeze(0)
#     if vertex_colors.shape[1] == 2:
#         print(f"UYARI: Köşe renkleri boyutu {vertex_colors.shape}. [N,3] bekleniyordu. 3. kanal 1.0 ekleniyor.")
#         ones = torch.ones(vertex_colors.shape[0], 1, device=vertex_colors.device, dtype=vertex_colors.dtype)
#         vertex_colors = torch.cat([vertex_colors, ones], dim=1)
#     if vertex_colors.shape[1] > 3:
#         vertex_colors = vertex_colors[:, :3]
#     return vertex_colors

# def save_obj(vertices, faces, vertex_colors, mesh_fpath, uvs=None, face_uvs=None, texture_map=None):
#     vertex_colors = safe_vertex_colors(vertex_colors)
#     mesh_dir = os.path.dirname(mesh_fpath)
#     os.makedirs(mesh_dir, exist_ok=True)
#     mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
#     mtl_fpath = os.path.join(mesh_dir, f"{mesh_name}.mtl")
#     texture_fpath = os.path.join(mesh_dir, f"{mesh_name}.png")

#     if texture_map is not None:
#         tex_np = texture_map.cpu().numpy()
#         if tex_np.ndim == 4 and tex_np.shape[0] == 1:
#             tex_np = tex_np[0]
#         if tex_np.ndim == 3 and tex_np.shape[0] in [3,4]:
#             tex_np = np.transpose(tex_np, (1,2,0))
#         tex_np = (np.clip(tex_np, 0.0, 1.0) * 255).astype(np.uint8)
#         try:
#             tex_image = Image.fromarray(tex_np)
#         except Exception as e:
#             raise Exception(f"Doku haritası oluşturulamadı: shape={tex_np.shape}, err={e}") from e
#         tex_image.save(texture_fpath)
#         with open(mtl_fpath, "w") as f:
#             f.write("newmtl material_0\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\nillum 1\n")
#             f.write(f"map_Kd {mesh_name}.png\n")

#     with open(mesh_fpath, "w") as f:
#         if texture_map is not None:
#             f.write(f"mtllib {mesh_name}.mtl\nusemtl material_0\n")
#         for i in range(vertices.shape[0]):
#             v = vertices[i]
#             vc = vertex_colors[i] if vertex_colors is not None else (0.8,0.8,0.8)
#             f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vc[0]:.6f} {vc[1]:.6f} {vc[2]:.6f}\n")
#         if uvs is not None:
#             for i in range(uvs.shape[0]):
#                 uv = uvs[i]
#                 f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
#         for i in range(faces.shape[0]):
#             face = faces[i]
#             if uvs is None or texture_map is None:
#                 f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
#             else:
#                 face_uv = face_uvs[i]
#                 f.write(f"f {face[0]+1}/{face_uv[0]+1} {face[1]+1}/{face_uv[1]+1} {face[2]+1}/{face_uv[2]+1}\n")

# def save_glb(vertices, faces, vertex_colors, glb_fpath, uvs=None, face_uvs=None, texture_map=None):
#     vertex_colors = safe_vertex_colors(vertex_colors)
#     try:
#         if texture_map is not None:
#             tex_np = texture_map.cpu().numpy()
#             if tex_np.ndim == 4 and tex_np.shape[0] == 1:
#                 tex_np = tex_np[0]
#             if tex_np.ndim == 3 and tex_np.shape[0] in [3,4]:
#                 tex_np = np.transpose(tex_np, (1,2,0))
#             tex_np = (np.clip(tex_np, 0.0, 1.0) * 255).astype(np.uint8)
#             tex_image = Image.fromarray(tex_np)
#             material = trimesh.visual.TextureVisuals(image=tex_image)
#             mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(),
#                                    visual=material, uv=uvs.numpy(), faces_uv=face_uvs.numpy())
#         else:
#             colors_255 = (vertex_colors.numpy() * 255).astype(np.uint8)
#             mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(),
#                                    vertex_colors=colors_255)
#         mesh.export(glb_fpath, file_type='glb')
#     except Exception as e:
#         raise gr.Error(f"GLB kaydı başarısız: {e}")

# # -------------------- Config & Model --------------------
# config_path = "configs/instant-mesh-large.yaml"
# config = OmegaConf.load(config_path)
# model_config = config.model_config
# infer_config = config.infer_config

# # Yüksek detay doku çözünürlüğü
# infer_config.texture_size = 2048
# print(f"[DEBUG] texture_size = {infer_config.texture_size}")

# # Reconstruction model checkpoint (LARGE)
# print("[DEBUG] Loading Reconstruction Model (LARGE checkpoint)...")
# model_ckpt_path = hf_hub_download(
#     repo_id="TencentARC/InstantMesh",
#     filename="instant_mesh_large.ckpt",   # LARGE checkpoint (dosya mevcut olmalıdır)
#     repo_type="model",
#     cache_dir="./ckpts/"
# )

# model = instantiate_from_config(model_config)
# state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
# state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.") and "source_camera" not in k}
# model.load_state_dict(state_dict, strict=False)
# model = model.to("cpu").eval()
# print("[DEBUG] Reconstruction model (LARGE) loaded to CPU (ready).")

# # -------------------- Cleanup Helper --------------------
# def aggressive_cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         print(f"[DEBUG] VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
#     else:
#         print("[DEBUG] CPU mode (no CUDA)")

# # -------------------- Preprocess --------------------
# def preprocess(input_image, remove_bg):
#     if remove_bg:
#         try:
#             session = rembg.new_session()
#             input_image = rembg.remove(input_image, session=session)
#         except Exception as e:
#             print(f"UYARI: Rembg hatası: {e}")
#             gr.Warning("Rembg başarısız oldu. Lütfen arka planı kaldırılmış bir resim yükleyin.")
#     input_image = input_image.convert("RGB")
#     input_image = input_image.resize((320, 320))
#     return input_image

# # -------------------- OBJ/GLB Üretimi (RTX 4000 Ada optimize) --------------------
# def generate_obj_glb(input_image, steps=50, seed=42):
#     global device, model
#     seed_everything(seed)
#     aggressive_cleanup()

#     pipeline = None
#     try:
#         print("[DEBUG] Loading Zero123Plus pipeline (FP16, RTX4000-optimized)...")
#         pipeline = DiffusionPipeline.from_pretrained(
#             "sudo-ai/zero123plus-v1.2",
#             custom_pipeline="zero123plus",
#             torch_dtype=torch.float16,
#             cache_dir="./ckpts/"
#         )

#         # Scheduler (trailing spacing)
#         pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
#             pipeline.scheduler.config, timestep_spacing="trailing"
#         )

#         # Memory/VRAM optimizasyonu:
#         # - attention slicing VRAM'i düşürür (hafif performans maliyeti)
#         pipeline.enable_attention_slicing()
#         # - Eğer xformers yüklüyse, memory-efficient attention'ı aç (daha hızlı ve daha az VRAM)
#         try:
#             pipeline.enable_xformers_memory_efficient_attention()
#             print("[DEBUG] xformers memory efficient attention enabled.")
#         except Exception:
#             print("[DEBUG] xformers not available or could not be enabled (ok to ignore).")

#         # Doğrudan CUDA'ya taşı (RTX 4000 için ideal)
#         pipeline.to(device)
#         print("[DEBUG] Pipeline moved to device:", device)

#         # İnference (FP16 amp cast)
#         print("[DEBUG] Running diffusion...")
#         if device.type == "cuda":
#             # torch.cuda.amp.autocast ile FP16 hız/VRAM kazancı
#             with torch.cuda.amp.autocast():
#                 result = pipeline(input_image, num_inference_steps=steps)
#         else:
#             # CPU fallback (yavaş)
#             with torch.cpu.amp.autocast(enabled=False):
#                 result = pipeline(input_image, num_inference_steps=steps)

#         z_image = result.images[0]
#         z_image_np = np.array(z_image).copy()
#         del z_image, result
#         aggressive_cleanup()
#         z_image = Image.fromarray(z_image_np)

#     except (ImportError, AttributeError, KeyError) as e:
#         error_msg = f"Bağımlılık Hatası: {type(e).__name__}: {e}"
#         print(f"[HATA] {error_msg}")
#         raise gr.Error(error_msg)
#     except Exception as e:
#         error_msg = f"Diffusion Hatası: {type(e).__name__}: {str(e)}"
#         print(f"[HATA] {error_msg}")
#         raise gr.Error(error_msg)
#     finally:
#         if pipeline is not None:
#             # pipeline'i serbest bırak (model zaten GPU'da olabilir)
#             try:
#                 pipeline.to("cpu")
#             except Exception:
#                 pass
#             del pipeline
#         aggressive_cleanup()

#     # Tensor'a çevir
#     img_tensor = torch.from_numpy(np.array(z_image)).permute(2,0,1).unsqueeze(0).float() / 255.0
#     input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)

#     # Model reconstruction için cihaza taşı
#     model.to(device, dtype=torch.float32)
#     img_tensor = img_tensor.to(device)
#     input_cameras = input_cameras.to(device)

#     model.init_flexicubes_geometry(device, fovy=30.0)

#     output_dir = "./outputs"
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = int(time.time())
#     mesh_basename = f"mesh_{timestamp}"
#     mesh_fpath = os.path.join(output_dir, f"{mesh_basename}.obj")
#     mesh_glb_fpath = os.path.join(output_dir, f"{mesh_basename}.glb")

#     print("[DEBUG] Extracting mesh from model...")
#     try:
#         with torch.no_grad():
#             planes = model.forward_planes(img_tensor, input_cameras)
#             del img_tensor, input_cameras
#             aggressive_cleanup()

#             vertices, faces, vertex_colors, uvs, texture_map = model.extract_mesh(
#                 planes, use_texture_map=True, **infer_config
#             )
#             del planes
#             aggressive_cleanup()

#             face_uvs = faces.clone()

#             # Eğer numpy ise tensor'a çevir
#             if isinstance(vertices, np.ndarray):
#                 vertices = torch.from_numpy(vertices).to(device)
#                 faces = torch.from_numpy(faces).to(device)
#                 vertex_colors = torch.from_numpy(vertex_colors).to(device)
#                 uvs = torch.from_numpy(uvs).to(device)
#                 face_uvs = faces.clone()

#             # Koordinat dönüşümü (y-z değişimi)
#             vertices = vertices[:, [1,2,0]]

#             # CPU'ya taşıyıp kaydet
#             vertices_cpu = vertices.cpu().float()
#             faces_cpu = faces.cpu()
#             vertex_colors_cpu = vertex_colors.cpu().float()
#             uvs_cpu = uvs.cpu().float()
#             face_uvs_cpu = face_uvs.cpu()

#             # Serbest bırak
#             del vertices, faces, vertex_colors, uvs, face_uvs
#             aggressive_cleanup()

#             print(f"[DEBUG] Saving OBJ -> {mesh_fpath}")
#             save_obj(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_fpath,
#                      uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map)

#             print(f"[DEBUG] Saving GLB -> {mesh_glb_fpath}")
#             save_glb(vertices_cpu, faces_cpu, vertex_colors_cpu, mesh_glb_fpath,
#                      uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map)

#         print(f"[DEBUG] Done. OBJ: {mesh_fpath}, GLB: {mesh_glb_fpath}")
#     except RuntimeError as e:
#         raise gr.Error(f"Rekonstrüksiyon Hatası (RuntimeError): {e}. Muhtemelen VRAM veya model iç hatası.")
#     except Exception as e:
#         raise gr.Error(f"Genel Hata: {e}")
#     finally:
#         # Modeli CPU'ya al, VRAM temizle
#         model.to("cpu")
#         aggressive_cleanup()

#     return mesh_fpath, mesh_glb_fpath

# # -------------------- Gradio UI --------------------
# with gr.Blocks() as demo:
#     gr.Markdown("<h2><b>InstantMesh 3D Generator - RTX 4000 Ada (LARGE optimized)</b></h2>")
#     gr.Markdown("""
# **LARGE model** kullanılıyor ve RTX 4000 Ada için optimize edildi.

# - Diffusion Steps: 20-50 (daha yüksek adım = daha detaylı, daha fazla süre)
# - Texture Size: 1024x1024 (yüksek detay)
# """)
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(label="Giriş Resmi", type="pil")
#             remove_bg = gr.Checkbox(label="Arka Planı Kaldır (Rembg)", value=True)
#             steps = gr.Slider(label="Diffusion Adımları (Steps)", minimum=20, maximum=100, value=40, step=5)
#             seed = gr.Number(value=42, label="Seed", precision=0)
#             generate_btn = gr.Button("3D Model Oluştur (OBJ/GLB)", variant="primary")
#         with gr.Column():
#             output_obj = gr.File(label="OBJ Dosyası")
#             output_glb = gr.File(label="GLB Dosyası")

#     generate_btn.click(
#         fn=lambda img, rm, s, sd: generate_obj_glb(preprocess(img, rm), s, int(sd)),
#         inputs=[input_image, remove_bg, steps, seed],
#         outputs=[output_obj, output_glb]
#     )

# demo.queue(max_size=1)
# demo.launch(server_name="0.0.0.0", share=True)
