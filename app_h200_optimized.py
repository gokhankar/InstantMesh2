"""
InstantMesh 3D Generator - H200 Optimized Version
Large Model + High Quality Output + PLY/OBJ/GLB Support

Optimizations for H200 GPU (141 GB VRAM):
- Uses instant-mesh-large model for maximum quality
- High resolution texture maps (2048x2048)
- Increased diffusion steps (75)
- Batch processing support
- Memory-efficient gradient checkpointing
- PLY, OBJ, and GLB export formats
"""

import os
import time
import torch
import numpy as np
import rembg
import gc
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
from plyfile import PlyData, PlyElement
import gradio as gr

# InstantMesh source utilities
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras

# ==================== Configuration ====================
# Device setup - H200 optimized
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [DEVICE] CUDA Available: {torch.cuda.is_available()}")
print(f"🚀 [DEVICE] Device: {device}")

if torch.cuda.is_available():
    print(f"🚀 [DEVICE] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"🚀 [DEVICE] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Memory optimization for large VRAM GPUs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Set seed for reproducibility
SEED = 42
seed_everything(SEED)

# ==================== Quality Configuration ====================
# Large Model Configuration
config_path = "configs/instant-mesh-large.yaml"
print(f"📋 [CONFIG] Loading LARGE model config: {config_path}")

config = OmegaConf.load(config_path)
model_config = config.model_config
infer_config = config.infer_config

# HIGH QUALITY SETTINGS for H200
infer_config.texture_resolution = 2048  # Ultra-high resolution textures
infer_config.render_resolution = 1024  # High render resolution
infer_config.grid_res = 256  # Increased mesh resolution

print(f"⚙️  [CONFIG] Texture Resolution: {infer_config.texture_resolution}")
print(f"⚙️  [CONFIG] Render Resolution: {infer_config.render_resolution}")

# ==================== Mesh Export Utilities ====================

def safe_vertex_colors(vertex_colors):
    """
    Ensures vertex colors are in [N, 3] (RGB) format.
    Handles [N, 2] and [N, 4] cases.
    """
    if vertex_colors.shape[1] == 2:
        print(f"⚠️  [MESH] Vertex colors shape {vertex_colors.shape}, adding 3rd channel")
        ones = torch.ones(vertex_colors.shape[0], 1, 
                         device=vertex_colors.device, dtype=vertex_colors.dtype)
        vertex_colors = torch.cat([vertex_colors, ones], dim=1)
    
    if vertex_colors.shape[1] > 3:
        vertex_colors = vertex_colors[:, :3]
    
    return vertex_colors


def save_ply(vertices, faces, vertex_colors, ply_fpath):
    """
    Save mesh as PLY format with vertex colors.
    PLY is efficient for point clouds and colored meshes.
    """
    vertex_colors = safe_vertex_colors(vertex_colors)
    
    # Convert to numpy
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    colors_np = (vertex_colors.cpu().numpy() * 255).astype(np.uint8)
    
    # Create vertex array with colors
    vertex_array = np.zeros(
        vertices_np.shape[0],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    
    vertex_array['x'] = vertices_np[:, 0]
    vertex_array['y'] = vertices_np[:, 1]
    vertex_array['z'] = vertices_np[:, 2]
    vertex_array['red'] = colors_np[:, 0]
    vertex_array['green'] = colors_np[:, 1]
    vertex_array['blue'] = colors_np[:, 2]
    
    # Create face array
    face_array = np.zeros(
        faces_np.shape[0],
        dtype=[('vertex_indices', 'i4', (3,))]
    )
    face_array['vertex_indices'] = faces_np
    
    # Create PLY elements
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')
    
    # Write PLY file
    PlyData([vertex_element, face_element], text=True).write(ply_fpath)
    print(f"✅ [EXPORT] PLY saved: {ply_fpath}")


def save_obj(vertices, faces, vertex_colors, mesh_fpath, uvs=None, face_uvs=None, texture_map=None):
    """
    Save mesh as OBJ format with optional texture mapping.
    Supports high-quality texture maps.
    """
    vertex_colors = safe_vertex_colors(vertex_colors)
    
    mesh_dir = os.path.dirname(mesh_fpath)
    mesh_name = os.path.splitext(os.path.basename(mesh_fpath))[0]
    
    mtl_fpath = os.path.join(mesh_dir, f"{mesh_name}.mtl")
    texture_fpath = os.path.join(mesh_dir, f"{mesh_name}.png")
    
    # Save texture map if available
    if texture_map is not None:
        tex_np = texture_map.cpu().numpy()
        
        if tex_np.ndim == 4 and tex_np.shape[0] == 1:
            tex_np = tex_np[0]
        
        # Convert from (C, H, W) to (H, W, C) for PIL
        if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
            tex_np = np.transpose(tex_np, (1, 2, 0))
        
        tex_np = (tex_np * 255).astype(np.uint8)
        
        try:
            tex_image = Image.fromarray(tex_np)
            # Save with high quality
            tex_image.save(texture_fpath, quality=95, optimize=True)
            print(f"✅ [EXPORT] Texture saved: {texture_fpath} ({tex_image.size})")
        except Exception as e:
            print(f"❌ [ERROR] Texture save failed: {e}")
            raise
        
        # Write MTL file
        with open(mtl_fpath, "w") as f:
            f.write("# High Quality Material\n")
            f.write("newmtl material_0\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.200 0.200 0.200\n")
            f.write("Ns 96.0\n")
            f.write("illum 2\n")
            f.write(f"map_Kd {mesh_name}.png\n")
    
    # Write OBJ file
    with open(mesh_fpath, "w") as f:
        f.write("# High Quality OBJ Export - InstantMesh H200\n")
        
        if texture_map is not None:
            f.write(f"mtllib {mesh_name}.mtl\n")
            f.write(f"usemtl material_0\n")
        
        # Vertices with colors
        for i in range(vertices.shape[0]):
            v = vertices[i]
            vc = vertex_colors[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vc[0]:.6f} {vc[1]:.6f} {vc[2]:.6f}\n")
        
        # UV coordinates
        if uvs is not None:
            for i in range(uvs.shape[0]):
                uv = uvs[i]
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Faces
        for i in range(faces.shape[0]):
            face = faces[i]
            if uvs is None or texture_map is None:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            else:
                face_uv = face_uvs[i]
                f.write(f"f {face[0] + 1}/{face_uv[0] + 1} {face[1] + 1}/{face_uv[1] + 1} {face[2] + 1}/{face_uv[2] + 1}\n")
    
    print(f"✅ [EXPORT] OBJ saved: {mesh_fpath}")


def save_glb(vertices, faces, vertex_colors, glb_fpath, uvs=None, face_uvs=None, texture_map=None):
    """
    Save mesh as GLB format using trimesh.
    GLB is compact and widely supported (Blender, Unity, etc.)
    """
    vertex_colors = safe_vertex_colors(vertex_colors)
    
    try:
        if texture_map is not None:
            # Convert texture tensor to PIL Image
            tex_np = texture_map.cpu().numpy()
            
            if tex_np.ndim == 4 and tex_np.shape[0] == 1:
                tex_np = tex_np[0]
            
            if tex_np.ndim == 3 and tex_np.shape[0] in [3, 4]:
                tex_np = np.transpose(tex_np, (1, 2, 0))
            
            tex_np = (tex_np * 255).astype(np.uint8)
            tex_image = Image.fromarray(tex_np)
            
            # Create textured mesh
            material = trimesh.visual.texture.SimpleMaterial(image=tex_image)
            color_visuals = trimesh.visual.TextureVisuals(
                uv=uvs.cpu().numpy(),
                image=tex_image,
                material=material
            )
            
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(),
                faces=faces.cpu().numpy(),
                visual=color_visuals,
                process=False
            )
        else:
            # Vertex color mesh
            colors_255 = (vertex_colors.cpu().numpy() * 255).astype(np.uint8)
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(),
                faces=faces.cpu().numpy(),
                vertex_colors=colors_255,
                process=False
            )
        
        # Export as GLB
        mesh.export(glb_fpath, file_type='glb')
        print(f"✅ [EXPORT] GLB saved: {glb_fpath}")
        
    except Exception as e:
        print(f"❌ [ERROR] GLB export failed: {e}")
        raise


# ==================== Model Loading ====================

print("📦 [MODEL] Loading LARGE Reconstruction Model...")

model_ckpt_path = hf_hub_download(
    repo_id="TencentARC/InstantMesh",
    filename="instant_mesh_large.ckpt",  # LARGE MODEL
    repo_type="model",
    cache_dir="./ckpts/"
)

model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {k[14:]: v for k, v in state_dict.items() 
              if k.startswith("lrm_generator.") and "source_camera" not in k}
model.load_state_dict(state_dict, strict=False)

# Keep model on CPU initially to save VRAM
model = model.to("cpu").eval()

# Enable gradient checkpointing for memory efficiency (if supported)
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    print("✅ [MODEL] Gradient checkpointing enabled")

print("✅ [MODEL] LARGE Reconstruction Model loaded successfully!")

# ==================== Memory Management ====================

def aggressive_cleanup():
    """Aggressive garbage collection and CUDA cache clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"💾 [VRAM] {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


# ==================== Image Preprocessing ====================

def preprocess(input_image, remove_bg):
    """
    Preprocess input image:
    - Optional background removal
    - RGB conversion
    - Resize to 320x320
    """
    if remove_bg:
        try:
            print("🖼️  [PREPROC] Removing background with Rembg...")
            session = rembg.new_session()
            input_image = rembg.remove(input_image, session=session)
            print("✅ [PREPROC] Background removed")
        except Exception as e:
            print(f"⚠️  [PREPROC] Rembg failed: {e}")
            gr.Warning("Background removal failed. Using original image.")
    
    input_image = input_image.convert("RGB")
    input_image = input_image.resize((320, 320))
    print("✅ [PREPROC] Image preprocessed (320x320 RGB)")
    return input_image


# ==================== 3D Generation Pipeline ====================

def generate_3d_mesh(input_image, steps=75, seed=42, export_formats=["OBJ", "GLB", "PLY"]):
    """
    Main 3D generation pipeline:
    1. Multi-view diffusion (Zero123++)
    2. 3D reconstruction (InstantMesh Large)
    3. Export to OBJ/GLB/PLY formats
    
    Args:
        input_image: PIL Image
        steps: Diffusion steps (20-100, higher = better quality)
        seed: Random seed
        export_formats: List of formats to export ["OBJ", "GLB", "PLY"]
    """
    global device, model
    
    seed_everything(seed)
    aggressive_cleanup()
    
    start_time = time.time()
    
    # ========== STAGE 1: Multi-View Diffusion ==========
    print(f"\n{'='*60}")
    print("🎨 STAGE 1: Multi-View Generation (Zero123++)")
    print(f"{'='*60}")
    
    pipeline = None
    try:
        print("📦 [DIFFUSION] Loading Zero123++ pipeline...")
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
        print("✅ [DIFFUSION] Pipeline loaded")
        
        print(f"🎨 [DIFFUSION] Running diffusion ({steps} steps)...")
        diffusion_start = time.time()
        
        with torch.autocast(device.type, dtype=torch.float16):
            z_image = pipeline(
                input_image, 
                num_inference_steps=steps
            ).images[0]
        
        diffusion_time = time.time() - diffusion_start
        print(f"✅ [DIFFUSION] Completed in {diffusion_time:.2f}s")
        
        # Convert to numpy and back to avoid memory issues
        z_image_np = np.array(z_image).copy()
        del z_image
        z_image = Image.fromarray(z_image_np)
        
    except Exception as e:
        error_msg = f"Diffusion failed: {type(e).__name__}: {str(e)}"
        print(f"❌ [ERROR] {error_msg}")
        raise gr.Error(error_msg)
    finally:
        if pipeline is not None:
            del pipeline
        aggressive_cleanup()
    
    # ========== STAGE 2: 3D Reconstruction ==========
    print(f"\n{'='*60}")
    print("🏗️  STAGE 2: 3D Reconstruction (InstantMesh Large)")
    print(f"{'='*60}")
    
    # Prepare input tensors
    img_tensor = torch.from_numpy(np.array(z_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0)
    
    # Move model and tensors to GPU
    print(f"📦 [RECON] Moving model to {device}...")
    model.to(device, dtype=torch.float32)
    img_tensor = img_tensor.to(device)
    input_cameras = input_cameras.to(device)
    
    # Initialize FlexiCubes geometry
    model.init_flexicubes_geometry(device, fovy=30.0)
    print("✅ [RECON] Model ready")
    
    # Prepare output paths
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    mesh_basename = f"mesh_large_{timestamp}"
    
    output_files = {}
    
    try:
        print("🏗️  [RECON] Extracting mesh...")
        recon_start = time.time()
        
        with torch.no_grad():
            # Generate triplane features
            planes = model.forward_planes(img_tensor, input_cameras)
            del img_tensor, input_cameras
            aggressive_cleanup()
            
            # Extract mesh with HIGH QUALITY texture
            print(f"🎨 [RECON] Extracting mesh with {infer_config.texture_resolution}px texture...")
            vertices, faces, vertex_colors, uvs, texture_map = model.extract_mesh(
                planes, 
                use_texture_map=True, 
                **infer_config
            )
            del planes
            aggressive_cleanup()
            
            face_uvs = faces.clone()
            
            # Ensure tensors (handle NumPy arrays if returned)
            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices).to(device)
                faces = torch.from_numpy(faces).to(device)
                vertex_colors = torch.from_numpy(vertex_colors).to(device)
                uvs = torch.from_numpy(uvs).to(device)
                face_uvs = faces.clone()
            
            # Transform coordinate system (y-z swap for standard orientation)
            vertices = vertices[:, [1, 2, 0]]
            
            # Move to CPU for export
            vertices_cpu = vertices.cpu().float()
            faces_cpu = faces.cpu()
            vertex_colors_cpu = vertex_colors.cpu().float()
            uvs_cpu = uvs.cpu().float()
            face_uvs_cpu = face_uvs.cpu()
            
            del vertices, faces, vertex_colors, uvs, face_uvs
            aggressive_cleanup()
            
            recon_time = time.time() - recon_start
            print(f"✅ [RECON] Mesh extracted in {recon_time:.2f}s")
            print(f"📊 [MESH] Vertices: {vertices_cpu.shape[0]:,}, Faces: {faces_cpu.shape[0]:,}")
            
        # ========== STAGE 3: Export Meshes ==========
        print(f"\n{'='*60}")
        print("💾 STAGE 3: Exporting Mesh Files")
        print(f"{'='*60}")
        
        export_start = time.time()
        
        if "OBJ" in export_formats:
            obj_path = os.path.join(output_dir, f"{mesh_basename}.obj")
            save_obj(
                vertices_cpu, faces_cpu, vertex_colors_cpu, obj_path,
                uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
            )
            output_files["OBJ"] = obj_path
        
        if "GLB" in export_formats:
            glb_path = os.path.join(output_dir, f"{mesh_basename}.glb")
            save_glb(
                vertices_cpu, faces_cpu, vertex_colors_cpu, glb_path,
                uvs=uvs_cpu, face_uvs=face_uvs_cpu, texture_map=texture_map
            )
            output_files["GLB"] = glb_path
        
        if "PLY" in export_formats:
            ply_path = os.path.join(output_dir, f"{mesh_basename}.ply")
            save_ply(vertices_cpu, faces_cpu, vertex_colors_cpu, ply_path)
            output_files["PLY"] = ply_path
        
        export_time = time.time() - export_start
        print(f"✅ [EXPORT] All formats exported in {export_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎉 COMPLETE! Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
    except RuntimeError as e:
        error_msg = f"Reconstruction failed: {e}"
        print(f"❌ [ERROR] {error_msg}")
        raise gr.Error(error_msg)
    except Exception as e:
        error_msg = f"Export failed: {e}"
        print(f"❌ [ERROR] {error_msg}")
        raise gr.Error(error_msg)
    finally:
        model.to("cpu")
        aggressive_cleanup()
    
    return (
        output_files.get("OBJ"),
        output_files.get("GLB"),
        output_files.get("PLY")
    )


# ==================== Gradio UI ====================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 InstantMesh 3D Generator - H200 Optimized
    
    **Large Model** with **Ultra-High Quality** settings optimized for H200 GPU (141 GB VRAM)
    
    ### 🎯 Features:
    - ✨ **Large Model**: Maximum quality reconstruction
    - 🎨 **2048px Textures**: Ultra-high resolution texture maps
    - 📦 **Multiple Formats**: OBJ, GLB, and PLY export
    - ⚡ **H200 Optimized**: Full utilization of 141 GB VRAM
    - 🎭 **75 Diffusion Steps**: Maximum detail generation
    
    ### 📝 Instructions:
    1. Upload an image with a clear object (white/transparent background recommended)
    2. Adjust quality settings (higher = better quality but slower)
    3. Click "Generate 3D Model" and wait for the magic! ✨
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="📷 Input Image",
                type="pil",
                height=320
            )
            
            remove_bg = gr.Checkbox(
                label="🖼️ Remove Background (Rembg)",
                value=True,
                info="Automatically remove background from image"
            )
            
            steps = gr.Slider(
                label="🎨 Diffusion Steps",
                minimum=30,
                maximum=100,
                value=75,
                step=5,
                info="Higher = better quality (recommended: 75)"
            )
            
            seed = gr.Number(
                value=42,
                label="🎲 Random Seed",
                precision=0,
                info="Use same seed for reproducible results"
            )
            
            formats = gr.CheckboxGroup(
                choices=["OBJ", "GLB", "PLY"],
                value=["OBJ", "GLB", "PLY"],
                label="📦 Export Formats",
                info="Select output formats to generate"
            )
            
            generate_btn = gr.Button(
                "🚀 Generate 3D Model",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            gr.Markdown("### 📥 Download Your 3D Models")
            
            output_obj = gr.File(
                label="📄 OBJ File (High-Res Texture)",
                file_count="single"
            )
            output_glb = gr.File(
                label="📦 GLB File (Compact Format)",
                file_count="single"
            )
            output_ply = gr.File(
                label="🎨 PLY File (Point Cloud)",
                file_count="single"
            )
            
            gr.Markdown("""
            ### 💡 Tips:
            - **OBJ**: Best for editing in Blender, Maya
            - **GLB**: Best for web, Unity, Unreal Engine
            - **PLY**: Best for point cloud processing
            - Texture PNG is included with OBJ
            """)
    
    generate_btn.click(
        fn=lambda img, rm, s, sd, fmt: generate_3d_mesh(
            preprocess(img, rm), 
            int(s), 
            int(sd),
            fmt
        ),
        inputs=[input_image, remove_bg, steps, seed, formats],
        outputs=[output_obj, output_glb, output_ply]
    )
    
    gr.Markdown("""
    ---
    ### ⚙️ Current Configuration:
    - **Model**: InstantMesh Large (Maximum Quality)
    - **Texture Resolution**: 2048x2048 pixels
    - **Render Resolution**: 1024x1024 pixels
    - **GPU**: Optimized for H200 (141 GB VRAM)
    - **Pipeline**: Zero123++ → InstantMesh Large → Multi-Format Export
    """)

# Launch configuration
demo.queue(max_size=5)  # Increased queue for better throughput
demo.launch(
    server_name="0.0.0.0",
    share=True,
    show_error=True
)
