# InstantMesh H200 Optimized Setup Guide

## 🚀 Makine Özellikleri

✅ **Mevcut Sistem:**
- GPU: H200 (141 GB VRAM)
- CPU: 24 vCPU
- RAM: 240 GB
- Boot Disk: 720 GB NVMe
- Scratch Disk: 5 TB NVMe

✅ **Desteklenen Model:** instant-mesh-large (Maksimum Kalite)

## 📦 Kurulum Adımları

### 1. Conda Ortamı Oluşturma

```bash
# Yeni conda ortamı oluştur
conda create --name instantmesh python=3.10
conda activate instantmesh

# Ninja derleyicisi (C++ kodları için gerekli)
conda install Ninja

# CUDA 12.1 kurulumu
conda install cuda -c nvidia/label/cuda-12.1.0
```

### 2. PyTorch ve Bağımlılıklar

```bash
# PyTorch ve xformers (CUDA 12.1 için)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# Diğer bağımlılıklar
pip install -r requirements.txt
```

### 3. Model Checkpoint'lerini İndirme

Model checkpoint'leri otomatik olarak indirilecek, ancak manuel indirmek isterseniz:

```bash
# Klasör oluştur
mkdir -p ckpts

# Large model checkpoint
wget https://huggingface.co/TencentARC/InstantMesh/resolve/main/instant_mesh_large.ckpt -O ckpts/instant_mesh_large.ckpt

# Diffusion model (Zero123++)
wget https://huggingface.co/TencentARC/InstantMesh/resolve/main/diffusion_pytorch_model.bin -O ckpts/diffusion_pytorch_model.bin
```

## 🎯 Çalıştırma

### H200 Optimized Versiyonu (ÖNERİLEN)

```bash
python app_h200_optimized.py
```

### Orijinal Large Model Versiyonu

```bash
python app.py
```

## ⚙️ Konfigürasyon Ayarları

### Yüksek Kalite Modunda:

```yaml
# configs/instant-mesh-large.yaml
infer_config:
  texture_resolution: 2048    # Ultra-yüksek çözünürlük (H200'de çalışır)
  render_resolution: 1024     # Yüksek render kalitesi
  grid_res: 256              # Detaylı mesh
```

### Diffusion Ayarları:

```python
# app_h200_optimized.py içinde
steps = 75                    # Diffusion adımları (30-100 arası)
seed = 42                     # Reproducibility için
```

## 📊 Beklenen Performans

### T4 GPU ile Karşılaştırma:

| Metrik | T4 (16GB) | H200 (141GB) |
|--------|-----------|--------------|
| Model | Base Only | **Large (Recommended)** |
| Texture Resolution | 512px | **2048px** |
| Diffusion Steps | 30-40 | **75-100** |
| Mesh Quality | Düşük | **Ultra Yüksek** |
| İşlem Süresi | ~5-10 dk | ~2-4 dk |
| Batch Size | 1 | 1-4 |
| VRAM Kullanımı | >14GB (Yetersiz) | ~30-50GB (Rahat) |

### H200 ile Avantajlar:

✅ **9x Daha Fazla VRAM**: 141 GB vs 16 GB
✅ **Large Model**: Daha yüksek kalite mesh üretimi
✅ **2048px Textures**: 4x daha yüksek çözünürlük
✅ **Batch Processing**: Aynı anda birden fazla görüntü
✅ **No OOM Errors**: Bellek taşması yok
✅ **Faster Training**: Fine-tuning imkanı

## 📁 Çıktı Formatları

### OBJ Format (.obj + .mtl + .png)
- **Kullanım**: Blender, Maya, 3ds Max
- **Avantajlar**: Düzenlenebilir, texture desteği
- **Boyut**: Orta (~5-20 MB)

### GLB Format (.glb)
- **Kullanım**: Unity, Unreal Engine, Web (Three.js)
- **Avantajlar**: Kompakt, tek dosya
- **Boyut**: Küçük (~2-10 MB)

### PLY Format (.ply)
- **Kullanım**: Point cloud processing, MeshLab
- **Avantajlar**: Vertex colors, hafif
- **Boyut**: Çok Küçük (~1-5 MB)

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory

```bash
# Texture resolution'ı düşür
# configs/instant-mesh-large.yaml içinde:
texture_resolution: 1024  # 2048 yerine

# veya Diffusion steps'i azalt:
steps = 50  # 75 yerine
```

### Problem: Model İndirme Hatası

```bash
# Hugging Face token ile manuel indirme
huggingface-cli login
huggingface-cli download TencentARC/InstantMesh --local-dir ./ckpts/
```

### Problem: Gradio Share Link Çalışmıyor

```bash
# Firewall ayarları:
sudo ufw allow 7860

# Veya local çalıştır:
# app_h200_optimized.py içinde:
demo.launch(server_name="127.0.0.1", share=False)
```

## 📈 Kalite İyileştirme İpuçları

### Giriş Görüntüsü:
1. ✅ Arka plan temiz (beyaz/transparan)
2. ✅ Obje merkezi ve net
3. ✅ İyi ışıklandırma
4. ✅ Yüksek çözünürlük (en az 512x512)
5. ❌ Gürültülü arka plan
6. ❌ Çoklu objeler

### Parametre Optimizasyonu:
```python
# Maksimum Kalite:
steps = 100
texture_resolution = 2048
render_resolution = 1024

# Dengeli (Hız + Kalite):
steps = 75
texture_resolution = 2048
render_resolution = 512

# Hızlı Test:
steps = 50
texture_resolution = 1024
render_resolution = 512
```

## 🎨 Örnek Kullanım

```bash
# 1. Ortamı aktifleştir
conda activate instantmesh

# 2. Uygulamayı başlat
python app_h200_optimized.py

# 3. Tarayıcıda aç
# Local: http://localhost:7860
# Public: Gradio share link (terminalde görünecek)

# 4. Görüntü yükle ve parametreleri ayarla
# 5. "Generate 3D Model" butonuna tıkla
# 6. OBJ/GLB/PLY dosyalarını indir
```

## 💡 İleri Seviye Kullanım

### Fine-tuning (Kendi Modelini Eğit):

```bash
# Zero123++ fine-tuning
python train.py --base configs/zero123plus-finetune.yaml --gpus 0 --num_nodes 1

# InstantMesh Large training (çok dataset gerekli)
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0 --num_nodes 1
```

### Batch Processing:

```python
# Çoklu görüntü işleme
import glob
from PIL import Image

images = glob.glob("input_images/*.png")
for img_path in images:
    img = Image.open(img_path)
    output = generate_3d_mesh(img, steps=75, seed=42)
    print(f"Processed: {img_path}")
```

## 📚 Referanslar

- **InstantMesh Paper**: https://arxiv.org/abs/2404.07191
- **Hugging Face Model**: https://huggingface.co/TencentARC/InstantMesh
- **GitHub Repo**: https://github.com/TencentARC/InstantMesh
- **Zero123++**: https://github.com/SUDO-AI-3D/zero123plus

## 🆘 Destek

Sorun yaşarsanız:
1. GitHub Issues: https://github.com/TencentARC/InstantMesh/issues
2. Hugging Face Discussions: https://huggingface.co/TencentARC/InstantMesh/discussions
3. Log dosyalarını kontrol edin
4. VRAM kullanımını izleyin: `nvidia-smi`

---

**Not**: H200 GPU ile Large model'i rahatça çalıştırabilirsiniz. T4'te yaşadığınız sorunlar artık olmayacak! 🎉
