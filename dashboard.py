import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
from pytorch_msssim import MS_SSIM

import struct
from src.models.nif_codec import NIFCodec
from src.models.entropy_model import split_checkerboard, merge_checkerboard, unpack_gmm_string, pack_gmm_string

# Force CUDA determinism
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model globally
checkpoint_path = 'checkpoints/nif_epoch_300.pth'
model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
if os.path.exists(checkpoint_path):
    print(f"Carregando checkpoint de {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.update(force=True)
    print("Modelo carregado e atualizado com sucesso!")
else:
    print("AVISO: Checkpoint nif_epoch_300.pth nao encontrado. Certifique-se de que ele esta na pasta checkpoints/.")

ms_ssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def pil_to_base64(img, format="PNG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['image']
    q_val = float(request.form.get('quality', 0.5))
    
    try:
        img_bytes = file.read()
        orig_size = len(img_bytes)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Crop alignment (divisible by 64)
        w, h = img.size
        w_new = (w // 64) * 64
        h_new = (h // 64) * 64
        
        if w != w_new or h != h_new:
            transform = transforms.Compose([
                transforms.CenterCrop((h_new, w_new)),
                transforms.ToTensor()
            ])
            x = transform(img).unsqueeze(0).to(device)
            # Recreate original image matching cropped dimensions
            x_np = x.squeeze(0).cpu().permute(1, 2, 0).numpy()
            x_np = (x_np * 255.0).astype(np.uint8)
            img_cropped = Image.fromarray(x_np)
            orig_size_bytes = len(io.BytesIO(img_bytes).getvalue())  # estimate or use cropped
        else:
            transform = transforms.ToTensor()
            x = transform(img).unsqueeze(0).to(device)
            img_cropped = img
        
        B, C, H, W = x.size()
        num_pixels = B * H * W
        
        # Quantize quality
        q_val_quant = int(q_val * 255) / 255.0
        quality = torch.tensor([[q_val_quant]], device=device)
        
        with torch.no_grad():
            out = model.compress(x, quality)
            
            # Local decompress for metrics
            z_strings = out["strings"][0]
            y_strings = out["strings"][1]
            z_shape = out["shape"]
            
            out_dec = model.decompress([z_strings, y_strings], z_shape, quality)
            x_hat = torch.clamp(out_dec["x_hat"], 0.0, 1.0)
            
            # Metrics
            psnr = compute_psnr(x, x_hat)
            msssim = ms_ssim_metric(x, x_hat).item()
            
            # Write to a buffer using struct in the exact format as save_nif_file
            buffer = bytearray()
            # 1. Magic Bytes (4 bytes)
            buffer.extend(b"NIF1")
            # 2. Header: Width (2B), Height (2B), Channels (1B), Quality (1B), Flags (1B)
            buffer.extend(struct.pack("!HHBBB", W, H, 3, int(q_val_quant * 255), 0))
            # 3. Shape of z
            buffer.extend(struct.pack("!HH", z_shape[0], z_shape[1]))
            # 4. Stream z
            z_bytes = z_strings[0]
            buffer.extend(struct.pack("!I", len(z_bytes)))
            buffer.extend(z_bytes)
            # 5. Stream y
            num_strings = len(y_strings)
            buffer.extend(struct.pack("!B", num_strings))
            for y_list in y_strings:
                y_bytes = y_list[0]
                buffer.extend(struct.pack("!I", len(y_bytes)))
                buffer.extend(y_bytes)
                
            compressed_bytes = bytes(buffer)
            comp_size = len(compressed_bytes)
            
            # bpp
            bpp = (comp_size * 8) / num_pixels
            
            # Base64 representations
            orig_base64 = pil_to_base64(img_cropped)
            
            # Reconstructed image to base64
            x_hat_np = x_hat.squeeze(0).cpu().permute(1, 2, 0).numpy()
            x_hat_np = (x_hat_np * 255.0).astype(np.uint8)
            img_rec = Image.fromarray(x_hat_np)
            rec_base64 = pil_to_base64(img_rec)
            
            # Generate the compressed binary as base64 for download
            nif_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
            
            return jsonify({
                'resolution': f"{w_new}x{h_new}",
                'orig_size_kb': round(orig_size / 1024.0, 2),
                'comp_size_kb': round(comp_size / 1024.0, 2),
                'ratio': round(orig_size / comp_size, 2),
                'bpp': round(bpp, 4),
                'psnr': round(psnr, 2) if psnr != float('inf') else "Lossless",
                'msssim': round(msssim, 5),
                'orig_img': orig_base64,
                'rec_img': rec_base64,
                'nif_file': nif_base64,
                'filename': f"imagem_q{int(q_val*100)}.nif"
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Erro no processamento: {str(e)}"}), 500

@app.route('/decompress', methods=['POST'])
def decompress_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo .nif enviado'}), 400
        
    file = request.files['file']
    try:
        nif_bytes = file.read()
        comp_size = len(nif_bytes)
        offset = 0
        
        # 1. Valida Magic Bytes
        magic = nif_bytes[offset:offset+4]
        offset += 4
        if magic != b"NIF1":
            return jsonify({'error': 'Arquivo NIF inválido ou corrompido (MAGIC mismatch)'}), 400
            
        # 2. Ler Cabeçalho Global
        width, height, channels, quality_int, flags = struct.unpack_from("!HHBBB", nif_bytes, offset)
        offset += 7
        
        q_val = quality_int / 255.0
        quality = torch.tensor([[q_val]], device=device)
        
        # 3. Ler Shape do Hyperprior z
        z_h, z_w = struct.unpack_from("!HH", nif_bytes, offset)
        offset += 4
        z_shape = (z_h, z_w)
        
        # 4. Ler bytes do Hyperprior
        z_len, = struct.unpack_from("!I", nif_bytes, offset)
        offset += 4
        z_strings = [nif_bytes[offset:offset+z_len]]
        offset += z_len
        
        # 5. Ler strings do Latente Principal y
        num_strings, = struct.unpack_from("!B", nif_bytes, offset)
        offset += 1
        
        y_strings = []
        for _ in range(num_strings):
            y_len, = struct.unpack_from("!I", nif_bytes, offset)
            offset += 4
            y_bytes = nif_bytes[offset:offset+y_len]
            offset += y_len
            y_strings.append([y_bytes])
            
        H, W = height, width
        
        with torch.no_grad():
            out_dec = model.decompress([z_strings, y_strings], z_shape, quality)
            x_hat = torch.clamp(out_dec["x_hat"], 0.0, 1.0)
            
            # Reconstructed image to base64
            x_hat_np = x_hat.squeeze(0).cpu().permute(1, 2, 0).numpy()
            x_hat_np = (x_hat_np * 255.0).astype(np.uint8)
            img_rec = Image.fromarray(x_hat_np)
            rec_base64 = pil_to_base64(img_rec)
            
            return jsonify({
                'resolution': f"{W}x{H}",
                'quality': round(q_val, 2),
                'comp_size_kb': round(comp_size / 1024.0, 2),
                'rec_img': rec_base64
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Erro na descompressao: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
