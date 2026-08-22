import argparse
import os
import struct
import torch
import hashlib
from torchvision import transforms
from PIL import Image
import numpy as np

from src.models.nif_codec import NIFCodec

# Magic bytes representando o formato NIF versão 1
MAGIC_BYTES = b"NIF1"


def get_checkpoint_md5_hash(checkpoint_path):
    """
    Retorna os primeiros 4 bytes do MD5 do checkpoint para validação de compatibilidade.
    Lê apenas a cabeceira do arquivo (64KB) para ser instantâneo.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return b"\x00\x00\x00\x00"
    hasher = hashlib.md5()
    try:
        with open(checkpoint_path, "rb") as f:
            chunk = f.read(65536)
            hasher.update(chunk)
        return hasher.digest()[:4]
    except Exception:
        return b"\x00\x00\x00\x00"


def save_nif_file(output_path, width, height, quality_int, z_shape, z_string, y_strings, ckpt_hash_bytes):
    """
    Serializa os dados estruturados de compressão em um arquivo binário (.nif).
    """
    with open(output_path, "wb") as f:
        # 1. Magic Bytes (4 bytes)
        f.write(MAGIC_BYTES)
        
        # 2. Cabeçalho Global (Header): 
        # Largura (2B), Altura (2B), Canais (1B, default RGB=3), Qualidade (1B), Flags (1B)
        header = struct.pack("!HHBBB", width, height, 3, quality_int, 0)
        f.write(header)
        
        # 3. Grava Hash MD5 do Checkpoint (4 bytes) para auditoria de produção
        f.write(ckpt_hash_bytes)
        
        # 4. Metadados de formato do Hyperprior: shape de z (H_z, W_z)
        f.write(struct.pack("!HH", z_shape[0], z_shape[1]))
        
        # 5. Stream do Hyperprior z
        # Gravamos o tamanho do z_string (4B) e os bytes
        z_bytes = z_string[0]
        f.write(struct.pack("!I", len(z_bytes)))
        f.write(z_bytes)
        
        # 5. Stream Principal de Latentes y (Slices + Checkerboard)
        # y_strings é uma lista de lists de bytes: [[slice_0_anchor], [slice_0_non_anchor], ...]
        # Gravamos a quantidade total de strings (1B) e em seguida o par [tamanho (4B) + bytes] para cada um
        num_strings = len(y_strings)
        f.write(struct.pack("!B", num_strings))
        
        for y_list in y_strings:
            y_bytes = y_list[0]
            f.write(struct.pack("!I", len(y_bytes)))
            f.write(y_bytes)


def load_nif_file(input_path):
    """
    Lê e desserializa o arquivo binário (.nif), retornando as dimensões, metadados e os streams de bytes.
    """
    with open(input_path, "rb") as f:
        data = f.read()
        
    offset = 0
    
    # 1. Valida Magic Bytes
    magic = data[offset:offset+4]
    offset += 4
    if magic != MAGIC_BYTES:
        raise ValueError("Formato de arquivo inválido. Assinatura 'NIF1' não encontrada.")
        
    # 2. Ler Cabeçalho Global
    width, height, channels, quality_int, flags = struct.unpack_from("!HHBBB", data, offset)
    offset += 7
    
    # 3. Ler Hash MD5 do Checkpoint (4 bytes)
    ckpt_hash_bytes = data[offset:offset+4]
    offset += 4
    
    # 4. Ler Shape do Hyperprior z
    z_h, z_w = struct.unpack_from("!HH", data, offset)
    offset += 4
    z_shape = (z_h, z_w)
    
    # 5. Ler bytes do Hyperprior
    z_len, = struct.unpack_from("!I", data, offset)
    offset += 4
    z_string = [data[offset:offset+z_len]]
    offset += z_len
    
    # 6. Ler strings do Latente Principal y
    num_strings, = struct.unpack_from("!B", data, offset)
    offset += 1
    
    y_strings = []
    for _ in range(num_strings):
        y_len, = struct.unpack_from("!I", data, offset)
        offset += 4
        y_bytes = data[offset:offset+y_len]
        offset += y_len
        y_strings.append([y_bytes])
        
    return width, height, quality_int, z_shape, z_string, y_strings, ckpt_hash_bytes


@torch.no_grad()
def compress_image(model, image_path, output_path, q_val, device, ckpt_hash_bytes):
    """
    Carrega uma imagem e gera o arquivo NIF compactado.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Redimensiona para múltiplos de 64 (exigência do autoencoder)
    w_new = (w // 64) * 64
    h_new = (h // 64) * 64
    if w != w_new or h != h_new:
        transform = transforms.CenterCrop((h_new, w_new))
        img = transform(img)
        print(f"Imagem recortada de {w}x{h} para {w_new}x{h_new} para alinhamento da rede.")
        w, h = w_new, h_new

    # Prepara input
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    q_val_quant = int(q_val * 255) / 255.0
    quality = torch.tensor([[q_val_quant]], device=device)
    
    # Roda o compressor
    out = model.compress(x, quality)
    
    # Salva no disco
    quality_int = int(q_val * 255)
    z_string = out["strings"][0]
    y_strings = out["strings"][1]
    
    save_nif_file(output_path, w, h, quality_int, out["shape"], z_string, y_strings, ckpt_hash_bytes)
    
    # Calcula estatísticas de compressão
    orig_size = os.path.getsize(image_path)
    comp_size = os.path.getsize(output_path)
    bpp = (comp_size * 8) / (w * h)
    ratio = orig_size / comp_size
    
    print("\n" + "="*50)
    print(" COMPRESSÃO CONCLUÍDA COM SUCESSO!")
    print("="*50)
    print(f"Arquivo de saída:    {output_path}")
    print(f"Resolução:           {w}x{h}")
    print(f"Fator de Qualidade:  {q_val:.2f} (Header: {quality_int})")
    print(f"Tamanho Original:    {orig_size / 1024:.2f} KB")
    print(f"Tamanho Comprimido:  {comp_size / 1024:.2f} KB")
    print(f"Taxa de Compressão:  {ratio:.2f}x (Redução de {100 * (1 - 1/ratio):.1f}%)")
    print(f"Bitrate Real (Bpp):  {bpp:.4f} bpp")
    print("="*50 + "\n")


@torch.no_grad()
def decompress_image(model, input_path, output_path, device, ckpt_hash_bytes):
    """
    Carrega o arquivo NIF e reconstrói a imagem original.
    """
    # Carrega os dados do arquivo binário
    w, h, quality_int, z_shape, z_string, y_strings, file_ckpt_hash = load_nif_file(input_path)
    
    # Valida compatibilidade do hash do checkpoint
    if file_ckpt_hash != ckpt_hash_bytes and file_ckpt_hash != b"\x00\x00\x00\x00":
        print(f"\n[WARNING] ALERTA DE COMPATIBILIDADE DE MODELO:")
        print(f"  * O arquivo foi compactado com outro checkpoint de pesos!")
        print(f"  * Hash do arquivo: {file_ckpt_hash.hex()} | Hash ativo local: {ckpt_hash_bytes.hex()}")
        print(f"  * A descompressão pode resultar em NaNs ou ruído visual severo.\n")
        
    q_val = quality_int / 255.0
    quality = torch.tensor([[q_val]], device=device)
    
    # Roda a descompressão
    strings = [z_string, y_strings]
    out = model.decompress(strings, z_shape, quality)
    
    x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)
    
    # Salva a imagem reconstruída
    x_hat_np = x_hat.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    x_hat_np = (x_hat_np * 255.0).astype(np.uint8)
    img_hat = Image.fromarray(x_hat_np)
    img_hat.save(output_path)
    
    print("\n" + "="*50)
    print(" DESCOMPRESSÃO CONCLUÍDA COM SUCESSO!")
    print("="*50)
    print(f"Arquivo desserializado: {input_path}")
    print(f"Imagem Reconstruída:    {output_path}")
    print(f"Resolução:              {w}x{h}")
    print(f"Qualidade Recuperada:   {q_val:.2f}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Utilitário de Codificação/Decodificação do Formato NIF (.nif)")
    parser.add_argument("mode", choices=["compress", "decompress"], help="Modo de operação: compress ou decompress")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pth) do modelo treinado")
    parser.add_argument("--input", type=str, required=True, help="Caminho do arquivo de entrada (.png/.jpg para compress, .nif para decompress)")
    parser.add_argument("--output", type=str, required=True, help="Caminho do arquivo de saída (.nif para compress, .png para decompress)")
    parser.add_argument("--quality", "-q", type=float, default=0.5, help="Fator de qualidade de compressão (0.1 a 1.0) - Apenas para compressão")
    parser.add_argument("--use_sigmoid", action="store_true", default=None, help="Forçar uso de restrição Sigmoid no LatentScaling (auto-detectado por padrão)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Auto-detecta uso de Sigmoid baseado no checkpoint ou flag explícita
    use_sigmoid = args.use_sigmoid if args.use_sigmoid is not None else ("v4" in args.checkpoint or "sanity" in args.checkpoint)
    
    # Inicializa modelo e carrega pesos
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8, use_sigmoid=use_sigmoid).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
        model.eval()
        # Inicializa tabelas CDF
        model.update(force=True)
    except Exception as e:
        print(f"Erro ao carregar o checkpoint: {e}")
        return

    # Obtém o hash de compatibilidade do checkpoint
    ckpt_hash = get_checkpoint_md5_hash(args.checkpoint)

    if args.mode == "compress":
        compress_image(model, args.input, args.output, args.quality, device, ckpt_hash)
    elif args.mode == "decompress":
        decompress_image(model, args.input, args.output, device, ckpt_hash)

if __name__ == "__main__":
    main()
