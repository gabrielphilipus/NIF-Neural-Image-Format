import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import glob

from src.models.nif_codec import NIFCodec
from pytorch_msssim import MS_SSIM
import lpips

class NIFLoss(nn.Module):
    """
    Função de perda multi-objetivo adaptativa baseada em Taxa-Distorção (R-D).
    Calcula a taxa de bits (bpp) e combina MSE, MS-SSIM e LPIPS mascarado por SFM.
    O multiplicador de Lagrange (lambda) é ajustado dinamicamente para cada imagem
    com base no nível de qualidade amostrado.
    """
    def __init__(self, use_lpips=True, device="cpu"):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.use_lpips = use_lpips
        if use_lpips:
            # LPIPS baseado em AlexNet ou VGG (AlexNet é mais rápido e consome menos memória)
            self.lpips_metric = lpips.LPIPS(net='alex', spatial=True).to(device)
            # Congela pesos do extrator LPIPS
            for p in self.lpips_metric.parameters():
                p.requires_grad = False

    def forward(self, x, x_hat, likelihoods, sfm, q, lambda_min=10.0, lambda_max=1000.0):
        B, C, H, W = x.size()
        num_pixels = B * H * W

        # 1. Taxa de bits (Rate Loss): Bits per Pixel (bpp)
        # bpp = -log2(p) / num_pixels
        bpp_y = torch.log(likelihoods["y"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=x.device)))
        bpp_z = torch.log(likelihoods["z"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=x.device)))
        rate_loss = bpp_y + bpp_z

        # 2. Perda de Distorção (Distortion Loss)
        # MSE pura (fidelidade pixel-a-pixel)
        mse_loss = nn.functional.mse_loss(x, x_hat)

        # MS-SSIM (fidelidade de estrutura e brilho perceptual)
        # MS_SSIM retorna valor de 0 a 1 (1 é perfeito), por isso minimizamos 1 - MS-SSIM
        msssim_val = self.ms_ssim(x, x_hat)
        msssim_loss = 1.0 - msssim_val

        # LPIPS mascarada por SFM (evitar alucinação generativa)
        if self.use_lpips:
            # lpips_metric retorna mapa de erro espacial [B, 1, H_feat, W_feat]
            lpips_map = self.lpips_metric(x, x_hat)
            if lpips_map.shape[-2:] != sfm.shape[-2:]:
                sfm_resized = nn.functional.interpolate(sfm, size=lpips_map.shape[-2:], mode='bilinear', align_corners=False)
            else:
                sfm_resized = sfm
            
            # Em áreas com alta fidelidade estrutural (sfm próximo de 1), atenuamos a perda perceptual pura
            # para dar preferência a MS-SSIM e MSE rígidos.
            # Em áreas de textura estocástica (sfm próximo de 0), permitimos que o LPIPS guie a otimização.
            weighted_lpips = (lpips_map * (1.0 - sfm_resized)).mean()
        else:
            weighted_lpips = torch.tensor(0.0, device=x.device)

        # 3. Mapeamento Exponencial de Lambda para Taxa Variável
        # lambda(q) = lambda_min * (lambda_max / lambda_min) ^ q
        lambda_val = lambda_min * ((lambda_max / lambda_min) ** q)
        lambda_mean = lambda_val.mean()

        # Distorção total balanceada
        distortion = 0.4 * mse_loss + 0.4 * msssim_loss
        if self.use_lpips:
            distortion = distortion + 0.2 * weighted_lpips

        total_loss = rate_loss + lambda_mean * distortion

        return total_loss, rate_loss, mse_loss, msssim_loss, weighted_lpips


class ImageFolderCustom(Dataset):
    """
    Carregador de imagens customizado que aceita tanto imagens diretamente
    no diretório principal (flat) quanto organizadas em subpastas.
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        
        self.image_paths = []
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP')
        for ext in extensions:
            # Imagens diretamente na pasta
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            # Imagens em subpastas de nível 1
            self.image_paths.extend(glob.glob(os.path.join(folder_path, "*", ext)))
            
        self.image_paths = sorted(list(set(self.image_paths)))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em {folder_path}. "
                               f"Formatos suportados: PNG, JPG, JPEG, BMP.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Retorna imagem e label dummy para manter compatibilidade com a assinatura do DataLoader
        return img, 0


class SafeResize(object):
    """
    Redimensiona a imagem caso ela seja menor do que o tamanho mínimo de crop (size),
    preservando a proporção original. Se já for maior ou igual, mantém intacta.
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if w < self.size or h < self.size:
            # Encontra o fator de escala para que a menor dimensão seja 'size'
            scale = self.size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Garante que ambas sejam pelo menos 'size'
            new_w = max(new_w, self.size)
            new_h = max(new_h, self.size)
            return img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return img


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento NIF")
    parser.add_argument("--dataset", type=str, required=True, help="Caminho para o diretório de imagens de treino")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do lote (batch size)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate inicial")
    parser.add_argument("--lambda_min", type=float, default=10.0, help="Lambda mínimo da curva RD")
    parser.add_argument("--lambda_max", type=float, default=1000.0, help="Lambda máximo da curva RD")
    parser.add_argument("--no_lpips", action="store_true", help="Desativar perda perceptual LPIPS")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Pasta para salvar checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Treinando no dispositivo: {device}")

    # Cria diretório de checkpoints
    os.makedirs(args.save_path, exist_ok=True)
    
    # Inicializa o TensorBoard SummaryWriter
    log_dir = os.path.join(args.save_path, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard habilitado. Logs salvos em: {log_dir}")

    # 1. Transformações e Data Loader
    # Cortar imagens para 256x256 é padrão na literatura de compressão neural
    transform = transforms.Compose([
        SafeResize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    try:
        # Usa o carregador customizado para aceitar pastas de imagens planas
        dataset = ImageFolderCustom(args.dataset, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Dataset carregado com {len(dataset)} imagens de treino.")
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return

    # 2. Inicialização do Modelo, Otimizador e Perda
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
    
    # CompressAI possui um otimizador específico ou parâmetros auxiliares no modelo
    # Parâmetros de rede vs Parâmetros do modelo de entropia (quantização de hyperprior)
    parameters = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]

    optimizer = optim.Adam(parameters, lr=args.lr)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    criterion = NIFLoss(use_lpips=not args.no_lpips, device=device)

    # 3. Loop de Treinamento
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_bpp = 0
        epoch_mse = 0
        epoch_msssim = 0
        epoch_lpips = 0

        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            B = x.size(0)

            # Amostra qualidade q uniformemente entre 0.1 e 1.0 para suporte a taxa de bits variável
            q = torch.rand(B, 1, device=device) * 0.9 + 0.1

            # Forward pass
            out = model(x, q)
            
            x_hat = out["x_hat"]
            likelihoods = out["likelihoods"]
            sfm = out["sfm"]

            # Limitar x_hat entre 0 e 1 antes de calcular a distorção
            x_hat = torch.clamp(x_hat, 0.0, 1.0)

            # Cálculo da perda
            loss, rate, mse, msssim, weighted_lpips = criterion(
                x, x_hat, likelihoods, sfm, q, 
                lambda_min=args.lambda_min, 
                lambda_max=args.lambda_max
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilização de treino
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()

            # Passo de otimizador auxiliar do CompressAI (responsável pelas tabelas de probabilidade)
            aux_optimizer.zero_grad()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # Acumula métricas
            epoch_loss += loss.item()
            epoch_bpp += rate.item()
            epoch_mse += mse.item()
            epoch_msssim += (1.0 - msssim.item()) # Converte de volta para MS-SSIM real (0 a 1)
            epoch_lpips += weighted_lpips.item()

            # Log a cada lote no TensorBoard (passo global)
            global_step = epoch * len(dataloader) + i
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_Bpp", rate.item(), global_step)
            writer.add_scalar("Train/Batch_MSE", mse.item(), global_step)
            writer.add_scalar("Train/Batch_MS-SSIM", 1.0 - msssim.item(), global_step)
            if not args.no_lpips:
                writer.add_scalar("Train/Batch_LPIPS", weighted_lpips.item(), global_step)

            # Gravação visual imediata do primeiro lote ou a cada 100 lotes
            if global_step == 0 or global_step % 100 == 0:
                with torch.no_grad():
                    num_display = min(4, B)
                    orig_grid = make_grid(x[:num_display], normalize=True)
                    recon_grid = make_grid(x_hat[:num_display], normalize=True)
                    sfm_grid = make_grid(sfm[:num_display], normalize=True)
                    
                    writer.add_image("Visual/Original", orig_grid, global_step)
                    writer.add_image("Visual/Reconstructed", recon_grid, global_step)
                    writer.add_image("Visual/Structural_Fidelity_Mask", sfm_grid, global_step)

            if i % 10 == 0:
                print(f"Época [{epoch+1}/{args.epochs}] | Batch [{i}/{len(dataloader)}] | "
                      f"Loss: {loss.item():.4f} | Bpp: {rate.item():.4f} | MSE: {mse.item():.6f} | "
                      f"MS-SSIM: {(1.0 - msssim.item()):.4f} | LPIPS: {weighted_lpips.item():.4f}")

        lr_scheduler.step()

        # Resumo da época
        num_batches = len(dataloader)
        print(f"==== Fim da Época {epoch+1} ====")
        print(f"Média - Loss: {epoch_loss/num_batches:.4f} | Bpp: {epoch_bpp/num_batches:.4f} | "
              f"MSE: {epoch_mse/num_batches:.6f} | MS-SSIM: {epoch_msssim/num_batches:.4f} | "
              f"LPIPS: {epoch_lpips/num_batches:.4f}")

        # Log de métricas médias por época no TensorBoard
        writer.add_scalar("Epoch/Loss", epoch_loss / num_batches, epoch + 1)
        writer.add_scalar("Epoch/Bpp", epoch_bpp / num_batches, epoch + 1)
        writer.add_scalar("Epoch/MSE", epoch_mse / num_batches, epoch + 1)
        writer.add_scalar("Epoch/MS-SSIM", epoch_msssim / num_batches, epoch + 1)
        if not args.no_lpips:
            writer.add_scalar("Epoch/LPIPS", epoch_lpips / num_batches, epoch + 1)

        # Log visual removido daqui pois agora ocorre durante a época no global_step 0 e de 100 em 100 lotes

        # Salva o checkpoint a cada 5 épocas
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_path, f"nif_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'aux_optimizer_state_dict': aux_optimizer.state_dict(),
                'loss': epoch_loss / num_batches,
            }, checkpoint_path)
            print(f"Checkpoint salvo em: {checkpoint_path}")

    # Encerra o writer do TensorBoard
    writer.close()

if __name__ == "__main__":
    main()
