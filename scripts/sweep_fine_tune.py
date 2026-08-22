import sys
import os
sys.path.append(os.getcwd())
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.tensorboard import SummaryWriter
from src.models.nif_codec import NIFCodec
from pytorch_msssim import MS_SSIM
import lpips

class NIFDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.main(x)

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))

class NormalizedNIFLoss(nn.Module):
    def __init__(self, w_mse=0.40, w_ssim=0.40, w_lpips=0.20, s_mse=600.0, s_ssim=60.0, s_lpips=14.0, use_lpips=True, device='cpu'):
        super().__init__()
        self.w_mse = w_mse
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.s_mse = s_mse
        self.s_ssim = s_ssim
        self.s_lpips = s_lpips
        self.use_lpips = use_lpips
        
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=False, channel=3)
        if use_lpips and w_lpips > 0:
            self.lpips_metric = lpips.LPIPS(net='alex', spatial=True).to(device)
            for p in self.lpips_metric.parameters():
                p.requires_grad = False
        else:
            self.lpips_metric = None

    def forward(self, x, x_hat, likelihoods, sfm, q, lambda_min=0.15, lambda_max=15.0):
        B, C, H, W = x.size()
        num_pixels_per_sample = H * W
        
        # 1. Rate Loss
        log_lik_y = torch.log(likelihoods["y"]).flatten(start_dim=1).sum(dim=1)
        log_lik_z = torch.log(likelihoods["z"]).flatten(start_dim=1).sum(dim=1)
        bpp_y = log_lik_y / (-num_pixels_per_sample * torch.log(torch.tensor(2.0, device=x.device)))
        bpp_z = log_lik_z / (-num_pixels_per_sample * torch.log(torch.tensor(2.0, device=x.device)))
        rate_loss = bpp_y + bpp_z # [B]
        
        # 2. Raw Distortion terms
        mse_raw = nn.functional.mse_loss(x, x_hat, reduction='none').flatten(start_dim=1).mean(dim=1) # [B]
        ssim_raw = 1.0 - self.ms_ssim(x, x_hat) # [B]
        
        if self.use_lpips and self.lpips_metric is not None and self.w_lpips > 0:
            lpips_map = self.lpips_metric(x, x_hat)
            if lpips_map.shape[-2:] != sfm.shape[-2:]:
                sfm_resized = nn.functional.interpolate(sfm, size=lpips_map.shape[-2:], mode='bilinear', align_corners=False)
            else:
                sfm_resized = sfm
            lpips_raw = (lpips_map * (1.0 - sfm_resized)).flatten(start_dim=1).mean(dim=1) # [B]
        else:
            lpips_raw = torch.zeros(B, device=x.device)
            
        # 3. Scaled Distortion Terms (~1.0 magnitude each)
        mse_scaled = self.s_mse * mse_raw
        ssim_scaled = self.s_ssim * ssim_raw
        lpips_scaled = self.s_lpips * lpips_raw
        
        # 4. Normalized Composite Distortion
        d_tilde = self.w_mse * mse_scaled + self.w_ssim * ssim_scaled + self.w_lpips * lpips_scaled
        
        # 5. Lambda(q)
        q_flat = q.squeeze(1)
        lambda_val = lambda_min * ((lambda_max / lambda_min) ** q_flat)
        
        total_loss = rate_loss + lambda_val * d_tilde
        
        return (
            total_loss.mean(),
            rate_loss.mean(),
            mse_raw.mean(),
            ssim_raw.mean(),
            lpips_raw.mean(),
            mse_scaled.mean(),
            ssim_scaled.mean(),
            lpips_scaled.mean(),
            d_tilde.mean()
        )

class ImageFolderCustom(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        for root, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    self.image_files.append(os.path.join(root, file))
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

def evaluate_kodak_validation(model, val_dataset_dir, device):
    images = [os.path.join(val_dataset_dir, f"kodim{i:02d}.png") for i in range(1, 25)]
    valid_images = [img for img in images if os.path.exists(img)]
    if not valid_images:
        return {}
        
    qualities = [0.1, 0.3, 0.5, 0.7, 0.9]
    metrics = {q: {"bpp": [], "psnr": []} for q in qualities}
    
    model.eval()
    with torch.no_grad():
        for img_path in valid_images:
            img = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            w, h = img.shape[-2:]
            num_pixels = w * h
            
            for q in qualities:
                q_t = torch.tensor([[q]], device=device)
                out = model(img, q_t)
                x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)
                lik = out["likelihoods"]
                
                bpp_y = torch.log(lik["y"]).flatten(start_dim=1).sum(dim=1) / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
                bpp_z = torch.log(lik["z"]).flatten(start_dim=1).sum(dim=1) / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
                bpp = (bpp_y + bpp_z).item()
                psnr = compute_psnr(img, x_hat).item()
                
                metrics[q]["bpp"].append(bpp)
                metrics[q]["psnr"].append(psnr)
                
    summary = {}
    for q in qualities:
        summary[q] = {
            "bpp": float(np.mean(metrics[q]["bpp"])),
            "psnr": float(np.mean(metrics[q]["psnr"]))
        }
    return summary

def run_fine_tune_sweep():
    parser = argparse.ArgumentParser(description="Sweep de Calibração de Fine-Tuning NIF (v4 -> v5)")
    parser.add_argument("--config_name", type=str, required=True, help="Nome da configuração (ex: Config_A)")
    parser.add_argument("--w_mse", type=float, required=True, help="Peso MSE")
    parser.add_argument("--w_ssim", type=float, required=True, help="Peso MS-SSIM")
    parser.add_argument("--w_lpips", type=float, required=True, help="Peso LPIPS")
    parser.add_argument("--lambda_min", type=float, default=0.15, help="Lambda Min")
    parser.add_argument("--lambda_max", type=float, default=15.0, help="Lambda Max")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas do sweep")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="DIV2K_train_HR/")
    parser.add_argument("--val_dataset", type=str, default="kodak24/")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints_v4_production/nif_epoch_300.pth")
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs")
    writer = SummaryWriter(log_dir)
    
    print("\n" + "="*80)
    print(f" INICIANDO SWEEP: {args.config_name}")
    print(f" Pesos: w_mse={args.w_mse:.2f}, w_ssim={args.w_ssim:.2f}, w_lpips={args.w_lpips:.2f}")
    print(f" Lambda Range: [{args.lambda_min:.3f}, {args.lambda_max:.3f}]")
    print(f" Checkpoint Base: {args.base_checkpoint}")
    print(f" Diretório de Saída: {args.save_dir}")
    print("="*80 + "\n")
    
    # Modelo
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8, use_sigmoid=True).to(device)
    ckpt = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    model.train()
    
    discriminator = NIFDiscriminator(in_channels=3).to(device)
    if 'discriminator_state_dict' in ckpt:
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
    discriminator.train()
    
    # Parâmetros e Otimizadores
    cond_names = ['cond_enc1', 'cond_enc2', 'cond_enc3', 'cond_enc4', 'cond_dec1', 'cond_dec2', 'cond_dec3', 'cond_dec4', 'latent_scaler']
    cond_params = []
    main_params = []
    aux_params = []
    for name, p in model.named_parameters():
        if name.endswith(".quantiles"):
            aux_params.append(p)
        elif any(cname in name for cname in cond_names):
            cond_params.append(p)
        else:
            main_params.append(p)
            
    optimizer = optim.Adam([
        {'params': main_params, 'lr': args.lr},
        {'params': cond_params, 'lr': args.lr * 5.0}
    ])
    aux_optimizer = optim.Adam(aux_params, lr=1e-3)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    
    criterion = NormalizedNIFLoss(
        w_mse=args.w_mse, w_ssim=args.w_ssim, w_lpips=args.w_lpips,
        s_mse=600.0, s_ssim=60.0, s_lpips=14.0,
        use_lpips=(args.w_lpips > 0), device=device
    )
    
    # Dataset
    train_transforms = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolderCustom(args.dataset, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Loop de Treino do Sweep
    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()
        
        ep_loss, ep_rate, ep_mse, ep_ssim, ep_lpips = 0, 0, 0, 0, 0
        ep_mse_s, ep_ssim_s, ep_lpips_s, ep_d_tilde = 0, 0, 0, 0
        
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            B = x.size(0)
            q = torch.rand(B, 1, device=device) * 0.9 + 0.1
            
            out = model(x, q)
            x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)
            likelihoods = out["likelihoods"]
            sfm = out["sfm"]
            
            # Discriminador (GAN)
            if args.w_lpips > 0:
                pred_real = discriminator(x)
                pred_fake = discriminator(x_hat.detach())
                sfm_d = nn.functional.interpolate(sfm, size=pred_fake.shape[-2:], mode='bilinear', align_corners=False)
                loss_d_real = nn.functional.mse_loss(pred_real, torch.ones_like(pred_real), reduction='none')
                loss_d_fake = nn.functional.mse_loss(pred_fake, torch.zeros_like(pred_fake), reduction='none')
                loss_d = 0.5 * ((loss_d_real * (1.0 - sfm_d)).mean() + (loss_d_fake * (1.0 - sfm_d)).mean())
                
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
                
                # Gerador Adv
                pred_fake_g = discriminator(x_hat)
                loss_g_adv_map = nn.functional.mse_loss(pred_fake_g, torch.ones_like(pred_fake_g), reduction='none')
                loss_g_adv = (loss_g_adv_map * (1.0 - sfm_d)).mean()
                adv_w = 0.05 * (1.0 - q.mean().item())
            else:
                loss_g_adv = torch.tensor(0.0, device=device)
                adv_w = 0.0
                
            (
                loss, rate, mse_r, ssim_r, lpips_r,
                mse_s, ssim_s, lpips_s, d_tilde
            ) = criterion(x, x_hat, likelihoods, sfm, q, lambda_min=args.lambda_min, lambda_max=args.lambda_max)
            
            total_g_loss = loss + adv_w * loss_g_adv
            
            optimizer.zero_grad()
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params + cond_params, max_norm=1.0)
            optimizer.step()
            
            aux_optimizer.zero_grad()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            
            ep_loss += total_g_loss.item()
            ep_rate += rate.item()
            ep_mse += mse_r.item()
            ep_ssim += ssim_r.item()
            ep_lpips += lpips_r.item()
            ep_mse_s += mse_s.item()
            ep_ssim_s += ssim_s.item()
            ep_lpips_s += lpips_s.item()
            ep_d_tilde += d_tilde.item()
            
        num_batches = len(train_loader)
        avg_loss = ep_loss / num_batches
        avg_rate = ep_rate / num_batches
        avg_mse = ep_mse / num_batches
        avg_ssim = ep_ssim / num_batches
        avg_lpips = ep_lpips / num_batches
        avg_mse_s = ep_mse_s / num_batches
        avg_ssim_s = ep_ssim_s / num_batches
        avg_lpips_s = ep_lpips_s / num_batches
        
        # Log TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Bpp", avg_rate, epoch)
        writer.add_scalar("Train/MSE_Raw", avg_mse, epoch)
        writer.add_scalar("Scaled_Terms/MSE", avg_mse_s, epoch)
        writer.add_scalar("Scaled_Terms/SSIM", avg_ssim_s, epoch)
        writer.add_scalar("Scaled_Terms/LPIPS", avg_lpips_s, epoch)
        
        print(f"Época [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:.4f} | Bpp: {avg_rate:.4f} | MSE_Raw: {avg_mse:.6f} | MSE_Scaled: {avg_mse_s:.3f} | SSIM_Scaled: {avg_ssim_s:.3f} | LPIPS_Scaled: {avg_lpips_s:.3f}")
        
        # Validação Kodak24 periódica (a cada 3 épocas e na última)
        if epoch % 3 == 0 or epoch == args.epochs:
            val_rd = evaluate_kodak_validation(model, args.val_dataset, device)
            print(f"  --> Validação Kodak24 (Época {epoch:02d}):")
            for q_eval in [0.1, 0.5, 0.9]:
                b = val_rd[q_eval]['bpp']
                p = val_rd[q_eval]['psnr']
                writer.add_scalar(f"Val_RD/Bpp_q_{q_eval}", b, epoch)
                writer.add_scalar(f"Val_RD/PSNR_q_{q_eval}", p, epoch)
                print(f"      q={q_eval:.1f} => Bitrate: {b:.4f} bpp | PSNR: {p:.2f} dB")
                
            # --- BPP WATCHDOG EXTENDIDO ---
            bpp_low = val_rd[0.1]['bpp']
            bpp_mid = val_rd[0.5]['bpp']
            bpp_high = val_rd[0.9]['bpp']
            
            if bpp_low > 2.0 or bpp_mid > 2.0 or bpp_high > 3.0:
                print(f"\n[ALERTA BPP WATCHDOG] Bitrate explodiu fora do teto de segurança! (q=0.1: {bpp_low:.2f}, q=0.5: {bpp_mid:.2f}, q=0.9: {bpp_high:.2f})")
                print(f"Interrompendo antecipadamente {args.config_name} para proteger tempo de GPU.\n")
                break
                
    # Salva checkpoint final do sweep
    final_ckpt_path = os.path.join(args.save_dir, "nif_sweep_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args)
    }, final_ckpt_path)
    print(f"\nCheckpoint do sweep salvo com sucesso em: {final_ckpt_path}\n")

if __name__ == "__main__":
    run_fine_tune_sweep()
