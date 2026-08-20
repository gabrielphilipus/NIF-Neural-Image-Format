import torch
import torch.nn as nn
from compressai.entropy_models import GaussianMixtureConditional
from compressai.layers import GDN
import struct
import numpy as np

def pack_gmm_string(rv_bytes, abs_max, zero_bitmap):
    zb_bytes = zero_bitmap.cpu().to(torch.uint8).numpy().tobytes()
    header = struct.pack("!iH", abs_max, len(zb_bytes))
    return header + zb_bytes + rv_bytes

def unpack_gmm_string(packed_bytes, device):
    abs_max, zb_len = struct.unpack("!iH", packed_bytes[:6])
    zb_bytes = packed_bytes[6:6+zb_len]
    rv_bytes = packed_bytes[6+zb_len:]
    zb_np = np.frombuffer(zb_bytes, dtype=np.uint8).copy()
    zero_bitmap = torch.from_numpy(zb_np).to(device).to(torch.long)
    return rv_bytes, abs_max, zero_bitmap

class MaskedConv2d(nn.Conv2d):
    """
    Convolution 2D com máscara para implementar o contexto em xadrez (Checkerboard).
    Garante que o peso central seja sempre zero, de forma que a predição de um pixel
    dependa apenas de seus vizinhos de cor oposta (tabuleiro de xadrez).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Registra a máscara como buffer para que seja salva com o modelo
        kh, kw = self.weight.shape[-2:]
        mask = torch.ones(self.weight.shape)
        # Zera o peso central
        mask[:, :, kh // 2, kw // 2] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return nn.functional.conv2d(
            x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def split_checkerboard(tensor):
    """
    Divide a grade espacial H x W em âncoras (pixels pretos) e não-âncoras (pixels brancos).
    Retorna dois tensores de dimensão [B, C, H, W // 2]. Assume que W é par.
    """
    B, C, H, W = tensor.shape
    even_rows = tensor[:, :, 0::2, :]
    odd_rows = tensor[:, :, 1::2, :]
    
    anchors_even = even_rows[:, :, :, 0::2]
    non_anchors_even = even_rows[:, :, :, 1::2]
    
    anchors_odd = odd_rows[:, :, :, 1::2]
    non_anchors_odd = odd_rows[:, :, :, 0::2]
    
    anchors = torch.cat([anchors_even, anchors_odd], dim=2)
    non_anchors = torch.cat([non_anchors_even, non_anchors_odd], dim=2)
    return anchors, non_anchors


def merge_checkerboard(anchors, non_anchors, H, W):
    """
    Mescla as âncoras e não-âncoras reconstruídas de volta na grade espacial H x W.
    """
    B, C, _, _ = anchors.shape
    device = anchors.device
    dtype = anchors.dtype
    
    out = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    
    anchors_even, anchors_odd = anchors.chunk(2, dim=2)
    non_anchors_even, non_anchors_odd = non_anchors.chunk(2, dim=2)
    
    out[:, :, 0::2, 0::2] = anchors_even
    out[:, :, 0::2, 1::2] = non_anchors_even
    out[:, :, 1::2, 1::2] = anchors_odd
    out[:, :, 1::2, 0::2] = non_anchors_odd
    
    return out


class ChannelCheckerboardEntropyModel(nn.Module):
    """
    Modelo de Entropia Híbrido: Channel-wise + Spatial Checkerboard baseada em GMM.
    Divide o latente 'y' em fatias de canais (slices). Para cada fatia:
    1. Prediz as âncoras (pixels pretos no tabuleiro) usando a hyperprior e as fatias anteriores.
    2. Prediz as não-âncoras (pixels brancos) usando a hyperprior, fatias anteriores e as âncoras decodificadas.
    Utiliza GaussianMixtureConditional para modelar a distribuição latente com maior flexibilidade.
    """
    def __init__(self, in_channels=192, num_slices=8, latent_dim=192, K=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_slices = num_slices
        self.latent_dim = latent_dim
        self.K = K
        
        # Tamanho de cada fatia de canal
        assert in_channels % num_slices == 0, f"Canais {in_channels} deve ser divisível por num_slices {num_slices}"
        self.slice_size = in_channels // num_slices

        # Condicionador Gaussiano de Mistura do CompressAI
        self.gaussian_conditional = GaussianMixtureConditional(K=K)

        # Módulos de fusão de contexto para cada fatia
        self.channel_context_networks = nn.ModuleList()
        self.spatial_context_networks = nn.ModuleList()
        self.entropy_parameter_networks = nn.ModuleList()

        for k in range(num_slices):
            # Canais das fatias anteriores já decodificadas
            prev_channels = k * self.slice_size
            
            # Rede de contexto de canal (Channel Context)
            if prev_channels > 0:
                self.channel_context_networks.append(
                    nn.Sequential(
                        nn.Conv2d(prev_channels, self.slice_size * 2, kernel_size=3, padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(self.slice_size * 2, self.slice_size * 2, kernel_size=3, padding=1)
                    )
                )
            else:
                self.channel_context_networks.append(None)

            # Rede de contexto espacial baseada em xadrez (Spatial Checkerboard)
            self.spatial_context_networks.append(
                MaskedConv2d(self.slice_size, self.slice_size * 2, kernel_size=3, padding=1)
            )

            # Rede de parâmetros de entropia que combina Hyperprior + Channel + Spatial
            # e gera média (mu), escala (sigma) e pesos (weights) para o GMM
            input_dim = latent_dim  # Tamanho da hyperprior vinda do hyper-decoder
            if prev_channels > 0:
                input_dim += self.slice_size * 2  # Adiciona canal-contexto
            input_dim += self.slice_size * 2  # Adiciona espacial-contexto (para a predição final)

            self.entropy_parameter_networks.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, self.slice_size * self.K * 4, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(self.slice_size * self.K * 4, self.slice_size * self.K * 3, kernel_size=1)
                )
            )

    def _get_checkerboard_mask(self, x):
        """
        Gera uma máscara de xadrez onde 1 representa âncoras (pixels pretos) e 0 representa não-âncoras (brancos).
        """
        B, C, H, W = x.size()
        device = x.device
        coords_h = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords_w = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, coords_h.size(3))
        mask = ((coords_h + coords_w) % 2 == 0).float()
        return mask

    def forward(self, y, hyper_features):
        """
        Durante o treinamento, calculamos a probabilidade usando o modelo GMM.
        """
        B, C, H, W = y.size()
        device = y.device
        
        y_slices = torch.chunk(y, self.num_slices, dim=1)
        y_hat_slices = []
        likelihoods_list = []

        checkerboard_mask = self._get_checkerboard_mask(y_slices[0])

        for k in range(self.num_slices):
            curr_slice = y_slices[k]
            prev_slices = y_hat_slices
            
            # 1. Obter contexto de canal
            if k > 0:
                prev_latents = torch.cat(prev_slices, dim=1)
                channel_ctx = self.channel_context_networks[k](prev_latents)
            else:
                channel_ctx = None

            # 2. Primeira Fase (Âncoras)
            spatial_ctx_anchors = torch.zeros(B, self.slice_size * 2, H, W, device=device)
            inputs_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_anchor.append(channel_ctx)
            inputs_anchor.append(spatial_ctx_anchors)
            
            params_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_anchor, dim=1))
            mu_anchor, scale_anchor, weight_anchor_logits = params_anchor.chunk(3, dim=1)
            
            # Ativação softplus para estabilidade das escalas do GMM
            scale_anchor = torch.nn.functional.softplus(scale_anchor)
            
            # Softmax nos pesos da mistura GMM
            B_a, CK_a, H_a, W_a = weight_anchor_logits.shape
            weight_anchor = weight_anchor_logits.reshape(B_a, self.K, self.slice_size, H_a, W_a)
            weight_anchor = torch.softmax(weight_anchor, dim=1)
            weight_anchor = weight_anchor.reshape(B_a, CK_a, H_a, W_a)

            # Calcula a média ponderada do GMM (esperança matemática) para centralização
            mu_anchor_gmm = (mu_anchor.reshape(B, self.K, self.slice_size, H, W) * 
                             weight_anchor.reshape(B, self.K, self.slice_size, H, W)).sum(dim=1)

            # Quantiza âncoras centrando pela média GMM para consistência espacial
            y_hat_anchor = self.gaussian_conditional.quantize(
                curr_slice - mu_anchor_gmm, 
                "noise" if self.training else "dequantize", 
                means=None
            ) + mu_anchor_gmm
            y_anchor_only = y_hat_anchor * checkerboard_mask
            
            # 3. Segunda Fase (Não-Âncoras)
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_anchor_only)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor, weight_non_anchor_logits = params_non_anchor.chunk(3, dim=1)
            
            scale_non_anchor = torch.nn.functional.softplus(scale_non_anchor)
            
            # Softmax nos pesos da mistura GMM
            B_na, CK_na, H_na, W_na = weight_non_anchor_logits.shape
            weight_non_anchor = weight_non_anchor_logits.reshape(B_na, self.K, self.slice_size, H_na, W_na)
            weight_non_anchor = torch.softmax(weight_non_anchor, dim=1)
            weight_non_anchor = weight_non_anchor.reshape(B_na, CK_na, H_na, W_na)

            # Mescla parâmetros
            mu = mu_anchor * checkerboard_mask + mu_non_anchor * (1.0 - checkerboard_mask)
            scale = scale_anchor * checkerboard_mask + scale_non_anchor * (1.0 - checkerboard_mask)
            weight = weight_anchor * checkerboard_mask + weight_non_anchor * (1.0 - checkerboard_mask)

            # O GMM calcula a verossimilhança com mu, scale e weight
            y_hat_slice, slice_likelihoods = self.gaussian_conditional(
                curr_slice, 
                scale, 
                means=mu, 
                weights=weight
            )
            
            y_hat_slices.append(y_hat_slice)
            likelihoods_list.append(slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        likelihoods = torch.cat(likelihoods_list, dim=1)

        return y_hat, likelihoods

    def compress(self, y, hyper_features):
        """
        Compacta o latente 'y' usando GMM e codificação de entropia checkerboard.
        """
        B, C, H, W = y.size()
        device = y.device
        
        y_slices = torch.chunk(y, self.num_slices, dim=1)
        y_hat_slices = []
        strings_list = []

        for k in range(self.num_slices):
            curr_slice = y_slices[k]
            prev_slices = y_hat_slices
            
            if k > 0:
                prev_latents = torch.cat(prev_slices, dim=1)
                channel_ctx = self.channel_context_networks[k](prev_latents)
            else:
                channel_ctx = None

            # 2. Primeira Fase (Âncoras)
            spatial_ctx_anchors = torch.zeros(B, self.slice_size * 2, H, W, device=device)
            inputs_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_anchor.append(channel_ctx)
            inputs_anchor.append(spatial_ctx_anchors)
            
            params_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_anchor, dim=1))
            mu_anchor, scale_anchor, weight_anchor_logits = params_anchor.chunk(3, dim=1)
            
            scale_anchor = torch.nn.functional.softplus(scale_anchor)
            
            B_a, CK_a, H_a, W_a = weight_anchor_logits.shape
            weight_anchor = weight_anchor_logits.reshape(B_a, self.K, self.slice_size, H_a, W_a)
            weight_anchor = torch.softmax(weight_anchor, dim=1)
            weight_anchor = weight_anchor.reshape(B_a, CK_a, H_a, W_a)

            y_k_anchors, y_k_non_anchors = split_checkerboard(curr_slice)
            mu_anchor_split, _ = split_checkerboard(mu_anchor)
            scale_anchor_split, _ = split_checkerboard(scale_anchor)
            weight_anchor_split, _ = split_checkerboard(weight_anchor)

            # Pre-calcular abs_max
            abs_max = (
                max(torch.abs(y_k_anchors.max()).int().item(), torch.abs(y_k_anchors.min()).int().item()) + 1
            )
            abs_max = 1 if abs_max < 1 else abs_max

            # Verificar se as âncoras são todas zero
            y_k_anchors_quant = torch.round(y_k_anchors)
            zero_bitmap = torch.where(
                torch.sum(torch.abs(y_k_anchors_quant), (0, 2, 3)) == 0, 0, 1
            )

            if zero_bitmap.sum() == 0:
                rv = b""
                packed_anchor = pack_gmm_string(rv, abs_max, zero_bitmap)
                strings_list.append([packed_anchor])
                y_k_anchors_hat = torch.zeros_like(y_k_anchors)
            else:
                # Comprime as âncoras usando GMM
                anchor_res, _ = self.gaussian_conditional.compress(
                    y_k_anchors, scale_anchor_split, mu_anchor_split, weight_anchor_split
                )
                rv, abs_max, zero_bitmap = anchor_res
                packed_anchor = pack_gmm_string(rv, abs_max, zero_bitmap)
                strings_list.append([packed_anchor])

                # Decomprime as âncoras localmente para o contexto espacial
                y_k_anchors_hat = self.gaussian_conditional.decompress(
                    rv, abs_max, zero_bitmap, scale_anchor_split, mu_anchor_split, weight_anchor_split
                ).to(device)

            zeros_non_anchors = torch.zeros_like(y_k_non_anchors)
            y_k_anchor_only = merge_checkerboard(y_k_anchors_hat, zeros_non_anchors, H, W)

            # 3. Segunda Fase (Não-Âncoras)
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_k_anchor_only)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor, weight_non_anchor_logits = params_non_anchor.chunk(3, dim=1)
            
            scale_non_anchor = torch.nn.functional.softplus(scale_non_anchor)
            
            B_na, CK_na, H_na, W_na = weight_non_anchor_logits.shape
            weight_non_anchor = weight_non_anchor_logits.reshape(B_na, self.K, self.slice_size, H_na, W_na)
            weight_non_anchor = torch.softmax(weight_non_anchor, dim=1)
            weight_non_anchor = weight_non_anchor.reshape(B_na, CK_na, H_na, W_na)

            _, mu_non_anchor_split = split_checkerboard(mu_non_anchor)
            _, scale_non_anchor_split = split_checkerboard(scale_non_anchor)
            _, weight_non_anchor_split = split_checkerboard(weight_non_anchor)

            # Pre-calcular abs_max_na
            abs_max_na = (
                max(torch.abs(y_k_non_anchors.max()).int().item(), torch.abs(y_k_non_anchors.min()).int().item()) + 1
            )
            abs_max_na = 1 if abs_max_na < 1 else abs_max_na

            # Verificar se não-âncoras são todas zero
            y_k_non_anchors_quant = torch.round(y_k_non_anchors)
            zero_bitmap_na = torch.where(
                torch.sum(torch.abs(y_k_non_anchors_quant), (0, 2, 3)) == 0, 0, 1
            )

            if zero_bitmap_na.sum() == 0:
                rv_na = b""
                packed_non_anchor = pack_gmm_string(rv_na, abs_max_na, zero_bitmap_na)
                strings_list.append([packed_non_anchor])
                y_k_non_anchors_hat = torch.zeros_like(y_k_non_anchors)
            else:
                # Comprime não-âncoras usando GMM
                non_anchor_res, _ = self.gaussian_conditional.compress(
                    y_k_non_anchors, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                )
                rv_na, abs_max_na, zero_bitmap_na = non_anchor_res
                packed_non_anchor = pack_gmm_string(rv_na, abs_max_na, zero_bitmap_na)
                strings_list.append([packed_non_anchor])

                # Decomprime não-âncoras localmente
                y_k_non_anchors_hat = self.gaussian_conditional.decompress(
                    rv_na, abs_max_na, zero_bitmap_na, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                ).to(device)

            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            y_hat_slices.append(y_hat_slice)

        return strings_list

    def decompress(self, strings_list, hyper_features, H, W):
        """
        Decomprime o latente 'y' usando o modelo GMM.
        """
        B = hyper_features.size(0)
        device = hyper_features.device
        
        y_hat_slices = []
        string_idx = 0

        for k in range(self.num_slices):
            prev_slices = y_hat_slices
            
            if k > 0:
                prev_latents = torch.cat(prev_slices, dim=1)
                channel_ctx = self.channel_context_networks[k](prev_latents)
            else:
                channel_ctx = None

            # 2. Primeira Fase (Âncoras)
            spatial_ctx_anchors = torch.zeros(B, self.slice_size * 2, H, W, device=device)
            inputs_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_anchor.append(channel_ctx)
            inputs_anchor.append(spatial_ctx_anchors)
            
            params_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_anchor, dim=1))
            mu_anchor, scale_anchor, weight_anchor_logits = params_anchor.chunk(3, dim=1)
            
            scale_anchor = torch.nn.functional.softplus(scale_anchor)
            
            B_a, CK_a, H_a, W_a = weight_anchor_logits.shape
            weight_anchor = weight_anchor_logits.reshape(B_a, self.K, self.slice_size, H_a, W_a)
            weight_anchor = torch.softmax(weight_anchor, dim=1)
            weight_anchor = weight_anchor.reshape(B_a, CK_a, H_a, W_a)

            mu_anchor_split, _ = split_checkerboard(mu_anchor)
            scale_anchor_split, _ = split_checkerboard(scale_anchor)
            weight_anchor_split, _ = split_checkerboard(weight_anchor)

            # Decomprime as âncoras do GMM
            packed_anchor = strings_list[string_idx][0]
            string_idx += 1
            rv, abs_max, zero_bitmap = unpack_gmm_string(packed_anchor, device)
            
            if zero_bitmap.sum() == 0:
                y_k_anchors_hat = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            else:
                y_k_anchors_hat = self.gaussian_conditional.decompress(
                    rv, abs_max, zero_bitmap, scale_anchor_split, mu_anchor_split, weight_anchor_split
                ).to(device)
            
            zeros_non_anchors = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            y_k_anchor_only = merge_checkerboard(y_k_anchors_hat, zeros_non_anchors, H, W)

            # 3. Segunda Fase (Não-Âncoras)
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_k_anchor_only)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor, weight_non_anchor_logits = params_non_anchor.chunk(3, dim=1)
            
            scale_non_anchor = torch.nn.functional.softplus(scale_non_anchor)
            
            B_na, CK_na, H_na, W_na = weight_non_anchor_logits.shape
            weight_non_anchor = weight_non_anchor_logits.reshape(B_na, self.K, self.slice_size, H_na, W_na)
            weight_non_anchor = torch.softmax(weight_non_anchor, dim=1)
            weight_non_anchor = weight_non_anchor.reshape(B_na, CK_na, H_na, W_na)

            _, mu_non_anchor_split = split_checkerboard(mu_non_anchor)
            _, scale_non_anchor_split = split_checkerboard(scale_non_anchor)
            _, weight_non_anchor_split = split_checkerboard(weight_non_anchor)

            # Decomprime as não-âncoras do GMM
            packed_non_anchor = strings_list[string_idx][0]
            string_idx += 1
            rv_na, abs_max_na, zero_bitmap_na = unpack_gmm_string(packed_non_anchor, device)
            
            if zero_bitmap_na.sum() == 0:
                y_k_non_anchors_hat = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            else:
                y_k_non_anchors_hat = self.gaussian_conditional.decompress(
                    rv_na, abs_max_na, zero_bitmap_na, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                ).to(device)

            # Mescla e acumula
            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return y_hat
