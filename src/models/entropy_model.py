import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import GaussianMixtureConditional
from compressai.layers import GDN
import struct
import numpy as np

def pack_gmm_string(rv_bytes, r_min, r_max, zero_bitmap):
    zb_bytes = zero_bitmap.cpu().to(torch.uint8).numpy().tobytes()
    # h = short com sinal (2 bytes), H = short sem sinal (2 bytes)
    header = struct.pack("!hhH", r_min, r_max, len(zb_bytes))
    return header + zb_bytes + rv_bytes

def unpack_gmm_string(packed_bytes, device):
    r_min, r_max, zb_len = struct.unpack("!hhH", packed_bytes[:6])
    zb_bytes = packed_bytes[6:6+zb_len]
    rv_bytes = packed_bytes[6+zb_len:]
    zb_np = np.frombuffer(zb_bytes, dtype=np.uint8).copy()
    zero_bitmap = torch.from_numpy(zb_np).to(device).to(torch.long)
    return rv_bytes, r_min, r_max, zero_bitmap


class AdaptiveRangeGaussianMixtureConditional(GaussianMixtureConditional):
    """
    Extensão do GaussianMixtureConditional do CompressAI que implementa
    o algoritmo de PMF adaptativa de intervalo assimétrico do DLPR.
    Reduz a tabela CDF de probabilidade ao intervalo [r_min, r_max] da imagem.
    """
    @torch.no_grad()
    def _build_cdf(self, scales, means, weights, r_min, r_max):
        num_latents = scales.size(1)
        num_samples = int(r_max - r_min + 1)
        TINY = 1e-10
        device = scales.device

        scales = scales.clamp_(0.11, 256)
        means = means - r_min

        scales_ = scales.unsqueeze(-1).expand(-1, -1, num_samples)
        means_ = means.unsqueeze(-1).expand(-1, -1, num_samples)
        weights_ = weights.unsqueeze(-1).expand(-1, -1, num_samples)

        samples = (
            torch.arange(num_samples).to(device).unsqueeze(0).expand(num_latents, -1)
        )

        pmf = torch.zeros_like(samples).float()
        for k in range(self.K):
            pmf += (
                0.5
                * (
                    1
                    + torch.erf(
                        (samples + 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
                - 0.5
                * (
                    1
                    + torch.erf(
                        (samples - 0.5 - means_[k]) / ((scales_[k] + TINY) * 2**0.5)
                    )
                )
            ) * weights_[k]

        cdf_limit = 2**self.entropy_coder_precision - 1
        pmf = torch.clamp(pmf, min=1.0 / cdf_limit, max=1.0)
        pmf_scaled = torch.round(pmf * cdf_limit)
        pmf_sum = torch.sum(pmf_scaled, 1, keepdim=True).expand(-1, num_samples)

        cdf = F.pad(
            torch.cumsum(pmf_scaled * cdf_limit / pmf_sum, 1).int(),
            (1, 0),
            "constant",
            0,
        )
        pmf_quantized = torch.diff(cdf, dim=1)

        pmf_zero_count = num_samples - torch.count_nonzero(pmf_quantized, dim=1)

        _, pmf_first_stealable_indices = torch.min(
            torch.where(
                pmf_quantized > pmf_zero_count.unsqueeze(-1).expand(-1, num_samples),
                pmf_quantized,
                torch.tensor(cdf_limit + 1).int(),
            ),
            dim=1,
        )

        pmf_real_zero_indices = (pmf_quantized == 0).nonzero().transpose(0, 1)
        if pmf_real_zero_indices.numel() > 0:
            pmf_quantized[pmf_real_zero_indices[0], pmf_real_zero_indices[1]] += 1

        pmf_real_steal_indices = torch.cat(
            (
                torch.arange(num_latents).to(device).unsqueeze(-1),
                pmf_first_stealable_indices.unsqueeze(-1),
            ),
            dim=1,
        ).transpose(0, 1)
        pmf_quantized[pmf_real_steal_indices[0], pmf_real_steal_indices[1]] -= (
            pmf_zero_count
        )

        cdf = F.pad(torch.cumsum(pmf_quantized, 1).int(), (1, 0), "constant", 0)
        cdf = F.pad(cdf, (0, 1), "constant", cdf_limit + 1)

        return cdf

    def compress(self, y, scales, means, weights):
        y_quantized = torch.round(y)
        
        r_min = int(y_quantized.min().item())
        r_max = int(y_quantized.max().item())
        
        if r_max < r_min:
            r_max = r_min
        if r_max == r_min:
            r_max += 1

        zero_bitmap = torch.where(
            torch.sum(torch.abs(y_quantized), (3, 2)).squeeze(0) == 0, 0, 1
        )

        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        symbols = y_quantized[:, nonzero] - r_min
        
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), r_min, r_max
        )

        num_latents = cdf.size(0)

        rv = self.entropy_coder._encoder.encode_with_indexes(
            symbols.reshape(-1).int().tolist(),
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        return (rv, r_min, r_max, zero_bitmap), y_quantized

    def decompress(self, strings, r_min, r_max, zero_bitmap, scales, means, weights):
        nonzero = torch.nonzero(zero_bitmap).flatten().tolist()
        cdf = self._build_cdf(
            *self.reshape_entropy_parameters(scales, means, weights, nonzero), r_min, r_max
        )

        num_latents = cdf.size(0)

        values = self.entropy_coder._decoder.decode_with_indexes(
            strings,
            torch.arange(num_latents).int().tolist(),
            cdf.cpu().tolist(),
            torch.tensor(cdf.size(1)).repeat(num_latents).int().tolist(),
            torch.tensor(0).repeat(num_latents).int().tolist(),
        )

        symbols = torch.tensor(values) + r_min
        symbols = symbols.reshape(scales.size(0), -1, scales.size(2), scales.size(3))

        y_hat = torch.zeros(
            scales.size(0), zero_bitmap.size(0), scales.size(2), scales.size(3)
        )
        y_hat[:, nonzero] = symbols.float()

        return y_hat

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

        # Condicionador Gaussiano de Mistura Adaptativo
        self.gaussian_conditional = AdaptiveRangeGaussianMixtureConditional(K=K)

        # Mini-rede para previsão do mapa de importância espacial a partir de hyper_features
        self.importance_network = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

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
        Aplica Channel Residual Learning (Noise Shaping) propagando o resíduo.
        Aplica também Importance Map para alocação espacial de bits.
        """
        B, C, H, W = y.size()
        device = y.device
        
        # Preve o mapa de importância espacial a partir das hyper_features
        importance_map = 0.1 + 0.9 * self.importance_network(hyper_features)
        
        # Multiplica o latente y pelo mapa de importância
        y_importance = y * importance_map
        
        y_slices = torch.chunk(y_importance, self.num_slices, dim=1)
        y_hat_slices = []
        likelihoods_list = []

        checkerboard_mask = self._get_checkerboard_mask(y_slices[0])
        
        # Inicializa o resíduo do canal anterior
        res = torch.zeros(B, self.slice_size, H, W, device=device)

        for k in range(self.num_slices):
            # Adiciona resíduo à fatia atual
            curr_slice = y_slices[k] + res
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
            
            # Calcula o resíduo para a próxima fatia (Noise Shaping)
            res = curr_slice - y_hat_slice
            
            y_hat_slices.append(y_hat_slice)
            likelihoods_list.append(slice_likelihoods)

        y_hat_importance = torch.cat(y_hat_slices, dim=1)
        likelihoods = torch.cat(likelihoods_list, dim=1)

        # Restaura a escala original do latente dividindo pelo mapa de importância
        y_hat = y_hat_importance / importance_map

        return y_hat, likelihoods

    def compress(self, y, hyper_features):
        """
        Compacta o latente 'y' usando GMM e codificação de entropia checkerboard adaptativa.
        Aplica Channel Residual Learning (Noise Shaping) propagando o resíduo na quantização.
        Aplica também Importance Map para alocação espacial de bits.
        """
        B, C, H, W = y.size()
        device = y.device
        
        # Preve o mapa de importância espacial a partir das hyper_features
        importance_map = 0.1 + 0.9 * self.importance_network(hyper_features)
        
        # Multiplica o latente y pelo mapa de importância
        y_importance = y * importance_map
        
        y_slices = torch.chunk(y_importance, self.num_slices, dim=1)
        y_hat_slices = []
        strings_list = []
        
        # Inicializa o resíduo do canal anterior
        res = torch.zeros(B, self.slice_size, H, W, device=device)

        for k in range(self.num_slices):
            # Adiciona resíduo da fatia anterior
            curr_slice = y_slices[k] + res
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

            # Verificar se as âncoras são todas zero
            y_k_anchors_quant = torch.round(y_k_anchors)
            zero_bitmap = torch.where(
                torch.sum(torch.abs(y_k_anchors_quant), (0, 2, 3)) == 0, 0, 1
            )

            if zero_bitmap.sum() == 0:
                rv = b""
                packed_anchor = pack_gmm_string(rv, 0, 1, zero_bitmap)
                strings_list.append([packed_anchor])
                y_k_anchors_hat = torch.zeros_like(y_k_anchors)
            else:
                # Comprime as âncoras usando o método adaptativo
                anchor_res, _ = self.gaussian_conditional.compress(
                    y_k_anchors, scale_anchor_split, mu_anchor_split, weight_anchor_split
                )
                rv, r_min, r_max, zero_bitmap = anchor_res
                packed_anchor = pack_gmm_string(rv, r_min, r_max, zero_bitmap)
                strings_list.append([packed_anchor])

                # Decomprime localmente
                y_k_anchors_hat = self.gaussian_conditional.decompress(
                    rv, r_min, r_max, zero_bitmap, scale_anchor_split, mu_anchor_split, weight_anchor_split
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

            # Verificar se não-âncoras são todas zero
            y_k_non_anchors_quant = torch.round(y_k_non_anchors)
            zero_bitmap_na = torch.where(
                torch.sum(torch.abs(y_k_non_anchors_quant), (0, 2, 3)) == 0, 0, 1
            )

            if zero_bitmap_na.sum() == 0:
                rv_na = b""
                packed_non_anchor = pack_gmm_string(rv_na, 0, 1, zero_bitmap_na)
                strings_list.append([packed_non_anchor])
                y_k_non_anchors_hat = torch.zeros_like(y_k_non_anchors)
            else:
                # Comprime não-âncoras usando o método adaptativo
                non_anchor_res, _ = self.gaussian_conditional.compress(
                    y_k_non_anchors, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                )
                rv_na, r_min_na, r_max_na, zero_bitmap_na = non_anchor_res
                packed_non_anchor = pack_gmm_string(rv_na, r_min_na, r_max_na, zero_bitmap_na)
                strings_list.append([packed_non_anchor])

                # Decomprime localmente
                y_k_non_anchors_hat = self.gaussian_conditional.decompress(
                    rv_na, r_min_na, r_max_na, zero_bitmap_na, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                ).to(device)

            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            
            # Atualiza o resíduo para a próxima fatia (Noise Shaping)
            res = curr_slice - y_hat_slice
            
            y_hat_slices.append(y_hat_slice)

        return strings_list

    def decompress(self, strings_list, hyper_features, H, W):
        """
        Decomprime o latente 'y' usando o modelo GMM adaptativo.
        Aplica também a reversão do Importance Map.
        """
        B = hyper_features.size(0)
        device = hyper_features.device
        
        # Preve o mapa de importância espacial a partir das hyper_features
        importance_map = 0.1 + 0.9 * self.importance_network(hyper_features)
        
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
            rv, r_min, r_max, zero_bitmap = unpack_gmm_string(packed_anchor, device)
            
            if zero_bitmap.sum() == 0:
                y_k_anchors_hat = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            else:
                y_k_anchors_hat = self.gaussian_conditional.decompress(
                    rv, r_min, r_max, zero_bitmap, scale_anchor_split, mu_anchor_split, weight_anchor_split
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
            rv_na, r_min_na, r_max_na, zero_bitmap_na = unpack_gmm_string(packed_non_anchor, device)
            
            if zero_bitmap_na.sum() == 0:
                y_k_non_anchors_hat = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            else:
                y_k_non_anchors_hat = self.gaussian_conditional.decompress(
                    rv_na, r_min_na, r_max_na, zero_bitmap_na, scale_non_anchor_split, mu_non_anchor_split, weight_non_anchor_split
                ).to(device)

            # Mescla e acumula
            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            y_hat_slices.append(y_hat_slice)

        y_hat_importance = torch.cat(y_hat_slices, dim=1)
        
        # Restaura a escala original do latente dividindo pelo mapa de importância
        y_hat = y_hat_importance / importance_map
        return y_hat
