import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN

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
    Modelo de Entropia Híbrido: Channel-wise + Spatial Checkerboard.
    Divide o latente 'y' em fatias de canais (slices). Para cada fatia:
    1. Prediz as âncoras (pixels pretos no tabuleiro) usando a hyperprior e as fatias anteriores.
    2. Prediz as não-âncoras (pixels brancos) usando a hyperprior, fatias anteriores e as âncoras decodificadas.
    """
    def __init__(self, in_channels=192, num_slices=8, latent_dim=192):
        super().__init__()
        self.in_channels = in_channels
        self.num_slices = num_slices
        self.latent_dim = latent_dim
        
        # Tamanho de cada fatia de canal
        assert in_channels % num_slices == 0, f"Canais {in_channels} deve ser divisível por num_slices {num_slices}"
        self.slice_size = in_channels // num_slices

        # Condicionador Gaussiano do CompressAI para modelar a distribuição latente
        self.gaussian_conditional = GaussianConditional(None)

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
            # e gera média (mu) e escala (sigma) para o GaussianConditional
            input_dim = latent_dim  # Tamanho da hyperprior vinda do hyper-decoder
            if prev_channels > 0:
                input_dim += self.slice_size * 2  # Adiciona canal-contexto
            input_dim += self.slice_size * 2  # Adiciona espacial-contexto (para a predição final)

            self.entropy_parameter_networks.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, self.slice_size * 4, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(self.slice_size * 4, self.slice_size * 2, kernel_size=1)
                )
            )

    def _get_checkerboard_mask(self, x):
        """
        Gera uma máscara de xadrez onde 1 representa âncoras (pixels pretos) e 0 representa não-âncoras (brancos).
        """
        B, C, H, W = x.size()
        device = x.device
        # Cria uma grade de coordenadas
        coords_h = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        coords_w = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, coords_h.size(3))
        mask = ((coords_h + coords_w) % 2 == 0).float()
        # Expande para a quantidade de canais
        return mask.expand(B, C, H, W)

    def forward(self, y, hyper_features):
        """
        Durante o treinamento, calculamos a probabilidade de forma paralelizada usando máscaras.
        """
        B, C, H, W = y.size()
        device = y.device
        
        # Lista para armazenar as fatias decodificadas/quantizadas e suas probabilidades
        y_slices = torch.chunk(y, self.num_slices, dim=1)
        y_hat_slices = []
        likelihoods_list = []

        # Gerar máscara do xadrez
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

            # 2. Primeira Fase (Âncoras): Predizer a distribuição dos pixels pretos usando apenas hyperprior + canal
            # Para manter o fluxo estruturado, criamos uma representação temporária onde as não-âncoras são mascaradas
            # E usamos a rede de parâmetros sem o contexto espacial (ou com contexto espacial zerado para as âncoras)
            
            # Para predição de âncoras: contexto espacial é zero
            spatial_ctx_anchors = torch.zeros(B, self.slice_size * 2, H, W, device=device)
            
            # Combina os inputs das âncoras
            inputs_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_anchor.append(channel_ctx)
            inputs_anchor.append(spatial_ctx_anchors)
            
            params_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_anchor, dim=1))
            mu_anchor, scale_anchor = params_anchor.chunk(2, dim=1)

            # Quantiza e reconstrói âncoras
            # Durante o treinamento, a quantização usa ruído uniforme ou round dependendo do CompressAI
            y_hat_anchor = self.gaussian_conditional.quantize(curr_slice, "noise" if self.training else "dequantize", means=mu_anchor)
            
            # Cria a representação com apenas as âncoras preenchidas e não-âncoras zeradas para o contexto espacial
            y_anchor_only = y_hat_anchor * checkerboard_mask
            
            # 3. Segunda Fase (Não-Âncoras): Computar contexto espacial das âncoras decodificadas
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_anchor_only)
            
            # Combina os inputs das não-âncoras (agora incluindo o contexto espacial real das âncoras vizinhas)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor = params_non_anchor.chunk(2, dim=1)

            # Combina as médias e escalas mesclando âncoras e não-âncoras
            mu = mu_anchor * checkerboard_mask + mu_non_anchor * (1.0 - checkerboard_mask)
            scale = scale_anchor * checkerboard_mask + scale_non_anchor * (1.0 - checkerboard_mask)

            # Quantização final da fatia inteira
            y_hat_slice, slice_likelihoods = self.gaussian_conditional(curr_slice, scale, means=mu)
            
            y_hat_slices.append(y_hat_slice)
            likelihoods_list.append(slice_likelihoods)

        # Concatenar fatias reconstruídas e probabilidades
        y_hat = torch.cat(y_hat_slices, dim=1)
        likelihoods = torch.cat(likelihoods_list, dim=1)

        return y_hat, likelihoods

    def compress(self, y, hyper_features):
        """
        Compacta o latente 'y' sequencialmente em fatias e em formato de xadrez,
        retornando uma lista contendo os bitstreams de strings codificados aritmeticamente.
        """
        B, C, H, W = y.size()
        device = y.device
        
        y_slices = torch.chunk(y, self.num_slices, dim=1)
        y_hat_slices = []
        strings_list = []

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
            mu_anchor, scale_anchor = params_anchor.chunk(2, dim=1)

            # Divide em xadrez
            y_k_anchors, y_k_non_anchors = split_checkerboard(curr_slice)
            mu_anchor_split, _ = split_checkerboard(mu_anchor)
            scale_anchor_split, _ = split_checkerboard(scale_anchor)

            # Comprime as âncoras
            anchor_strings = self.gaussian_conditional.compress(y_k_anchors, scale_anchor_split, means=mu_anchor_split)
            strings_list.append(anchor_strings)

            # Decomprime as âncoras localmente para contexto da segunda fase
            y_k_anchors_hat = self.gaussian_conditional.decompress(anchor_strings, scale_anchor_split, means=mu_anchor_split)
            zeros_non_anchors = torch.zeros_like(y_k_non_anchors)
            y_k_anchor_only = merge_checkerboard(y_k_anchors_hat, zeros_non_anchors, H, W)

            # 3. Segunda Fase (Não-Âncoras)
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_k_anchor_only)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor = params_non_anchor.chunk(2, dim=1)

            _, mu_non_anchor_split = split_checkerboard(mu_non_anchor)
            _, scale_non_anchor_split = split_checkerboard(scale_non_anchor)

            # Comprime as não-âncoras
            non_anchor_strings = self.gaussian_conditional.compress(y_k_non_anchors, scale_non_anchor_split, means=mu_non_anchor_split)
            strings_list.append(non_anchor_strings)

            # Decomprime não-âncoras localmente
            y_k_non_anchors_hat = self.gaussian_conditional.decompress(non_anchor_strings, scale_non_anchor_split, means=mu_non_anchor_split)

            # Reconstrói a fatia inteira e acumula
            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            y_hat_slices.append(y_hat_slice)

        return strings_list

    def decompress(self, strings_list, hyper_features, H, W):
        """
        Decomprime o latente 'y' sequencialmente em fatias e em formato de xadrez,
        usando as tabelas de entropia construídas.
        """
        B = hyper_features.size(0)
        device = hyper_features.device
        
        y_hat_slices = []
        string_idx = 0

        for k in range(self.num_slices):
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
            mu_anchor, scale_anchor = params_anchor.chunk(2, dim=1)

            mu_anchor_split, _ = split_checkerboard(mu_anchor)
            scale_anchor_split, _ = split_checkerboard(scale_anchor)

            # Decomprime âncoras
            anchor_strings = strings_list[string_idx]
            string_idx += 1
            y_k_anchors_hat = self.gaussian_conditional.decompress(anchor_strings, scale_anchor_split, means=mu_anchor_split)
            
            zeros_non_anchors = torch.zeros(B, self.slice_size, H, W // 2, device=device)
            y_k_anchor_only = merge_checkerboard(y_k_anchors_hat, zeros_non_anchors, H, W)

            # 3. Segunda Fase (Não-Âncoras)
            spatial_ctx_non_anchors = self.spatial_context_networks[k](y_k_anchor_only)
            inputs_non_anchor = [hyper_features]
            if channel_ctx is not None:
                inputs_non_anchor.append(channel_ctx)
            inputs_non_anchor.append(spatial_ctx_non_anchors)
            
            params_non_anchor = self.entropy_parameter_networks[k](torch.cat(inputs_non_anchor, dim=1))
            mu_non_anchor, scale_non_anchor = params_non_anchor.chunk(2, dim=1)

            _, mu_non_anchor_split = split_checkerboard(mu_non_anchor)
            _, scale_non_anchor_split = split_checkerboard(scale_non_anchor)

            # Decomprime não-âncoras
            non_anchor_strings = strings_list[string_idx]
            string_idx += 1
            y_k_non_anchors_hat = self.gaussian_conditional.decompress(non_anchor_strings, scale_non_anchor_split, means=mu_non_anchor_split)

            # Mescla e acumula
            y_hat_slice = merge_checkerboard(y_k_anchors_hat, y_k_non_anchors_hat, H, W)
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return y_hat
