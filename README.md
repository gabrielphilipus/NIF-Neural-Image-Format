# NIF - Neural Image Format 

[![Python Package](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml)
[![Pylint](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

O **NIF (Neural Image Format)** é um codec de compressão de imagem baseado em **aprendizado profundo de ponta a ponta (End-to-End Learned Image Compression)**. Ele utiliza redes neurais convolucionais profundas acopladas a mecanismos de atenção, modulação de taxa variável e codificação de entropia avançada para superar as limitações dos formatos tradicionais.

---

##  Destaques & Arquitetura

O codec foi estruturado com as tecnologias mais recentes em compressão neural de imagens:

1. **Codificador/Decodificador Variencial (FiLM-Conditioned)**: Suporta compressão em taxa variável com um único modelo treinado, controlando o nível de qualidade ($q$) continuamente via camadas de modulação FiLM.
2. **Mecanismo de Atenção por Canal (Squeeze-and-Excitation)**: Integrado em todas as 4 escalas do Encoder e Decoder para guiar dinamicamente a alocação de bits para regiões com texturas mais complexas (bordas, textos, etc.).
3. **Filtro de Loop Neural Residual (NLF)**: Um pós-processador leve baseado em blocos residuais acoplado ao final do decodificador para refinar a imagem de saída e eliminar artefatos residuais de reconstrução.
4. **Modelo de Entropia Checkerboard + GMM**:
   * Abordagem autoregressiva em xadrez espacial (Checkerboard) combinada com contexto entre canais (Channel-wise).
   * Substituição do modelo Gaussiano simples por **GMM (Gaussian Mixture Model)** com $K=3$ componentes para modelar distribuições hiper-complexas.
   * Codificador/Decodificador aritmético (ANS) embarcado nativamente em C++.

---

##  Resultados Obtidos (Benchmark Kodak24 Completo - 24 Imagens)

O modelo foi avaliado no benchmark padrão da literatura **Kodak24** utilizando a flag `--aggregate` do script de avaliação. Apresentamos abaixo os resultados consolidados do **Novo Modelo (Checkpoint 250)** em comparação com o **Modelo Original (Checkpoint 265 antigo)** que sofria de colapso de condicionamento (*Conditioning Collapse*):

### Tabela de Comparação de Desempenho (Média ± Desvio Padrão)

| Qualidade $q$ | Checkpoint 265 (Modelo Antigo com Colapso) | Checkpoint 250 (Modelo Novo Corrigido) |
| :---: | :--- | :--- |
| **0.10** | **Bpp:** $0.8438 \pm 0.1295$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.97903 \pm 0.00632$ <br>**LPIPS:** $0.05848 \pm 0.02047$ | **Bpp:** $0.7024 \pm 0.1190$ <br>**PSNR:** $27.56 \pm 2.38$ dB <br>**MS-SSIM:** $0.97507 \pm 0.00667$ <br>**LPIPS:** $0.07219 \pm 0.02410$ |
| **0.30** | **Bpp:** $0.8526 \pm 0.1300$ <br>**PSNR:** $27.98 \pm 2.48$ dB <br>**MS-SSIM:** $0.97919 \pm 0.00632$ <br>**LPIPS:** $0.05834 \pm 0.02049$ | **Bpp:** $0.7420 \pm 0.1230$ <br>**PSNR:** $27.61 \pm 2.40$ dB <br>**MS-SSIM:** $0.97608 \pm 0.00663$ <br>**LPIPS:** $0.07044 \pm 0.02408$ |
| **0.50** | **Bpp:** $0.8622 \pm 0.1304$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.97928 \pm 0.00631$ <br>**LPIPS:** $0.05824 \pm 0.02053$ | **Bpp:** $0.7841 \pm 0.1270$ <br>**PSNR:** $27.65 \pm 2.42$ dB <br>**MS-SSIM:** $0.97688 \pm 0.00669$ <br>**LPIPS:** $0.06933 \pm 0.02404$ |
| **0.70** | **Bpp:** $0.8735 \pm 0.1309$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.97940 \pm 0.00636$ <br>**LPIPS:** $0.05809 \pm 0.02045$ | **Bpp:** $0.8267 \pm 0.1301$ <br>**PSNR:** $27.67 \pm 2.43$ dB <br>**MS-SSIM:** $0.97749 \pm 0.00674$ <br>**LPIPS:** $0.06871 \pm 0.02401$ |
| **0.90** | **Bpp:** $0.8879 \pm 0.1314$ <br>**PSNR:** $27.97 \pm 2.47$ dB <br>**MS-SSIM:** $0.97949 \pm 0.00636$ <br>**LPIPS:** $0.05809 \pm 0.02048$ | **Bpp:** $0.8665 \pm 0.1330$ <br>**PSNR:** $27.68 \pm 2.43$ dB <br>**MS-SSIM:** $0.97791 \pm 0.00679$ <br>**LPIPS:** $0.06881 \pm 0.02417$ |

### Principais Conclusões dos Resultados:
* **Mitigação do Conditioning Collapse:** A variação do bitrate entre $q=0.10$ e $q=0.90$ aumentou de **5.2%** (no modelo antigo) para **23.4%** (no modelo novo), representando um aumento de **~4.5x na sensibilidade** da curva de taxa de bits.
* **Comportamento Monótono:** A qualidade de reconstrução (PSNR/MS-SSIM) agora evolui de forma monótona e crescente conforme a qualidade demandada $q$ se eleva.

---

##  Como Executar as Avaliações e Benchmarks

### 1. Script de Avaliação (`eval.py`)
Para rodar a avaliação do modelo de forma agregada sobre todas as imagens do dataset Kodak24:
```bash
python scripts/eval.py --checkpoint checkpoints_v2/nif_epoch_250.pth --image kodak24/ --aggregate --save_output
```

### 2. Script de Benchmark Comparativo (`benchmark.py`)
O script de benchmark foi estendido para rodar de forma automática tanto em arquivos individuais quanto em diretórios inteiros (em lote). 

Para avaliar e gerar curvas Rate-Distortion comparativas com **JPEG e WebP** com barras de erro do desvio padrão ($\pm$ std):
```bash
python scripts/benchmark.py --checkpoint checkpoints_v2/nif_epoch_250.pth --image kodak24 --output_dir results/benchmark_epoch_250
```
Isso gerará os arquivos:
* `results/benchmark_epoch_250/benchmark_aggregated.json` (pontos brutos e estatísticas agregadas).
* `rd_curve_psnr.png`, `rd_curve_msssim.png` e `rd_curve_lpips.png` contendo as curvas de Rate-Distortion.

---

##  Dashboard Web Interativo

O projeto acompanha um dashboard web local construído em Flask para demonstrações visuais e análises em tempo real:
* **Slide-to-Compare**: Comparador deslizante vertical interativo para visualizar lado a lado a imagem Original e a Reconstruída pela IA.
* **Estatísticas em Tempo Real**: Exibição instantânea do ganho de compressão, taxa (bpp), PSNR e MS-SSIM.
* **Download Binário**: Compactação e descompactação completas via drag-and-drop.

---

##  Como Executar o Projeto Localmente

### 1. Instalação das Dependências
Garante a instalação do PyTorch, CompressAI e dependências do dashboard:
```bash
pip install -r requirements.txt
pip install Flask
```

### 2. Treinamento do Modelo
Para treinar o modelo do zero usando o dataset DIV2K:
```bash
python scripts/train.py --dataset DIV2K_train_HR --epochs 300
```

### 3. Compactação e Descompactação via CLI
Para compactar uma imagem:
```bash
python nif_tool.py compress --checkpoint checkpoints_v2/nif_epoch_250.pth --input NOME_DA_IMAGEM.png --output imagem_comprimida.nif --quality 0.5
```
Para descompactar o arquivo `.nif`:
```bash
python nif_tool.py decompress --checkpoint checkpoints_v2/nif_epoch_250.pth --input imagem_comprimida.nif --output reconstruida.png
```

### 4. Abrir o Dashboard Interativo
Para iniciar a interface web:
```bash
python scripts/dashboard.py
```
Em seguida, abra o navegador em: `http://localhost:5000`

---

##  Latência & Custo Operacional (AWS g4dn.xlarge)

O script de benchmark mede a velocidade de compressão/descompressão do codec NIF e estima o custo para processar **1 Milhão de imagens** em uma instância de referência na nuvem (AWS EC2 `g4dn.xlarge`, custo de **$0.526 por hora**):

| Dispositivo | Resolução | Encode (ms) | Decode (ms) | Total (ms) | Custo / 1M Imagens (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | 256x256 | 375.18 | 255.60 | 630.79 | **$92.16** |
| **CPU** | 512x512 | 1212.72 | 880.88 | 2093.59 | **$305.90** |
| **CPU** | 1024x1024 | 6461.80 | 5582.96 | 12044.76 | **$1759.87** |
| **GPU (CUDA)** | 256x256 | 406.20 | 245.09 | 651.29 | **$95.16** |
| **GPU (CUDA)** | 512x512 | 1141.65 | 604.92 | 1746.57 | **$255.19** |
| **GPU (CUDA)** | 1024x1024 | 4427.06 | 2222.63 | 6649.69 | **$971.59** |

Para rodar este benchmark em sua máquina:
```bash
python scripts/benchmark_latency.py
```

---

##  Limitações & Possíveis Melhorias
O formato NIF é um projeto experimental de pesquisa em compressão neural de imagens e possui as seguintes limitações conhecidas:
1. **Curva RD Estreita**: A variação absoluta de qualidade obtida de PSNR (0.11 dB) ainda é modesta, indicando que o condicionamento FiLM necessita de mais capacidade. Recomenda-se a aplicação de *Positional Encoding* no parâmetro de qualidade $q$ no codificador/MLP no próximo treinamento.
2. **Dependência de Checkpoints**: Assim como todo formato neural, a descompressão requer exatamente o mesmo modelo e pesos de treino que comprimiram a imagem.
3. **Espaço de Cor**: Atualmente operamos em espaço de cor RGB puro de 24-bits. Uma implementação futura integrará YCbCr 4:2:0 para melhorar em mais 15-20% as taxas de bitrate.
