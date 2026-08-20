# NIF - Neural Image Format 
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
##  Resultados Obtidos (Avaliado no Benchmark Kodak24)
O modelo foi treinado por 300 épocas no dataset de alta resolução **DIV2K** e avaliado no benchmark padrão da literatura **Kodak24** (completamente disjunto do conjunto de treino), apresentando os seguintes resultados médios de taxa-distorção:

* **Fator de Qualidade q = 0.50**:
  * **Bitrate Médio:** **0.8503 bpp**
  * **PSNR Médio:** **28.77 dB**
  * **MS-SSIM Médio:** **0.98382** (fidelidade estrutural ultra-alta percebida pelo cérebro)
  * **LPIPS Médio:** **0.05065** (erro perceptual extremamente baixo)
* **Queda Perceptual Confiável**: A avaliação em imagens de teste que o modelo nunca viu comprova a capacidade de generalização e a robustez do pipeline.
---
##  Dashboard Web Interativo
O projeto acompanha um dashboard web local construído em Flask para demonstrações visuais e análises em tempo real:
* **Slide-to-Compare**: Comparador deslizante vertical interativo para visualizar lado a lado a imagem Original e a Reconstruída pela IA.
* **Estatísticas em Tempo Real**: Exibição instantânea do ganho de compressão, taxa (bpp), PSNR e MS-SSIM.
* **Download Binário**: Compactação e descompactação completas via drag-and-drop.
---
##  Como Executar

### 1. Instalação das Dependências

Garante a instalação do PyTorch, CompressAI e dependências do dashboard:
```bash
pip install -r requirements.txt
pip install Flask
```
2. Treinamento do Modelo
Para treinar o modelo do zero usando o dataset DIV2K:

```bash
python scripts/train.py --dataset DIV2K_train_HR --epochs 300
```
3. Compactação e Descompactação via CLI

Para compactar uma imagem:
```bash
python nif_tool.py compress --checkpoint checkpoints/nif_epoch_300.pth --input NOME_DA_IMAGEM.png --output imagem_comprimida.nif --quality 0.5
```
Para descompactar o arquivo .nif:
```bash
python nif_tool.py decompress --checkpoint checkpoints/nif_epoch_300.pth --input imagem_comprimida.nif --output reconstruida.png
```
4. Abrir o Dashboard Interativo

Para iniciar a interface web:
```bash
python scripts/dashboard.py
```
Em seguida, abra o navegador em: http://localhost:5000
