import os
import urllib.request
import zipfile

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 100 / total_size
        print(f"Progresso: {percent:.2f}% ({read_so_far / (1024*1024):.1f} MB de {total_size / (1024*1024):.1f} MB)", end="\r")
    else:
        print(f"Baixado: {read_so_far / (1024*1024):.1f} MB", end="\r")

def main():
    # URL do dataset oficial DIV2K (imagens de alta resolução de treino)
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    dest_zip = "DIV2K_train_HR.zip"
    dest_dir = "DIV2K_train_HR"
    
    print("="*80)
    print(" DOWNLOAD DO DATASET DIV2K (TREINAMENTO)")
    print("="*80)
    print(f"Iniciando o download de: {url}")
    print("Tamanho do arquivo: ~3.4 GB. Isso pode levar algum tempo dependendo da sua internet...")
    
    try:
        # Inicia download com barra de progresso
        urllib.request.urlretrieve(url, dest_zip, download_progress)
        print("\n\nDownload concluído com sucesso! Iniciando extração dos arquivos...")
        
        # Extrai o ZIP
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        print(f"\nExtração concluída! O dataset foi extraído para: {os.path.abspath(dest_dir)}")
        
        # Remove o arquivo ZIP para economizar espaço em disco
        if os.path.exists(dest_zip):
            os.remove(dest_zip)
            print("Arquivo ZIP temporário removido para liberar espaço.")
            
        print("="*80)
        print("Dataset pronto para uso!")
        print("="*80)
        
    except Exception as e:
        print(f"\nErro durante o download/extração: {e}")

if __name__ == "__main__":
    main()
