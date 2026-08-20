import os
import urllib.request
import zipfile
import shutil

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 100 / total_size
        print(f"Progresso: {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB)", end="\r")
    else:
        print(f"Baixado: {read_so_far / (1024*1024):.1f} MB", end="\r")

def main():
    url = "https://github.com/lemire/kodakimagecollection/archive/refs/heads/master.zip"
    dest_zip = "kodak.zip"
    dest_dir = "kodak24"
    
    print("Baixando dataset Kodak24...")
    try:
        urllib.request.urlretrieve(url, dest_zip, download_progress)
        print("\nDownload concluído. Extraindo arquivos...")
        
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall("temp_kodak")
            
        # Move PNG files to kodak24
        os.makedirs(dest_dir, exist_ok=True)
        extracted_folder = os.path.join("temp_kodak", "kodakimagecollection-master")
        
        for file_name in os.listdir(extracted_folder):
            if file_name.lower().endswith(".png"):
                shutil.move(os.path.join(extracted_folder, file_name), os.path.join(dest_dir, file_name))
                
        # Clean up temp
        shutil.rmtree("temp_kodak")
        if os.path.exists(dest_zip):
            os.remove(dest_zip)
            
        print(f"Dataset Kodak24 pronto na pasta: {os.path.abspath(dest_dir)}")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
