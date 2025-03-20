import requests
import gzip
import shutil

# URL do modelo FastText para português (Crawl vectors)
url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
local_gz = "cc.pt.300.vec.gz"
local_vec = "cc.pt.300.vec"

# Função para baixar o arquivo
def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Arquivo '{local_filename}' baixado com sucesso!")

# Função para descompactar o arquivo .gz
def decompress_gzip(gz_path, output_path):
    with gzip.open(gz_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Arquivo descompactado para '{output_path}'.")

if __name__ == "__main__":
    download_file(url, local_gz)
    decompress_gzip(local_gz, local_vec)
