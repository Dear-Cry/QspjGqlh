import os
import urllib.request
import tarfile

def download_CIFAR10(url, dir):
    # Download and decompress CIFAR-10 dataset
    filename = url.split('/')[-1]
    file_path = os.path.join(dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(dir):
            os.makedirs(dir)

        print("Downloading...")
        file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path)
        print("Download completes.")

        tarfile.open(name=file_path, mode="r:gz").extractall(dir)

        print("Decompression completes.")
    else:
        print("Data has already been downloaded.")
  