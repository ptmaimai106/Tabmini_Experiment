import os
import gzip
import shutil


def extract_gz_files(folder_path, output_folder):
    """
    Quét thư mục để tìm các tệp .tsv.gz và giải nén thành .tsv vào thư mục chỉ định
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tsv.gz"):
                gz_path = os.path.join(root, file)
                tsv_path = os.path.join(output_folder, file[:-3])  # Bỏ đuôi .gz

                with gzip.open(gz_path, 'rb') as f_in, open(tsv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                print(f"Đã giải nén: {gz_path} -> {tsv_path}")


# Thay đổi 'your_directory' và 'output_directory' thành đường dẫn phù hợp
folder_to_scan = "datasets_compressed"
output_directory = "datasets_decompressed"
extract_gz_files(folder_to_scan, output_directory)