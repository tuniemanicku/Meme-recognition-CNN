from pinterest_dl import PinterestDL
import os
import glob
import random
import shutil

def move_random_files(src_folder, dst_folder, percentage=20):
    """
    Moves a random percentage of files from src_folder to dst_folder.

    :param src_folder: Path to source folder
    :param dst_folder: Path to destination folder
    :param percentage: Percentage of files to move (default 20)
    """
    # Ensure destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # List all files in source folder (ignore subfolders)
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    if not all_files:
        print("No files to move.")
        return

    # Calculate number of files to move
    num_to_move = max(1, len(all_files) * percentage // 100)

    # Randomly select files
    files_to_move = random.sample(all_files, num_to_move)

    # Move files
    for file_name in files_to_move:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        shutil.move(src_path, dst_path)
        # print(f"Moved: {file_name}")

def delete_data(image_folder):
    # find all image files (jpg, png, jpeg, etc.)
    image_files = glob.glob(os.path.join(image_folder, "*.*"))

    # delete each file
    for f in image_files:
        try:
            os.remove(f)
            # print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    print("All images removed.")

def main():
    delete_data("memes")
    delete_data("test_memes")
    # Search for train images
    images = PinterestDL.with_api().search_and_download(
        query="memes",
        output_dir="memes",
        num=180
    )
    move_random_files(src_folder="memes", dst_folder="test_memes", percentage=20)

if __name__ == "__main__":
    main()