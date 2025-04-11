import os
import random


def reduce_dataset(images_root, labels_root, subset_percentage=0.10):
    """
    Keeps only a percentage of images (and corresponding JSON labels) in each subset directory.

    Parameters:
      images_root: Path to the root folder containing the image subsets (e.g., "images")
      labels_root: Path to the root folder containing the label subsets (e.g., "labels")
      subset_percentage: Fraction of images to keep (default 0.10 for 10%)
    """
    for subset in ["train", "test", "val"]:
        images_dir = os.path.join(images_root, subset)
        labels_dir = os.path.join(labels_root, subset)

        # List image files (you can add other image extensions if needed)
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total = len(image_files)
        if total == 0:
            print(f"No images found in {images_dir}.")
            continue

        # Determine how many images to keep (at least 1)
        keep_count = max(1, int(total * subset_percentage))

        # Randomly select keep_count images
        keep_set = set(random.sample(image_files, keep_count))
        print(f"For '{subset}': total images = {total}, keeping {len(keep_set)}, removing {total - len(keep_set)}")

        # Delete images and corresponding JSON files that are not in the keep_set
        for img_file in image_files:
            if img_file not in keep_set:
                # Remove image file
                img_path = os.path.join(images_dir, img_file)
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Error removing {img_path}: {e}")

                # Remove corresponding JSON file
                json_file = os.path.splitext(img_file)[0] + ".json"
                json_path = os.path.join(labels_dir, json_file)
                if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                    except Exception as e:
                        print(f"Error removing {json_path}: {e}")
                else:
                    print(f"JSON file {json_path} does not exist.")


if __name__ == "__main__":
    # Set your paths here (adjust if necessary)
    images_root = "images"
    labels_root = "labels"

    # Keep only 10% of the images in each subset
    reduce_dataset(images_root, labels_root, subset_percentage=0.10)
