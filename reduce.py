import os
import random
import glob
from pathlib import Path

def reduce_yolo_dataset_split(dataset_root_path: str, split_name: str, reduction_factor: float = 0.5, dry_run: bool = True):
    """
    Reduces a specific split (train, valid, test) of a YOLO dataset by deleting
    a random fraction of image-label pairs.

    Args:
        dataset_root_path (str): Path to the root directory of the YOLO dataset
                                 (e.g., '/path/to/dataset_root/').
        split_name (str): Name of the split to reduce (e.g., 'train', 'valid', 'test').
        reduction_factor (float): Fraction of files to REMOVE (0.5 means remove half, keep half).
                                  Must be between 0.0 and 1.0.
        dry_run (bool): If True, only print what would be deleted without actually deleting.
    """
    if not 0.0 <= reduction_factor <= 1.0:
        print("Error: reduction_factor must be between 0.0 and 1.0.")
        return

    print(f"\n--- Processing split: {split_name} ---")
    if dry_run:
        print("*** DRY RUN MODE: No files will be deleted. ***")

    img_dir = Path(dataset_root_path) / split_name / "images"
    label_dir = Path(dataset_root_path) / split_name / "labels"

    if not img_dir.is_dir():
        print(f"Image directory not found: {img_dir}")
        return
    # Label directory might not exist if it's an unlabeled test set, which is fine.

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_files.extend(glob.glob(str(img_dir / ext)))

    if not image_files:
        print(f"No image files found in {img_dir}")
        return

    num_total_images = len(image_files)
    num_to_remove = int(num_total_images * reduction_factor)
    num_to_keep = num_total_images - num_to_remove

    print(f"Found {num_total_images} images in '{split_name}/images/'.")
    print(f"Targeting to remove {num_to_remove} images (reduction factor: {reduction_factor*100:.1f}%)")
    print(f"This will keep approximately {num_to_keep} images.")

    if num_to_remove == 0:
        print("No images targeted for removal based on reduction factor.")
        return
    if num_to_remove >= num_total_images:
        print("Reduction factor targets all images for removal. Please adjust if this is not intended.")
        # You might want to add a confirmation step here for safety if reduction_factor is 1.0

    # Shuffle the list of image files to randomize selection
    random.shuffle(image_files)

    files_deleted_count = 0
    labels_deleted_count = 0

    # Select files to remove
    images_to_remove = image_files[:num_to_remove]

    for img_path_str in images_to_remove:
        img_path = Path(img_path_str)
        img_filename_stem = img_path.stem  # Filename without extension
        label_path = label_dir / f"{img_filename_stem}.txt"

        print(f"  Targeting for removal: {img_path}")
        if label_path.exists():
            print(f"    Corresponding label: {label_path}")
        else:
            print(f"    No corresponding label found for {img_path} (this is okay for test sets).")

        if not dry_run:
            try:
                os.remove(img_path)
                files_deleted_count += 1
                print(f"    DELETED: {img_path}")
                if label_path.exists():
                    os.remove(label_path)
                    labels_deleted_count +=1
                    print(f"    DELETED: {label_path}")
            except OSError as e:
                print(f"    Error deleting {img_path} or its label: {e}")

    print(f"--- {split_name} split processing complete ---")
    if dry_run:
        print(f"DRY RUN: Would have targeted {num_to_remove} image files for deletion.")
    else:
        print(f"Actually deleted {files_deleted_count} image files.")
        print(f"Actually deleted {labels_deleted_count} label files.")
    print(f"Remaining images in '{split_name}/images/' should be around: {num_total_images - files_deleted_count}")


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_ROOT = "" # IMPORTANT: SET YOUR ROOT PATH
    SPLITS_TO_PROCESS = ["valid"]  # Add "test" if you want to reduce it too
    REDUCTION_PERCENTAGE = 50.0 # Percentage of files to REMOVE
    DRY_RUN = False # SET TO False TO ACTUALLY DELETE FILES. START WITH True!

    # --- End Configuration ---

    if not Path(DATASET_ROOT).is_dir():
        print(f"Error: Dataset root directory not found: {DATASET_ROOT}")
        exit()

    reduction_factor_val = REDUCTION_PERCENTAGE / 100.0

    print(f"Starting dataset reduction process for root: {DATASET_ROOT}")
    print(f"Targeting to remove {REDUCTION_PERCENTAGE}% of files from each specified split.")
    if DRY_RUN:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!               DRY RUN IS ENABLED             !!!")
        print("!!! NO FILES WILL BE ACTUALLY DELETED THIS TIME. !!!")
        print("!!! Review the output carefully.                 !!!")
        print("!!! Set DRY_RUN = False to perform deletions.    !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        confirmation = input(
            f"WARNING: You are about to PERMANENTLY DELETE approximately {REDUCTION_PERCENTAGE}% of files\n"
            f"from splits {SPLITS_TO_PROCESS} in '{DATASET_ROOT}'.\n"
            "Ensure you have a BACKUP.\n"
            "Type 'YES_DELETE' to proceed: "
        )
        if confirmation != "YES_DELETE":
            print("Deletion cancelled by user.")
            exit()
        print("\n*** Proceeding with actual file deletion. ***\n")


    for split in SPLITS_TO_PROCESS:
        reduce_yolo_dataset_split(DATASET_ROOT, split, reduction_factor_val, DRY_RUN)

    print("\nDataset reduction process finished.")
    if DRY_RUN:
        print("Remember to set DRY_RUN = False in the script to actually delete files.")