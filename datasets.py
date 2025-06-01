# datasets.py
import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import glob
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from typing import Tuple, Any, List, Dict

# Assuming CIFAR_M and ImageFolder might be used for other experiments, you can keep their imports if needed.
# from CIFAR_M import CIFAR_M
# from dataset_folder import ImageFolder # This was for classification-style ImageFolder

class RandomMaskingGenerator:
    """
    Generates a random mask for MAE-style processing.
    Output: A boolean numpy array where True means "masked".
    """
    def __init__(self, input_size_patches: Tuple[int, int], mask_ratio: float):
        # input_size_patches: (num_patches_height, num_patches_width) for the ViT encoder
        self.height_patches, self.width_patches = input_size_patches
        self.num_patches = self.height_patches * self.width_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        if self.num_mask < 0: self.num_mask = 0
        if self.num_mask > self.num_patches: self.num_mask = self.num_patches


    def __repr__(self):
        return f"RandomMasker(total_patches={self.num_patches}, num_mask={self.num_mask})"

    def __call__(self) -> np.ndarray: # Returns boolean mask
        if self.num_mask == 0: # No masking, all visible
            return np.zeros(self.num_patches, dtype=bool)
        if self.num_mask == self.num_patches: # All masked (unlikely useful for recon training)
            return np.ones(self.num_patches, dtype=bool)
            
        # True means MASKED, False means VISIBLE
        mask = np.hstack([
            np.ones(self.num_mask, dtype=bool),         # Masked tokens
            np.zeros(self.num_patches - self.num_mask, dtype=bool) # Visible tokens
        ])
        np.random.shuffle(mask)
        return mask


class SemComInputProcessor:
    """
    Applies image transformations (resize, ToTensor) and generates the
    SemCom patch mask for the ViT encoder.
    """
    def __init__(self,
                 image_pixel_size: int, # Target H, W for the image tensor
                 semcom_patch_grid_size: Tuple[int, int], # (num_patches_h, num_patches_w)
                 mask_ratio: float,
                 is_train: bool): # is_train is not used here but kept for interface consistency
        self.image_pixel_size = image_pixel_size

        # Image transform to get a CHW tensor in [0,1] range
        # This is the target for SemCom reconstruction and input to SemCom.
        self.image_transform = transforms.Compose([
            transforms.Resize((image_pixel_size, image_pixel_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # Scales PIL image [0,255] to PyTorch tensor [0,1]
            # No further normalization if SemCom reconstructs to [0,1] (due to final sigmoid)
        ])

        self.mask_generator = RandomMaskingGenerator(semcom_patch_grid_size, mask_ratio)

    def __call__(self, image_pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        # image_pil: Input PIL Image
        transformed_image_tensor = self.image_transform(image_pil) # CHW, [0,1]
        
        # Generate boolean mask for SemCom encoder's patches (True means masked)
        semcom_patch_mask_np = self.mask_generator()
        semcom_patch_mask_tensor = torch.from_numpy(semcom_patch_mask_np) # Boolean Tensor
        
        return transformed_image_tensor, semcom_patch_mask_tensor


class YOLODataset(data.Dataset):
    def __init__(self,
                 img_root_dir_for_split: str,
                 img_pixel_size: int,
                 semcom_vit_patch_size: int, # Pass ViT patch size (e.g., 16 or 8)
                 semcom_encoder_mask_ratio: float,
                 is_train_split: bool,
                 num_object_classes: int # For creating target importance map
                 ):
        self.img_dir = os.path.join(img_root_dir_for_split, 'images')
        self.label_dir = os.path.join(img_root_dir_for_split, 'labels')
        self.img_pixel_size = img_pixel_size
        self.is_train_split = is_train_split
        self.vit_patch_size = semcom_vit_patch_size
        self.num_patches_h = img_pixel_size // semcom_vit_patch_size
        self.num_patches_w = img_pixel_size // semcom_vit_patch_size
        self.num_total_patches = self.num_patches_h * self.num_patches_w
        self.num_object_classes = num_object_classes


        # ... (img_files and label_files loading logic as before) ...
        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, '*.jpg')) +
            glob.glob(os.path.join(self.img_dir, '*.png')) +
            glob.glob(os.path.join(self.img_dir, '*.jpeg'))
        )
        self.label_files = [
            os.path.join(self.label_dir, os.path.splitext(os.path.basename(f))[0] + '.txt')
            for f in self.img_files
        ]
        initial_img_count = len(self.img_files)
        if self.is_train_split or (self.label_files and os.path.exists(self.label_files[0])): # Check if labels expected
            valid_indices = [i for i, lf in enumerate(self.label_files) if os.path.exists(lf)]
            self.img_files = [self.img_files[i] for i in valid_indices]
            self.label_files = [self.label_files[i] for i in valid_indices]
            if len(self.img_files) < initial_img_count:
                print(f"Warning: {initial_img_count - len(self.img_files)} images removed from {self.img_dir} due to missing labels.")
        if not self.img_files:
            raise FileNotFoundError(f"No image/label pairs found for {img_root_dir_for_split}.")
        # --- End img_files loading ---


        # SemComInputProcessor now takes patch_grid_size directly
        self.semcom_processor = SemComInputProcessor(
            image_pixel_size=img_pixel_size,
            semcom_patch_grid_size=(self.num_patches_h, self.num_patches_w), # Pass calculated grid size
            mask_ratio=semcom_encoder_mask_ratio,
            is_train=is_train_split
        )

    def __len__(self):
        return len(self.img_files)

    def _create_patch_importance_map(self, gt_boxes_abs_xyxy: torch.Tensor) -> torch.Tensor:
        """
        Creates a target importance map for FIM training.
        Output: [NumTotalPatches, 1], where 1 means important (fish), 0 means not.
        """
        target_map_flat = torch.zeros(self.num_total_patches, 1, dtype=torch.float32)

        if gt_boxes_abs_xyxy.numel() == 0:
            return target_map_flat

        # Create patch coordinates
        patch_coords_x = torch.arange(0, self.img_pixel_size, self.vit_patch_size)
        patch_coords_y = torch.arange(0, self.img_pixel_size, self.vit_patch_size)

        patch_idx = 0
        for r_idx in range(self.num_patches_h):
            for c_idx in range(self.num_patches_w):
                # Patch boundaries in absolute pixel coordinates
                p_x1 = patch_coords_x[c_idx]
                p_y1 = patch_coords_y[r_idx]
                p_x2 = p_x1 + self.vit_patch_size
                p_y2 = p_y1 + self.vit_patch_size

                # Check overlap with any GT box
                is_important = False
                for box_abs in gt_boxes_abs_xyxy:
                    b_x1, b_y1, b_x2, b_y2 = box_abs

                    # Check for intersection (IoU > 0 essentially)
                    inter_x1 = torch.max(p_x1, b_x1)
                    inter_y1 = torch.max(p_y1, b_y1)
                    inter_x2 = torch.min(p_x2, b_x2)
                    inter_y2 = torch.min(p_y2, b_y2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        is_important = True
                        break # Patch overlaps with at least one box
                
                if is_important:
                    target_map_flat[patch_idx, 0] = 1.0
                patch_idx += 1
        
        return target_map_flat


    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]: # Added FIM target
        # ... (img_pil loading as before) ...
        img_path = self.img_files[index]; label_path = self.label_files[index]
        try: img_pil = Image.open(img_path).convert('RGB')
        except Exception as e: print(f"Error loading image {img_path}: {e}. Retrying next."); return self.__getitem__((index + 1) % len(self))

        img_tensor_for_semcom, semcom_encoder_patch_mask = self.semcom_processor(img_pil)

        # ... (YOLO label loading -> boxes_normalized_xyxy, class_labels as before) ...
        boxes_normalized_xyxy = []; class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            class_id = int(parts[0]); cx, cy, w, h = map(float, parts[1:])
                            x_min = np.clip(cx - w / 2, 0.0, 1.0); y_min = np.clip(cy - h / 2, 0.0, 1.0)
                            x_max = np.clip(cx + w / 2, 0.0, 1.0); y_max = np.clip(cy + h / 2, 0.0, 1.0)
                            if x_max > x_min and y_max > y_min:
                                boxes_normalized_xyxy.append([x_min, y_min, x_max, y_max]); class_labels.append(class_id)
                        except ValueError: print(f"Warning: Malformed line in {label_path}: '{line.strip()}'")

        boxes_normalized_tensor = torch.as_tensor(boxes_normalized_xyxy, dtype=torch.float32)
        labels_tensor = torch.as_tensor(class_labels, dtype=torch.int64)
        abs_pixel_boxes_tensor = boxes_normalized_tensor.clone()
        if abs_pixel_boxes_tensor.numel() > 0:
            abs_pixel_boxes_tensor[:, [0, 2]] *= self.img_pixel_size; abs_pixel_boxes_tensor[:, [1, 3]] *= self.img_pixel_size
        
        yolo_gt_for_metrics_dict = {"boxes": abs_pixel_boxes_tensor, "labels": labels_tensor}

        # --- Create Target Importance Map for FIM ---
        fim_target_importance_map = self._create_patch_importance_map(abs_pixel_boxes_tensor)
        # Shape: [NumTotalPatches, 1]
        # --- End FIM Target ---

        semcom_input_tuple = (img_tensor_for_semcom, semcom_encoder_patch_mask)
        targets_tuple = (img_tensor_for_semcom.clone(), yolo_gt_for_metrics_dict, fim_target_importance_map) # Add FIM target

        return semcom_input_tuple, targets_tuple


# --- Modify build_dataset to pass patch_size to YOLODataset ---
def build_dataset(is_train: bool, args: Any) -> data.Dataset:
    if args.data_set == 'fish':
        # ... (data_split_subdir logic as before) ...
        if is_train: data_split_subdir = 'train'
        elif not args.eval: data_split_subdir = 'valid'
        else: data_split_subdir = 'test'
        current_split_root_dir = os.path.join(args.data_path, data_split_subdir)
        # ... (error checking for current_split_root_dir) ...
        if not os.path.isdir(current_split_root_dir):
            raise FileNotFoundError(f"Dataset dir for '{data_split_subdir}' not found: {current_split_root_dir}")

        dataset = YOLODataset(
            img_root_dir_for_split=current_split_root_dir,
            img_pixel_size=args.input_size,
            semcom_vit_patch_size=args.patch_size, # <--- PASS PATCH SIZE HERE
            semcom_encoder_mask_ratio=args.mask_ratio,
            is_train_split=is_train,
            num_object_classes=args.num_object_classes
        )
    else:
        raise NotImplementedError(f"Dataset '{args.data_set}' not implemented.")
    return dataset


# --- Modify yolo_collate_fn ---
def yolo_collate_fn(batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                       Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]] # Add FIM target type
                   ) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]]: # Add collated FIM target
    semcom_input_images = []
    semcom_encoder_masks = []
    semcom_reconstruction_targets = []
    yolo_gt_target_list_of_dicts = []
    fim_target_importance_maps = [] # For FIM targets

    for item in batch:
        semcom_input_tuple, targets_tuple = item
        
        semcom_input_images.append(semcom_input_tuple[0])
        semcom_encoder_masks.append(semcom_input_tuple[1])
        
        semcom_reconstruction_targets.append(targets_tuple[0])
        yolo_gt_target_list_of_dicts.append(targets_tuple[1])
        fim_target_importance_maps.append(targets_tuple[2]) # Collate FIM target

    collated_semcom_input_images = torch.stack(semcom_input_images, 0)
    collated_semcom_encoder_masks = torch.stack(semcom_encoder_masks, 0) # Should be okay if all are same shape
    collated_semcom_reconstruction_targets = torch.stack(semcom_reconstruction_targets, 0)
    collated_fim_target_importance_maps = torch.stack(fim_target_importance_maps, 0) # Stack FIM targets

    collated_semcom_input_tuple = (collated_semcom_input_images, collated_semcom_encoder_masks)
    collated_targets_tuple = (
        collated_semcom_reconstruction_targets,
        yolo_gt_target_list_of_dicts,
        collated_fim_target_importance_maps # Add to output
    )
    return collated_semcom_input_tuple, collated_targets_tuple