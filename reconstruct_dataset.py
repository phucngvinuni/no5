# reconstruct_dataset.py
import torch
import os
from PIL import Image
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from pathlib import Path
import argparse
import sys
from collections import OrderedDict
import shutil
import numpy as np

# --- Ensure local modules can be imported ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# It's crucial that model.py (where ViT_Reconstruction_Model_Default is registered)
# is imported BEFORE utils.get_model is called.
import model
import utils # For utils.get_model and utils.load_custom_checkpoint
from datasets import SemComInputProcessor # Assuming this has the necessary transforms

# --- Define seed_initial locally or ensure it's in the imported utils ---
def seed_initial_local(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
             torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed} for reconstruction script.")

def get_reconstruction_args():
    parser = argparse.ArgumentParser("SemCom Dataset Reconstruction Script", add_help=False)
    
    # --- Paths ---
    parser.add_argument('--semcom_checkpoint_path', required=True, type=str, help="Path to the trained SemCom model checkpoint.")
    parser.add_argument('--original_dataset_root', required=True, type=str, help="Root directory of the original dataset (containing splits like 'train', 'valid', 'test').")
    parser.add_argument('--split_to_reconstruct', default='train', type=str, choices=['train', 'valid', 'test'], help="Which split of the dataset to reconstruct.")
    parser.add_argument('--reconstructed_dataset_root', required=True, type=str, help="Root directory where reconstructed dataset will be saved.")

    # --- Model Architecture (MUST MATCH THE CHECKPOINT'S MODEL) ---
    parser.add_argument('--model', default='ViT_Reconstruction_Model_Default', type=str, help="Name of the SemCom model architecture used in the checkpoint.")
    parser.add_argument('--input_size', default=224, type=int, help="Image input size (H and W) the model was trained with.")
    parser.add_argument('--patch_size', default=8, type=int, help="Patch size used by the ViT encoder.")
    # Encoder params
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--encoder_depth', default=8, type=int)
    parser.add_argument('--encoder_num_heads', default=8, type=int)
    # Decoder params (for ViT blocks within decoder)
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--decoder_depth', default=6, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    # Quantizer params (assuming dual VQ setup, common dim for simplicity)
    parser.add_argument('--quantizer_dim', default=512, type=int, help="Embedding dimension for VQ codebooks.")
    parser.add_argument('--bits_vq_high', default=12, type=int, help="Bits for high-importance VQ.")
    parser.add_argument('--bits_vq_low', default=8, type=int, help="Bits for low-importance VQ.")
    parser.add_argument('--quantizer_commitment_cost', default=0.25, type=float) # Often part of VQ init
    # FIM params
    parser.add_argument('--fim_embed_dim', default=256, type=int)
    parser.add_argument('--fim_depth', default=2, type=int)
    parser.add_argument('--fim_num_heads', default=4, type=int)
    parser.add_argument('--fim_drop_rate', default=0.1, type=float)
    parser.add_argument('--fim_routing_threshold', default=0.6, type=float, help="Threshold for FIM routing to high VQ.") # If using dual VQ with routing
    # Dropout/DropPath (if varied from defaults during training)
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.1, type=float)

    # --- Reconstruction Process Parameters ---
    parser.add_argument('--snr_list_for_reconstruction', nargs='+', type=float, default=[22.0], help="List of SNR values (in dB) to reconstruct images for.")
    parser.add_argument('--channel_type_for_reconstruction', default='rayleigh',
                        choices=['none', 'awgn', 'rayleigh', 'rician', 'awgn_mimo'], # Add MIMO channel types if model supports
                        type=str, help="Channel type to simulate during reconstruction.")
    # Add MIMO antenna args if your model/channel simulation uses them
    parser.add_argument('--num_tx_antennas', default=1, type=int, help="Number of transmit antennas for MIMO.")
    parser.add_argument('--num_rx_antennas', default=1, type=int, help="Number of receive antennas for MIMO.")

    parser.add_argument('--batch_size_recon', default=16, type=int, help="Batch size for reconstruction inference.")
    parser.add_argument('--device', default='cuda', type=str, help="Device to run reconstruction on ('cuda' or 'cpu').")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility.")
    
    return parser.parse_args()


def reconstruct_and_save(args_recon):
    seed_initial_local(seed=args_recon.seed)
    device = torch.device(args_recon.device if torch.cuda.is_available() and args_recon.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # --- Load SemCom Model ---
    print(f"Loading SemCom model architecture: {args_recon.model}")
    # utils.get_model will pass all relevant args_recon attributes to the model factory
    semcom_model = utils.get_model(args_recon)
    
    print(f"Loading weights into SemCom model from {args_recon.semcom_checkpoint_path}")
    if not os.path.exists(args_recon.semcom_checkpoint_path):
        print(f"ERROR: SemCom checkpoint not found: {args_recon.semcom_checkpoint_path}"); return

    try:
        # Use weights_only=False as checkpoint contains args and other Python objects
        checkpoint = torch.load(args_recon.semcom_checkpoint_path, map_location='cpu', weights_only=False)
        print(f"  DEBUG (reconstruct_dataset): Checkpoint loaded. Keys: {list(checkpoint.keys())}")
    except _pickle.UnpicklingError as e_pickle:
        print(f"  ERROR (reconstruct_dataset): Pickle UnpicklingError while loading checkpoint: {e_pickle}")
        print(f"  Ensure the checkpoint file is not corrupted and was saved with a compatible PyTorch version.")
        return
    except Exception as e:
        print(f"  ERROR (reconstruct_dataset): Failed to load checkpoint with torch.load(): {e}")
        return

    model_state_dict_key = next((k for k in ['model', 'state_dict', 'model_state_dict'] if k in checkpoint), None)
    
    if model_state_dict_key:
        state_dict = checkpoint[model_state_dict_key]
        # Handle 'module.' prefix if the checkpoint was saved from DDP model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        try:
            missing_keys, unexpected_keys = semcom_model.load_state_dict(new_state_dict, strict=False)
            if missing_keys: print(f"  Warning (load_state_dict): Missing keys in model: {missing_keys}")
            if unexpected_keys: print(f"  Warning (load_state_dict): Unexpected keys in checkpoint: {unexpected_keys}")
            print("  SemCom model weights loaded.")
        except Exception as e_load_state:
            print(f"  ERROR (load_state_dict): Failed to load model state_dict: {e_load_state}")
            print(f"  This often means the loaded checkpoint's architecture (defined by args in recon.sh) does not match the current model's architecture.")
            return
    else:
        print(f"ERROR: Model state_dict not found in checkpoint with common keys ('model', 'state_dict', 'model_state_dict')."); return
    
    semcom_model.to(device)
    semcom_model.eval()

    # --- Prepare Input Processor (for resizing and ToTensor) ---
    # The patch grid size for SemComInputProcessor is derived inside YOLODataset from patch_size
    # Here, we only need the image transform part.
    # However, SemComInputProcessor also generates the mask. For reconstruction, we don't want masking.
    # So, a simpler transform is better.
    image_transform_for_recon = Compose([
        Resize((args_recon.input_size, args_recon.input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(), # Converts PIL [0,255] to Tensor [0,1]
    ])
    to_pil = ToPILImage()

    # --- Paths ---
    original_split_path = Path(args_recon.original_dataset_root) / args_recon.split_to_reconstruct
    original_image_dir = original_split_path / "images"
    original_label_dir = original_split_path / "labels" # For copying labels

    # Create base directory for this reconstruction run
    reconstructed_dataset_split_base = Path(args_recon.reconstructed_dataset_root) / args_recon.split_to_reconstruct
    reconstructed_label_dir_for_split = reconstructed_dataset_split_base / "labels"
    
    if not original_image_dir.is_dir():
        print(f"ERROR: Original image directory not found: {original_image_dir}"); return
    
    all_image_files = sorted(
        list(original_image_dir.glob('*.jpg')) +
        list(original_image_dir.glob('*.jpeg')) +
        list(original_image_dir.glob('*.png')) +
        list(original_image_dir.glob('*.bmp')) +
        list(original_image_dir.glob('*.webp'))
    )
    if not all_image_files: print(f"No images found in {original_image_dir}"); return
    print(f"Found {len(all_image_files)} images in {original_image_dir} to reconstruct for split '{args_recon.split_to_reconstruct}'.")

    # --- Reconstruction Loop for each specified SNR ---
    for snr_val in args_recon.snr_list_for_reconstruction:
        print(f"\n--- Reconstructing for SNR: {snr_val:.1f} dB ---")
        # Create SNR-specific image directory
        current_snr_reconstructed_image_dir = reconstructed_dataset_split_base / f"images_SNR_{snr_val:.0f}dB_{args_recon.channel_type_for_reconstruction}"
        current_snr_reconstructed_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy labels only once per split, not per SNR
        if not reconstructed_label_dir_for_split.exists():
             reconstructed_label_dir_for_split.mkdir(parents=True, exist_ok=True)
        if original_label_dir.is_dir():
            print(f"  Copying labels from {original_label_dir} to {reconstructed_label_dir_for_split} (if not already present)...")
            copied_labels_count_this_snr_run = 0
            for label_file in original_label_dir.glob("*.txt"):
                target_label_file = reconstructed_label_dir_for_split / label_file.name
                if not target_label_file.exists(): # Avoid re-copying if running for multiple SNRs
                    try:
                        shutil.copy2(label_file, target_label_file)
                        copied_labels_count_this_snr_run +=1
                    except Exception as e_copy:
                        print(f"    Warning: Could not copy label file {label_file.name}: {e_copy}")
            print(f"  Label copying complete for this split. Copied {copied_labels_count_this_snr_run} new label files.")
        else:
            print(f"  Warning: Original label directory {original_label_dir} not found. Labels will not be copied.")


        processed_count = 0
        for i in range(0, len(all_image_files), args_recon.batch_size_recon):
            batch_img_paths = all_image_files[i:i + args_recon.batch_size_recon]
            batch_img_tensors = []
            batch_img_names = []

            for img_path_obj in batch_img_paths: # img_path_obj is a Path object
                try:
                    img_pil = Image.open(img_path_obj).convert('RGB')
                    img_tensor = image_transform_for_recon(img_pil) # Apply defined transform
                    batch_img_tensors.append(img_tensor)
                    batch_img_names.append(img_path_obj.name)
                except Exception as e:
                    print(f"Skipping {img_path_obj.name} due to loading/transform error: {e}")
                    continue
            
            if not batch_img_tensors: # If all images in batch failed
                continue
                
            img_tensor_batch = torch.stack(batch_img_tensors).to(device)

            with torch.no_grad():
                # For reconstruction, bm_pos (encoder mask) is typically None or all False (mask_ratio=0)
                # The model's forward should handle bm_pos=None gracefully for full reconstruction.
                model_fwd_kwargs = {
                    '_eval': True,
                    'eval_snr_db': snr_val, 
                    'train_snr_db_min': snr_val, # Dummy values for training SNR range
                    'train_snr_db_max': snr_val,
                    # No need to pass channel_type here if model's forward uses its internal channel_simulator
                    # Or if you want to override, pass args_recon.channel_type_for_reconstruction
                }
                # Add MIMO args only if the channel type indicates MIMO and model supports it
                if "mimo" in args_recon.channel_type_for_reconstruction.lower():
                    model_fwd_kwargs['num_tx_antennas'] = args_recon.num_tx_antennas
                    model_fwd_kwargs['num_rx_antennas'] = args_recon.num_rx_antennas
                
                # Call model forward pass
                # The `bm_pos` for reconstruction should be all False (no masking)
                num_patches_for_batch = semcom_model.img_encoder.patch_embed.num_patches
                eval_bm_pos = torch.zeros(img_tensor_batch.shape[0], num_patches_for_batch, dtype=torch.bool, device=device)

                outputs_dict = semcom_model(img=img_tensor_batch, bm_pos=eval_bm_pos, **model_fwd_kwargs)

            reconstructed_batch = outputs_dict['reconstructed_image'].cpu()

            for idx_in_batch, recon_tensor in enumerate(reconstructed_batch):
                img_original_name = batch_img_names[idx_in_batch]
                try:
                    recon_pil_img = to_pil(recon_tensor.clamp(0, 1)) # Ensure range [0,1]
                    save_path = current_snr_reconstructed_image_dir / img_original_name
                    recon_pil_img.save(save_path)
                    processed_count += 1
                except Exception as e_save:
                    print(f"Error saving reconstructed image {img_original_name}: {e_save}")
            
            if ( (i // args_recon.batch_size_recon + 1) % 10 == 0 ) or (i + len(batch_img_paths) >= len(all_image_files)):
                print(f"  Processed {processed_count}/{len(all_image_files)} images for SNR {snr_val:.1f} dB...")
        
        print(f"Finished: Reconstructed {processed_count} images for SNR {snr_val:.1f} dB into {current_snr_reconstructed_image_dir}")

    print("\nDataset reconstruction script finished.")

if __name__ == '__main__':
    args_cli = get_reconstruction_args()
    
    # --- Argument Propagation to utils.get_model ---
    # Ensure that any arguments needed by your ViT_Reconstruction_Model's __init__
    # (and its factory function ViT_Reconstruction_Model_Default) are present in args_cli.
    # The get_reconstruction_args() function should define all necessary architectural args.
    # utils.get_model will then use these when calling timm.create_model.
    # Example: args_cli will have 'patch_size', 'encoder_embed_dim', etc.
    
    reconstruct_and_save(args_cli)