# evaluate_on_test_set.py
import torch
import os
import utils # This will import utils.py, which should now import model.py
from pathlib import Path
from ultralytics import YOLO # For loading your YOLO model

# Import necessary components from your project
from base_args import get_args # For parsing arguments like paths, SNR
from datasets import build_dataset, yolo_collate_fn # For loading your test dataset
from engine1 import evaluate_semcom_with_yolo # Your evaluation function
# 'get_model' and 'sel_criterion' will be accessed via the 'utils' module
# import model # No longer strictly needed here if utils.py imports it

def main_eval(args):
    args.distributed = False # Assuming single GPU for evaluation
    device = torch.device(args.device)
    utils.seed_initial(seed=args.seed) # For reproducibility

    print(f"--- Starting Final Evaluation on Test Set ---")
    print(f"Using SemCom checkpoint: {args.semcom_checkpoint_path}")
    print(f"Using YOLO weights: {args.yolo_weights}")
    print(f"Evaluation SNR for SemCom channel: {args.snr_db_eval} dB")
    print(f"Dataset root path for args: {args.data_path}, Test split will be derived.")

    # 1. Load Trained SemCom Model (Best PSNR Model)
    if not args.semcom_checkpoint_path or not os.path.isfile(args.semcom_checkpoint_path):
        print(f"Error: SemCom checkpoint not found at '{args.semcom_checkpoint_path}'")
        return

    # Use args.model to specify the architecture of the saved SemCom model
    print(f"Loading SemCom model architecture: {args.model}")
    # utils.get_model will call timm.create_model, which needs the model to be registered
    semcom_model = utils.get_model(args)
    
    print(f"Loading weights into SemCom model from {args.semcom_checkpoint_path}")
    # Use the robust checkpoint loading function from utils.py
    load_successful = utils.load_custom_checkpoint(
        model_to_load=semcom_model,
        checkpoint_path=args.semcom_checkpoint_path,
        model_key_in_ckpt='model|state_dict' # Common keys for model state
    )
    if not load_successful:
        print(f"Failed to load SemCom model weights from checkpoint. Exiting.")
        return
    
    semcom_model.to(device)
    semcom_model.eval() # Set to evaluation mode
    print(f"SemCom model configured and set to eval mode.")

    # Derive window_size for SemCom patch grid from the loaded model (after it's fully created)
    if hasattr(semcom_model, 'img_encoder') and hasattr(semcom_model.img_encoder, 'patch_embed'):
        patch_size_h = semcom_model.img_encoder.patch_embed.patch_size[0]
        patch_size_w = semcom_model.img_encoder.patch_embed.patch_size[1]
        args.window_size = (args.input_size // patch_size_h, args.input_size // patch_size_w)
    else: # Fallback if structure is different (e.g., model from timm hub without this specific structure)
        args.window_size = (args.input_size // 16, args.input_size // 16) # Assuming 16x16 patches
        print(f"Warning: SemCom patch_embed details not found directly, using default window_size based on 16x16 patches.")
    print(f"SemCom Patch Grid (window_size for dataset mask): {args.window_size} for input_size {args.input_size}")


    # 2. Load Pre-trained YOLOv11 Model
    yolo_model = None
    if args.yolo_weights and os.path.exists(args.yolo_weights):
        print(f"Loading YOLO model from: {args.yolo_weights}")
        try:
            yolo_model = YOLO(args.yolo_weights)  # Ultralytics YOLO
            yolo_model.to(device) # Move YOLO model to the same device as SemCom model
            yolo_model.eval()
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Object detection evaluation will be skipped.")
            yolo_model = None
    else:
        print(f"YOLO weights not found at '{args.yolo_weights}' or not specified. Object detection evaluation will be skipped.")

    # 3. Prepare Test Dataloader
    # The build_dataset function should use the 'test' subdirectory when args.eval is True
    args.eval_mode_active = args.eval # Store original --eval flag
    args.eval = True # Temporarily set for build_dataset to select 'test' split
    print(f"Building test dataset from: {os.path.join(args.data_path, 'test')}")
    test_dataset = build_dataset(is_train=False, args=args) # is_train=False for no training augs
    args.eval = args.eval_mode_active # Restore original --eval flag
    delattr(args, 'eval_mode_active') # Clean up temporary attribute
    
    if not test_dataset or len(test_dataset) == 0:
        print(f"Test dataset is empty or could not be loaded from '{os.path.join(args.data_path, 'test')}'. Exiting.")
        return

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=yolo_collate_fn # Your custom collate function
    )
    print(f"Test dataset loaded with {len(test_dataset)} samples, using {args.num_workers} workers.")

    # 4. Reconstruction Criterion (e.g., MSE for logging, not for training here)
    reconstruction_criterion = utils.sel_criterion(args).to(device)

    # 5. Perform Evaluation
    print("\nStarting evaluation on the test set...")
    eval_stats = evaluate_semcom_with_yolo(
        semcom_net=semcom_model,
        yolo_model=yolo_model,
        dataloader=test_dataloader,
        device=device,
        reconstruction_criterion=reconstruction_criterion,
        args=args,
        print_freq=getattr(args, 'save_freq', 20) # Use save_freq for print_freq or a default
    )

    print(f"\n--- FINAL TEST SET EVALUATION RESULTS (Eval SNR: {args.snr_db_eval} dB) ---")
    print(f"  SemCom Model Checkpoint: {args.semcom_checkpoint_path}")
    print(f"  YOLO Model Weights: {args.yolo_weights if yolo_model else 'Not Used/Loaded'}")
    print(f"  Reconstruction Loss (e.g., MSE): {eval_stats.get('rec_loss', float('nan')):.4f}")
    print(f"  VQ Loss (from SemCom): {eval_stats.get('vq_loss', float('nan')):.4f}")
    print(f"  PSNR: {eval_stats.get('psnr', float('nan')):.2f} dB")
    print(f"  SSIM: {eval_stats.get('ssim', float('nan')):.4f}")

    if yolo_model and utils.TORCHMETRICS_AVAILABLE:
        print(f"  Object Detection mAP: {eval_stats.get('map', 0.0):.4f}")
        print(f"  Object Detection mAP@.50IOU: {eval_stats.get('map_50', 0.0):.4f}") # Common mAP50
        print(f"  Object Detection mAP@.75IOU: {eval_stats.get('map_75', 0.0):.4f}")
        # Print other mAP metrics if available in eval_stats (torchmetrics provides many)
        for k, v in eval_stats.items():
            if (k.startswith('map_') and k not in ['map','map_50','map_75']) or \
               k.startswith('mar_') or k.startswith('map_cls_'):
                 print(f"  {k}: {v:.4f}")
    elif yolo_model and not utils.TORCHMETRICS_AVAILABLE:
        print("  Object Detection mAP: Not calculated (torchmetrics library not available or import failed).")
    else:
        print("  Object Detection mAP: Not calculated (YOLO model not loaded).")
    print("--------------------------------------------------------------------")


if __name__ == '__main__':
    eval_args = get_args() # Parse all arguments defined in base_args.py

    # Use the --resume argument to specify the SemCom checkpoint path for evaluation
    if not eval_args.resume:
        print("CRITICAL Error: Please provide the path to the SemCom model checkpoint to evaluate "
              "using the --resume argument.")
        print("Example: --resume path/to/your/model_best_psnr.pth")
        exit(1)
    eval_args.semcom_checkpoint_path = eval_args.resume

    # Set a default output_dir if not specified (e.g., for saving debug images if added later)
    if not eval_args.output_dir:
        eval_args.output_dir = "eval_output_on_test_set"
    Path(eval_args.output_dir).mkdir(parents=True, exist_ok=True)

    main_eval(eval_args)