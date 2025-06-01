# run_class_main.py
import sys
import os

# --- Ensure local modules are prioritized ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- Explicitly import your model definition file HERE ---
import model # Ensures model registration with timm
# -------------------------------------------------------

import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from pathlib import Path

import utils
import engine1
from base_args import get_args
import datasets
import optim_factory

NativeScaler = utils.NativeScalerWithGradNormCount

# --- LPIPS Import and Global Variable ---
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    # This print will happen once when the script starts
    print("Warning: lpips library not found. Perceptual loss will be skipped. `pip install lpips`")
# --- End LPIPS ---


def seed_initial(seed=0, rank=0):
    actual_seed = seed + rank
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        if torch.cuda.device_count() > 1:
             torch.cuda.manual_seed_all(actual_seed)
    print(f"Seed set to {actual_seed} (base_seed={seed}, rank={rank})")


def main(args):
    utils.init_distributed_mode(args) # Sets args.distributed, args.rank, etc.
    device = torch.device(args.device)
    seed_initial(seed=args.seed, rank=utils.get_rank())

    # --- LPIPS Criterion Initialization ---
    lpips_criterion_instance = None
    if LPIPS_AVAILABLE and hasattr(args, 'lpips_loss_weight') and args.lpips_loss_weight > 0:
        try:
            lpips_criterion_instance = lpips.LPIPS(net='alex', verbose=False).to(device)
            print(f"Using LPIPS perceptual loss with weight: {args.lpips_loss_weight}")
        except Exception as e_lpips:
            print(f"Error initializing LPIPS: {e_lpips}. LPIPS loss will be skipped.")
            lpips_criterion_instance = None
    elif hasattr(args, 'lpips_loss_weight') and args.lpips_loss_weight > 0 and not LPIPS_AVAILABLE:
        print("LPIPS loss weight > 0 but lpips library not available. LPIPS loss will be skipped.")


    print(f"Creating SemCom model: {args.model}")
    semcom_model = utils.get_model(args) # This now happens after model.py is imported

    # --- Calculate window_size (for dataset's RandomMaskingGenerator) ---
    # This logic should ideally use args.patch_size directly if model is configurable
    # Or derive from a loaded model if --resume is used with a different patch_size
    # For simplicity, assuming args.patch_size is the definitive source for new runs.
    if args.patch_size > 0:
        args.window_size = (args.input_size // args.patch_size, args.input_size // args.patch_size)
    else: # Fallback, though patch_size=0 is unlikely
        args.window_size = (args.input_size // 16, args.input_size // 16)
        print(f"Warning: args.patch_size is not positive. Using default 16x16 assumption for window_size.")
    print(f"Derived SemCom Patch Grid (window_size): {args.window_size} for input_size {args.input_size} using patch_size {args.patch_size}")

    optimizer = optim_factory.create_optimizer(args, semcom_model)
    loss_scaler = NativeScaler() if args.device == 'cuda' and torch.cuda.is_available() else None

    # --- Resume Logic ---
    if args.resume:
        print(f"INFO: --resume flag is set to: {args.resume}")
        if os.path.exists(args.resume):
            print(f"INFO: Checkpoint file found at {args.resume}.")
            # Pass the optimizer and scaler instances to be potentially updated
            load_successful = utils.load_custom_checkpoint(
                model_to_load=semcom_model,
                checkpoint_path=args.resume,
                model_key_in_ckpt='model|state_dict', # Try common keys
                optimizer_to_load=optimizer,      # Pass optimizer instance
                loss_scaler_to_load=loss_scaler,    # Pass scaler instance
                args_for_epoch_resume=args        # Pass args object to update start_epoch
            )
            if load_successful:
                print(f"INFO: load_custom_checkpoint reported success. args.start_epoch is now: {args.start_epoch}")
            else:
                print(f"WARNING: load_custom_checkpoint reported failure for {args.resume}. Training may start from scratch or with partial load.")
                # If only model weights loaded but not optimizer/epoch, it's a partial resume.
                # Setting args.start_epoch = 0 ensures LR scheduler restarts if optimizer didn't load.
                if not ('optimizer' in torch.load(args.resume, map_location='cpu') and 'epoch' in torch.load(args.resume, map_location='cpu')):
                     args.start_epoch = 0
        else:
            print(f"WARNING: Resume checkpoint file NOT FOUND at {args.resume}. Training from scratch.")
            args.start_epoch = 0
    else:
        print("INFO: --resume flag not set. Training from scratch.")
        args.start_epoch = 0 # Ensure it's 0 if not resuming

    semcom_model.to(device)
    model_without_ddp = semcom_model # For single GPU

    # --- Load YOLO Model ---
    yolo_model = None
    if args.yolo_weights and os.path.exists(args.yolo_weights):
        print(f"Loading YOLO model from: {args.yolo_weights}")
        try:
            yolo_model = YOLO(args.yolo_weights)
            if hasattr(yolo_model, 'fuse') and callable(yolo_model.fuse): yolo_model.fuse()
            # Set YOLO to eval mode
            if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'eval'): yolo_model.model.eval()
            elif hasattr(yolo_model, 'eval'): yolo_model.eval()
            print("YOLO model loaded and prepared for evaluation.")
        except Exception as e:
            print(f"  Error loading YOLO model: {e}. Proceeding without YOLO evaluation.")
            yolo_model = None
    else:
        print(f"YOLO weights path '{args.yolo_weights}' not found or not specified. No YOLO evaluation.")

    # --- Datasets and Dataloaders ---
    print("Building train dataset...")
    trainset = datasets.build_dataset(is_train=True, args=args)
    print("Building validation dataset...")
    valset = datasets.build_dataset(is_train=False, args=args)

    if not trainset or len(trainset) == 0: print("ERROR: Training dataset is empty!"); exit(1)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
        drop_last=True, collate_fn=datasets.yolo_collate_fn
    )
    dataloader_val = None
    if valset and len(valset) > 0:
        dataloader_val = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem,
            drop_last=False, collate_fn=datasets.yolo_collate_fn
        )
    else:
        print("Validation dataset is empty or could not be loaded. Validation will be skipped.")

    # --- LR Schedulers ---
    num_training_steps_per_epoch = len(trainloader)
    if num_training_steps_per_epoch == 0: print("ERROR - Training dataloader is empty!"); exit(1)
    print(f"Number of training batches (steps) per epoch: {num_training_steps_per_epoch}")

    lr_schedule_values = utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    wd_schedule_values = None
    if args.weight_decay > 0 and (args.weight_decay_end is None or args.weight_decay_end != args.weight_decay): # Only if scheduling WD
        wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end if args.weight_decay_end is not None else args.weight_decay, args.epochs, num_training_steps_per_epoch)
        if wd_schedule_values is not None: print(f"  Weight Decay Schedule: Max WD = {max(wd_schedule_values):.7f}, Min WD = {min(wd_schedule_values):.7f}")

    # --- Loss Criterions ---
    base_reconstruction_criterion = utils.sel_criterion(args).to(device) # L1Loss(reduction='none')
    fim_criterion = nn.BCEWithLogitsLoss().to(device)

    # --- Evaluation Only Mode ---
    if args.eval:
        print("Evaluation mode selected (args.eval=True).")
        eval_viz_dir_path = Path(args.eval_viz_output_dir) / f"eval_run_snr{args.snr_db_eval}"
        eval_viz_dir_path.mkdir(parents=True, exist_ok=True)
        if dataloader_val is None: print("  Validation dataloader not available. Skipping eval."); exit(0)
        
        print("  Starting evaluation-only run...")
        test_stats = engine1.evaluate_semcom_with_yolo(
            semcom_net=semcom_model, yolo_model=yolo_model, dataloader=dataloader_val,
            device=device, reconstruction_criterion=base_reconstruction_criterion,
            fim_criterion=fim_criterion, lpips_criterion=lpips_criterion_instance, args=args,
            current_epoch_num="eval_only", viz_output_dir=str(eval_viz_dir_path)
        )
        print(f"\n--- Final Evaluation Results (SNR: {args.snr_db_eval} dB) ---")
        for key, val in test_stats.items():
            if isinstance(val, float): print(f"  {key}: {val:.4f}")
            else: print(f"  {key}: {val}")
        sys.stdout.flush(); exit(0)

    # --- Training Loop ---
    print(f"Start training SemCom with FIM for image reconstruction for {args.epochs} epochs")
    max_psnr_eval = 0.0; best_map_50_eval = 0.0
    start_time_total_train = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Starting Training Epoch {epoch}/{args.epochs-1} ---")
        # if args.distributed: trainloader.sampler.set_epoch(epoch) # For DDP

        semcom_model.train()
        train_stats = engine1.train_epoch_semcom_reconstruction(
            model=semcom_model, base_reconstruction_criterion=base_reconstruction_criterion,
            fim_criterion=fim_criterion, lpips_criterion=lpips_criterion_instance,
            data_loader=trainloader, optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, args=args, max_norm=args.clip_grad,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq, print_freq=args.save_freq # Adjusted print_freq to save_freq
        )
        epoch_duration = time.time() - epoch_start_time
        print(f"--- Epoch {epoch} Training Finished in {epoch_duration:.2f}s ---")
        print_train_stat_str = f"  Avg Train TotalL: {train_stats.get('total_loss',0):.3f} (" \
                               f"Rec: {train_stats.get('rec_loss',0):.3f}, " \
                               f"VQ: {train_stats.get('vq_loss',0):.3f}, " \
                               f"FIM: {train_stats.get('fim_loss',0):.3f}"
        if lpips_criterion_instance and args.lpips_loss_weight > 0: # Check if LPIPS was active
            print_train_stat_str += f", LPIPS: {train_stats.get('lpips_loss',0):.3f}"
        print_train_stat_str += f") PSNR: {train_stats.get('psnr',0):.2f}, SSIM: {train_stats.get('ssim',0):.3f}"
        print(print_train_stat_str)

        eval_stats_exist_this_epoch = False # Flag to check if eval_stats were computed
        current_eval_stats = {}

        # Perform evaluation periodically
        if dataloader_val is not None and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            print(f"\n--- Evaluating at end of Epoch {epoch} (Eval SNR: {args.snr_db_eval} dB) ---")
            current_epoch_viz_dir_path = Path(args.eval_viz_output_dir) / f"epoch_{epoch}_snr{args.snr_db_eval:.0f}"
            current_epoch_viz_dir_path.mkdir(parents=True, exist_ok=True)
            
            current_eval_stats = engine1.evaluate_semcom_with_yolo( # Store eval_stats
                semcom_net=semcom_model, yolo_model=yolo_model, dataloader=dataloader_val,
                device=device, reconstruction_criterion=base_reconstruction_criterion,
                fim_criterion=fim_criterion, lpips_criterion=lpips_criterion_instance, args=args,
                current_epoch_num=epoch, viz_output_dir=str(current_epoch_viz_dir_path)
            )
            eval_stats_exist_this_epoch = True # Mark that eval_stats are available
            
            current_psnr_value = current_eval_stats.get('psnr', -float('inf')) # Use -inf for correct comparison
            current_map_50_value = current_eval_stats.get('map_50', -float('inf')) # Use -inf

            print_eval_stat_str = f"  Epoch {epoch} Eval: PSNR: {current_psnr_value:.2f}, SSIM: {current_eval_stats.get('ssim',0.0):.4f}"
            if lpips_criterion_instance and hasattr(args, 'lpips_loss_weight') and args.lpips_loss_weight > 0:
                print_eval_stat_str += f", LPIPS: {current_eval_stats.get('lpips_loss',0.0):.4f}"
            if yolo_model and engine1.TORCHMETRICS_AVAILABLE:
                print_eval_stat_str += f", mAP: {current_eval_stats.get('map',0.0):.4f}, mAP@50: {current_map_50_value:.4f}"
            print(print_eval_stat_str)

            if current_psnr_value > max_psnr_eval:
                max_psnr_eval = current_psnr_value
                print(f"  *** New best PSNR on eval: {max_psnr_eval:.2f} at epoch {epoch} ***")
                if args.output_dir and args.save_ckpt:
                    utils.save_model(args=args, model=semcom_model, model_without_ddp=model_without_ddp,
                               optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_psnr")
            
            if yolo_model and engine1.TORCHMETRICS_AVAILABLE and current_map_50_value > best_map_50_eval:
                best_map_50_eval = current_map_50_value
                print(f"  *** New best mAP@50 on eval: {best_map_50_eval:.4f} at epoch {epoch} ***")
                if args.output_dir and args.save_ckpt:
                     utils.save_model(args=args, model=semcom_model, model_without_ddp=model_without_ddp,
                                 optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_map")
            print("-------------------------------------------------------------------\n")
            sys.stdout.flush()

        # Save regular checkpoint (after potential best model saves)
        if args.output_dir and args.save_ckpt and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            utils.save_model( # This call was correctly here for numbered checkpoints
                args=args, model=semcom_model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )


    total_training_duration = time.time() - start_time_total_train
    total_time_str = str(datetime.timedelta(seconds=int(total_training_duration)))
    print(f'Total Training time for SemCom: {total_time_str}')
    print(f'Best PSNR achieved on validation: {max_psnr_eval:.2f} dB')
    print(f'Best mAP@50 achieved on validation: {best_map_50_eval:.4f}')
    sys.stdout.flush()

if __name__ == '__main__':
    opts = get_args()
    if not hasattr(opts, 'lpips_loss_weight'):
        print("Warning: --lpips_loss_weight not defined in args. Defaulting to 0.0 (LPIPS disabled).")
        opts.lpips_loss_weight = 0.0
    if not hasattr(opts, 'save_ckpt'): # Ensure save_ckpt exists, default to False if not set by argparse
        print("Warning: --save_ckpt not explicitly defined in args. Defaulting to False (no checkpoints). Add --save_ckpt to command or base_args.py to enable.")
        opts.save_ckpt = False


    if opts.output_dir: Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(opts, 'eval_viz_output_dir') and opts.eval_viz_output_dir :
        Path(opts.eval_viz_output_dir).mkdir(parents=True, exist_ok=True)
    
    main(opts)
    print("--- Script execution finished. ---")