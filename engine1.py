# engine1.py
import torch
import math
import torch.nn as nn
import sys
import numpy as np
from typing import Iterable, Dict, Tuple, List, Optional
import os

from timm.utils import AverageMeter # Assuming this is correctly in your environment
import utils # Assuming utils.py contains create_bbox_weight_map, calc_psnr, calc_ssim
from torchvision.utils import save_image
# from torchvision import transforms # Not directly used here, but good practice if transformations are needed

# Attempt to import LPIPS, CV2 and torchmetrics, set flags accordingly
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    # This print happens at import time. Consider moving to main if you want it once per run.
    # print("Warning: lpips library not found. Perceptual loss will be skipped. `pip install lpips`")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # print("Warning: OpenCV (cv2) not found. YOLO detection visualizations will not be saved. `pip install opencv-python`")

try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    # print("Warning: torchmetrics not found. mAP calculation will be skipped. `pip install torchmetrics`")


# --- EVALUATE FUNCTION ---
@torch.no_grad()
def evaluate_semcom_with_yolo(
    semcom_net: torch.nn.Module,
    yolo_model: Optional[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    reconstruction_criterion: torch.nn.Module, # Base L1 reduction='none' for weighted loss calc
    fim_criterion: Optional[torch.nn.Module], # For optional FIM loss logging
    lpips_criterion: Optional[nn.Module], # For optional LPIPS logging
    args,
    current_epoch_num: any,
    viz_output_dir: Optional[str] = None,
    print_freq=20,
    visualize_batches=1,
    visualize_images_per_batch=1
):
    semcom_net.eval()
    if yolo_model: yolo_model.eval()

    # Meters
    main_rec_loss_meter_eval = AverageMeter()
    vq_loss_meter_eval = AverageMeter()
    fim_loss_meter_eval = AverageMeter()
    lpips_loss_meter_eval = AverageMeter() # For LPIPS
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    map_calculator = None
    if yolo_model and TORCHMETRICS_AVAILABLE:
        try:
            # Remove respect_labels=True
            map_calculator = MeanAveragePrecision(iou_type="bbox", class_metrics=True) # <--- MODIFIED HERE
        except Exception as e_map:
            print(f"Error initializing MeanAveragePrecision: {e_map}. mAP calculation will be skipped.")
            map_calculator = None
        if map_calculator: map_calculator = map_calculator.to(device)
    
    if viz_output_dir: os.makedirs(viz_output_dir, exist_ok=True)
    print(f"\n--- Starting Evaluation (Epoch/Run: {current_epoch_num}, Eval SNR: {args.snr_db_eval:.1f} dB) ---")

    for batch_idx, (semcom_data_input, targets_tuple) in enumerate(dataloader):
        (original_imgs_tensor, bm_pos) = semcom_data_input
        (semcom_recon_target_gt, yolo_gt_targets_list_of_dicts, fim_target_map_batch_eval) = targets_tuple

        original_imgs_tensor = original_imgs_tensor.to(device, non_blocking=True)
        if bm_pos is not None: bm_pos = bm_pos.to(device, non_blocking=True)
        semcom_recon_target_gt = semcom_recon_target_gt.to(device, non_blocking=True)
        fim_target_map_batch_eval = fim_target_map_batch_eval.to(device, non_blocking=True)

        yolo_gt_for_metric_device = []
        if yolo_model and map_calculator:
            for gt_dict in yolo_gt_targets_list_of_dicts:
                yolo_gt_for_metric_device.append({
                    "boxes": gt_dict["boxes"].to(device), "labels": gt_dict["labels"].to(device)
                })
        
        outputs_dict = semcom_net(
            img=original_imgs_tensor, bm_pos=bm_pos, _eval=True, # Removed targets=None
            eval_snr_db=args.snr_db_eval
        )
        reconstructed_image_batch = outputs_dict['reconstructed_image']
        
        batch_size_eval = reconstructed_image_batch.size(0)
        current_device_eval = reconstructed_image_batch.device
        total_weighted_rec_loss_eval = torch.tensor(0.0, device=current_device_eval)
        for i in range(batch_size_eval):
            rec_img_single = reconstructed_image_batch[i]
            orig_img_single = semcom_recon_target_gt[i]
            gt_boxes_single = yolo_gt_targets_list_of_dicts[i]['boxes']
            
            weight_map = utils.create_bbox_weight_map(
                rec_img_single.shape[1:], gt_boxes_single.to(current_device_eval),
                args.inside_box_loss_weight, args.outside_box_loss_weight, current_device_eval
            )
            pixel_loss = reconstruction_criterion(rec_img_single, orig_img_single)
            weighted_loss = (pixel_loss * weight_map.unsqueeze(0)).mean()
            total_weighted_rec_loss_eval += weighted_loss
        
        avg_weighted_rec_loss_eval = total_weighted_rec_loss_eval / batch_size_eval if batch_size_eval > 0 else torch.tensor(0.0)
        main_rec_loss_meter_eval.update(avg_weighted_rec_loss_eval.item(), batch_size_eval)

        current_eval_vq_loss = outputs_dict.get('vq_loss', torch.tensor(0.0)).item()
        vq_loss_meter_eval.update(current_eval_vq_loss, batch_size_eval)
        
        current_eval_fim_loss = 0.0
        if fim_criterion and 'fim_importance_scores' in outputs_dict:
            fim_preds_eval_logits = outputs_dict['fim_importance_scores']
            # Ensure target map matches the number of patches FIM made predictions for
            # This logic depends on whether bm_pos (encoder_mask) was applied before FIM input
            if args.mask_ratio == 0.0: # FIM saw all patches
                target_fim_map = fim_target_map_batch_eval
            else: # FIM saw only visible patches
                if bm_pos is not None:
                    flat_fim_targets = fim_target_map_batch_eval.reshape(-1, 1)
                    flat_bm_pos_encoder_mask = bm_pos.reshape(-1)
                    target_fim_map = flat_fim_targets[~flat_bm_pos_encoder_mask].reshape(batch_size_eval, -1, 1)
                else: # Should not happen if mask_ratio > 0 and bm_pos is None
                    target_fim_map = fim_target_map_batch_eval 

            if fim_preds_eval_logits.shape == target_fim_map.shape:
                 with torch.amp.autocast(device_type=args.device, enabled=False): # Calculate FIM loss in FP32 for stability
                    fim_loss_eval = fim_criterion(fim_preds_eval_logits.float(), target_fim_map.to(fim_preds_eval_logits.dtype).float())
                    current_eval_fim_loss = fim_loss_eval.item()
            # else:
            #     print(f"Eval FIM Shape Mismatch: Pred {fim_preds_eval_logits.shape}, Target {target_fim_map.shape}")

        fim_loss_meter_eval.update(current_eval_fim_loss, batch_size_eval)

        # LPIPS Loss (optional for eval logging)
        current_eval_lpips_loss = 0.0
        if lpips_criterion and hasattr(args, 'lpips_loss_weight') and args.lpips_loss_weight > 0:
            with torch.amp.autocast(device_type=args.device, enabled=False): # LPIPS often prefers FP32
                recon_for_lpips_eval = (reconstructed_image_batch.float() * 2.0) - 1.0
                orig_for_lpips_eval = (semcom_recon_target_gt.float() * 2.0) - 1.0
                lpips_val_eval = lpips_criterion(recon_for_lpips_eval, orig_for_lpips_eval).mean()
                current_eval_lpips_loss = lpips_val_eval.item()
        lpips_loss_meter_eval.update(current_eval_lpips_loss, batch_size_eval)


        batch_psnr = utils.calc_psnr(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        batch_ssim = utils.calc_ssim(reconstructed_image_batch.detach().cpu(), semcom_recon_target_gt.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr) if batch_psnr else 0.0, batch_size_eval)
        ssim_meter.update(np.mean(batch_ssim) if batch_ssim else 0.0, batch_size_eval)

        if batch_idx < visualize_batches and viz_output_dir:
            # ... (visualization code as before) ...
            for i in range(min(batch_size_eval, visualize_images_per_batch)):
                save_image(reconstructed_image_batch[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_snr{args.snr_db_eval:.0f}dB_RECON.png"))
                save_image(original_imgs_tensor[i].cpu(), os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_ORIG.png"))
                if 'fim_importance_scores' in outputs_dict and CV2_AVAILABLE:
                    try:
                        fim_logits_viz = outputs_dict['fim_importance_scores'][i].squeeze().detach()
                        fim_scores_viz = torch.sigmoid(fim_logits_viz).cpu().numpy() # Apply sigmoid for visualization
                        num_h_patches = args.input_size // args.patch_size
                        num_w_patches = args.input_size // args.patch_size
                        if fim_scores_viz.size == num_h_patches * num_w_patches :
                            fim_map_2d_viz = fim_scores_viz.reshape(num_h_patches, num_w_patches)
                            fim_map_resized_viz = cv2.resize((fim_map_2d_viz * 255).astype(np.uint8), (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_FIM_PRED_MAP.png"), fim_map_resized_viz)
                        
                        # Visualize FIM Target Map
                        target_fim_map_viz_item = fim_target_map_batch_eval[i].squeeze().cpu().numpy()
                        if target_fim_map_viz_item.size == num_h_patches * num_w_patches:
                            target_fim_map_2d_viz = target_fim_map_viz_item.reshape(num_h_patches, num_w_patches)
                            target_fim_map_resized_viz = cv2.resize((target_fim_map_2d_viz * 255).astype(np.uint8), (args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i}_FIM_TARGET_MAP.png"), target_fim_map_resized_viz)
                    except Exception as e_viz_fim: print(f"Error during FIM map visualization: {e_viz_fim}")


        if yolo_model and map_calculator:
            if batch_idx < visualize_batches and viz_output_dir and CV2_AVAILABLE:
                # ... (YOLO on Original visualization code as before) ...
                try:
                    yolo_input_orig_eval = original_imgs_tensor.detach()
                    yolo_results_orig_eval = yolo_model(yolo_input_orig_eval, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                    for i_vo in range(min(len(yolo_results_orig_eval), visualize_images_per_batch)):
                        if hasattr(yolo_results_orig_eval[i_vo], 'plot'):
                            cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i_vo}_YOLO_ON_ORIG.png"), yolo_results_orig_eval[i_vo].plot())
                except Exception as e_yolo_orig: print(f"    Eval Viz Error (YOLO on Original): {e_yolo_orig}")
            
            yolo_preds_for_metric = []
            try:
                yolo_input_recon_eval = reconstructed_image_batch.detach() # Already [0,1] range
                # Verify YOLO input normalization expectations (if best.pt was trained with specific norm)
                # If needed: yolo_input_recon_eval = normalize_for_yolo(yolo_input_recon_eval)
                yolo_results_list_recon_eval = yolo_model(yolo_input_recon_eval, verbose=False, conf=args.yolo_conf_thres, iou=args.yolo_iou_thres)
                for i_vr, result_r_eval in enumerate(yolo_results_list_recon_eval):
                    if result_r_eval.boxes is not None and len(result_r_eval.boxes.xyxy) > 0:
                        yolo_preds_for_metric.append({
                            "boxes": result_r_eval.boxes.xyxy.to(device), "scores": result_r_eval.boxes.conf.to(device), "labels": result_r_eval.boxes.cls.to(torch.int64).to(device)
                        })
                        if batch_idx < visualize_batches and i_vr < visualize_images_per_batch and viz_output_dir and CV2_AVAILABLE:
                            if hasattr(result_r_eval, 'plot'):
                                cv2.imwrite(os.path.join(viz_output_dir, f"ep{current_epoch_num}_b{batch_idx}_i{i_vr}_snr{args.snr_db_eval:.0f}dB_YOLO_ON_RECON.png"), result_r_eval.plot())
                    else:
                        yolo_preds_for_metric.append({"boxes": torch.empty((0,4), dtype=torch.float32, device=device),"scores": torch.empty(0, dtype=torch.float32, device=device),"labels": torch.empty(0, dtype=torch.int64, device=device)})
            except Exception as e_yolo_recon:
                print(f"    Error during YOLO inference on reconstructed for mAP: {e_yolo_recon}")
                yolo_preds_for_metric = [{"boxes": torch.empty((0,4), dtype=torch.float32, device=device),"scores": torch.empty(0, dtype=torch.float32, device=device),"labels": torch.empty(0, dtype=torch.int64, device=device)} for _ in range(batch_size_eval)]

            if yolo_preds_for_metric and len(yolo_preds_for_metric) == len(yolo_gt_for_metric_device):
                try: map_calculator.update(yolo_preds_for_metric, yolo_gt_for_metric_device)
                except Exception as e_map_update: print(f"    Error updating mAP calculator: {e_map_update}")
            elif yolo_preds_for_metric : print(f"    Warning: Mismatch num YOLO preds ({len(yolo_preds_for_metric)}) vs GTs ({len(yolo_gt_for_metric_device)}). Skipping mAP update batch {batch_idx}.")

        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader):
            print_str = f'Eval Batch {batch_idx+1}/{len(dataloader)}: ' \
                        f'[RecL: {main_rec_loss_meter_eval.avg:.3f}] [VQL: {vq_loss_meter_eval.avg:.3f}] ' \
                        f'[FIM_L: {fim_loss_meter_eval.avg:.3f}] '
            if lpips_criterion: print_str += f'[LPIPS: {lpips_loss_meter_eval.avg:.3f}] '
            print_str += f'[PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.3f}]'
            print(print_str)
            sys.stdout.flush()

    final_stats = {
        'rec_loss': main_rec_loss_meter_eval.avg, 'vq_loss': vq_loss_meter_eval.avg,
        'fim_loss': fim_loss_meter_eval.avg, 'lpips_loss': lpips_loss_meter_eval.avg,
        'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg,
    }
    if map_calculator:
        # ... (final mAP calculation and logging as before) ...
        try:
            final_map_results = map_calculator.compute()
            print(f"\nFinal mAP Results (Eval SNR: {args.snr_db_eval:.1f} dB, Epoch/Run: {current_epoch_num}):")
            for k, v_tensor in final_map_results.items():
                v_item = v_tensor.item() if isinstance(v_tensor, torch.Tensor) else float(v_tensor)
                print(f"  {k}: {v_item:.4f}")
                safe_key = str(k).replace("map_per_class","map_cls").replace("(","_").replace(")","").replace("[","").replace("]","").replace(" ","_").replace("'","").replace(":","_")
                final_stats[safe_key] = v_item
            map_calculator.reset()
        except Exception as e: print(f"Error computing final mAP: {e}"); final_stats['map'] = 0.0
            
    print("--- Finished Evaluation ---")
    sys.stdout.flush()
    return final_stats


def train_semcom_reconstruction_batch(
    model: torch.nn.Module,
    input_samples_for_semcom: torch.Tensor,
    original_images_for_loss: torch.Tensor,
    yolo_gt_for_this_batch: List[Dict[str, torch.Tensor]],
    fim_target_importance_map_batch: torch.Tensor,
    bm_pos: Optional[torch.Tensor],
    base_reconstruction_criterion: torch.nn.Module,
    fim_criterion: torch.nn.Module,
    lpips_criterion: Optional[nn.Module], # Added LPIPS criterion
    args
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, float]: # Added LPIPS loss value
    outputs_dict = model(
        img=input_samples_for_semcom, bm_pos=bm_pos, _eval=False,
        train_snr_db_min=args.snr_db_train_min, train_snr_db_max=args.snr_db_train_max
    )
    reconstructed_image_batch = outputs_dict['reconstructed_image']
    fim_predicted_logits_batch = outputs_dict.get('fim_importance_scores')

    batch_size = reconstructed_image_batch.size(0)
    current_device = reconstructed_image_batch.device
    
    # 1. Bounding Box Weighted Reconstruction Loss
    total_weighted_rec_loss_tensor = torch.tensor(0.0, device=current_device)
    for i in range(batch_size):
        reconstructed_img_single = reconstructed_image_batch[i]
        original_img_single = original_images_for_loss[i]
        gt_boxes_single_img_abs = yolo_gt_for_this_batch[i]['boxes']
        weight_map_2d = utils.create_bbox_weight_map(
            reconstructed_img_single.shape[1:], gt_boxes_single_img_abs,
            args.inside_box_loss_weight, args.outside_box_loss_weight, current_device
        )
        pixel_wise_loss_single_image = base_reconstruction_criterion(reconstructed_img_single, original_img_single)
        weighted_pixel_loss_single_image = (pixel_wise_loss_single_image * weight_map_2d.unsqueeze(0)).mean()
        total_weighted_rec_loss_tensor += weighted_pixel_loss_single_image
    final_reconstruction_loss = total_weighted_rec_loss_tensor / batch_size if batch_size > 0 else torch.tensor(0.0, device=current_device)
    current_rec_loss_val = final_reconstruction_loss.item()

    # 2. FIM-Weighted VQ Loss
    vq_loss_component = torch.tensor(0.0, device=current_device)
    current_vq_loss_val = 0.0
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        vq_loss_tensor = outputs_dict['vq_loss'] # This is self.current_vq_loss from the model
        if isinstance(vq_loss_tensor, torch.Tensor):
            # This vq_loss_tensor is already the mean FIM-weighted loss from both quantizers
            vq_loss_component = args.vq_loss_weight * vq_loss_tensor
            current_vq_loss_val = vq_loss_tensor.item() # Log this value

    # 3. FIM Training Loss
    fim_loss_component = torch.tensor(0.0, device=current_device)
    current_fim_loss_val = 0.0
    if fim_predicted_logits_batch is not None:
        target_fim_map_for_loss = fim_target_importance_map_batch # Assuming mask_ratio=0 for now
        if args.mask_ratio > 0 and bm_pos is not None:
             flat_fim_targets = fim_target_importance_map_batch.reshape(-1, 1)
             flat_bm_pos_encoder_mask = bm_pos.reshape(-1)
             target_fim_map_for_loss = flat_fim_targets[~flat_bm_pos_encoder_mask].reshape(batch_size, -1, 1)

        if fim_predicted_logits_batch.shape == target_fim_map_for_loss.shape:
            fim_loss = fim_criterion(fim_predicted_logits_batch, target_fim_map_for_loss.to(fim_predicted_logits_batch.dtype))
            fim_loss_component = args.fim_loss_weight * fim_loss
            current_fim_loss_val = fim_loss.item()
        # else:
            # print(f"Train FIM Shape Mismatch: Pred {fim_predicted_logits_batch.shape}, Target {target_fim_map_for_loss.shape}")


    # 4. LPIPS Perceptual Loss
    perceptual_loss_component = torch.tensor(0.0, device=current_device)
    current_lpips_loss_val = 0.0
    if lpips_criterion is not None and args.lpips_loss_weight > 0:
        # LPIPS expects images in range [-1, 1]
        recon_for_lpips = (reconstructed_image_batch * 2.0) - 1.0
        orig_for_lpips = (original_images_for_loss * 2.0) - 1.0
        # Disable autocast for LPIPS if it's sensitive, or ensure it works with current precision
        with torch.amp.autocast(device_type=args.device, enabled=False): # Often LPIPS is better in FP32
             lpips_val = lpips_criterion(recon_for_lpips.float(), orig_for_lpips.float()).mean()
        perceptual_loss_component = args.lpips_loss_weight * lpips_val
        current_lpips_loss_val = lpips_val.item()
            
    total_loss = final_reconstruction_loss + vq_loss_component + fim_loss_component + perceptual_loss_component
            
    return total_loss, reconstructed_image_batch, current_vq_loss_val, current_rec_loss_val, current_fim_loss_val, current_lpips_loss_val


def train_epoch_semcom_reconstruction(
    model: torch.nn.Module,
    base_reconstruction_criterion: torch.nn.Module,
    fim_criterion: torch.nn.Module,
    lpips_criterion: Optional[nn.Module], # Added LPIPS
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: Optional[torch.cuda.amp.GradScaler], 
    args,
    max_norm: Optional[float] = None,           
    start_steps: int = 0,
    lr_schedule_values: Optional[np.ndarray] = None,
    wd_schedule_values: Optional[np.ndarray] = None,
    update_freq: int = 1,
    print_freq: int = 50
):
    model.train(True)
    
    main_rec_loss_meter = AverageMeter()
    vq_loss_meter_train = AverageMeter()
    fim_loss_meter_train = AverageMeter()
    lpips_loss_meter_train = AverageMeter() # LPIPS meter
    total_loss_meter = AverageMeter()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    if update_freq > 1 : optimizer.zero_grad()
    print_prefix = f'Ep:[{epoch}]'

    for data_iter_step, (semcom_data_input, targets_tuple) in enumerate(data_loader):
        effective_iter = start_steps + (data_iter_step // update_freq)
        if (data_iter_step % update_freq == 0):
            if lr_schedule_values is not None and effective_iter < len(lr_schedule_values):
                for param_group in optimizer.param_groups: param_group["lr"] = lr_schedule_values[effective_iter] * param_group.get("lr_scale", 1.0)
            if wd_schedule_values is not None and effective_iter < len(wd_schedule_values) and args.weight_decay > 0:
                for param_group in optimizer.param_groups:
                    if param_group.get("weight_decay", 0.0) > 0: param_group["weight_decay"] = wd_schedule_values[effective_iter]

        (original_images, bm_pos) = semcom_data_input
        (original_images_for_loss, yolo_gt_targets_list, fim_target_map_batch) = targets_tuple
        
        original_images = original_images.to(device, non_blocking=True)
        original_images_for_loss = original_images_for_loss.to(device, non_blocking=True)
        if bm_pos is not None: bm_pos = bm_pos.to(device, non_blocking=True)
        fim_target_map_batch = fim_target_map_batch.to(device, non_blocking=True)
        
        yolo_gt_on_device = [{"boxes": gt_dict["boxes"].to(device), "labels": gt_dict["labels"].to(device)} for gt_dict in yolo_gt_targets_list]
        samples_for_semcom_input = original_images
        
        with torch.amp.autocast(device_type=args.device, enabled=(loss_scaler is not None)):
            total_loss, reconstructed_batch, vq_loss_val, rec_loss_val, fim_loss_val, lpips_loss_val = train_semcom_reconstruction_batch(
                model=model, input_samples_for_semcom=samples_for_semcom_input,
                original_images_for_loss=original_images_for_loss, yolo_gt_for_this_batch=yolo_gt_on_device,
                fim_target_importance_map_batch=fim_target_map_batch, bm_pos=bm_pos,
                base_reconstruction_criterion=base_reconstruction_criterion,
                fim_criterion=fim_criterion, lpips_criterion=lpips_criterion, args=args # Pass lpips_criterion
            )
        
        total_loss_value_item = total_loss.item()
        if not math.isfinite(total_loss_value_item): print(f"{print_prefix} Total Loss is {total_loss_value_item}, stopping."); sys.exit(1)

        loss_for_backward = total_loss / update_freq
        clip_grad_val_for_scaler = max_norm if max_norm is not None and max_norm > 0 else None
        if loss_scaler is not None:
            loss_scaler(loss_for_backward, optimizer, parameters=model.parameters(), clip_grad=clip_grad_val_for_scaler, update_grad=((data_iter_step + 1) % update_freq == 0))
            if ((data_iter_step + 1) % update_freq == 0) and update_freq > 1: optimizer.zero_grad()
        else:
            loss_for_backward.backward()
            if ((data_iter_step + 1) % update_freq == 0):
                if clip_grad_val_for_scaler: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_val_for_scaler)
                optimizer.step(); optimizer.zero_grad()

        if device == 'cuda': torch.cuda.synchronize()

        total_loss_meter.update(total_loss_value_item, original_images.size(0))
        main_rec_loss_meter.update(rec_loss_val, original_images.size(0))
        vq_loss_meter_train.update(vq_loss_val, original_images.size(0))
        fim_loss_meter_train.update(fim_loss_val, original_images.size(0))
        lpips_loss_meter_train.update(lpips_loss_val, original_images.size(0))
        
        batch_psnr_train = utils.calc_psnr(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        batch_ssim_train = utils.calc_ssim(reconstructed_batch.detach().cpu(), original_images_for_loss.detach().cpu())
        psnr_meter.update(np.mean(batch_psnr_train) if batch_psnr_train else 0.0, original_images.size(0))
        ssim_meter.update(np.mean(batch_ssim_train) if batch_ssim_train else 0.0, original_images.size(0))

        if (data_iter_step + 1) % print_freq == 0 or (data_iter_step + 1) == len(data_loader):
            lr = optimizer.param_groups[0]["lr"]
            print_str = f'{print_prefix} It:[{data_iter_step+1}/{len(data_loader)}] ' \
                        f'TotalL: {total_loss_meter.avg:.3f} (Rec: {main_rec_loss_meter.avg:.3f} ' \
                        f'VQ: {vq_loss_meter_train.avg:.3f} FIM: {fim_loss_meter_train.avg:.3f} '
            if lpips_criterion and args.lpips_loss_weight > 0: print_str += f'LPIPS: {lpips_loss_meter_train.avg:.3f} '
            print_str += f') PSNR: {psnr_meter.avg:.2f} SSIM: {ssim_meter.avg:.3f} LR: {lr:.2e}'
            print(print_str)
            sys.stdout.flush()

    train_stat = {
        'total_loss': total_loss_meter.avg, 'rec_loss': main_rec_loss_meter.avg,
        'vq_loss': vq_loss_meter_train.avg, 'fim_loss': fim_loss_meter_train.avg,
        'lpips_loss': lpips_loss_meter_train.avg,
        'psnr': psnr_meter.avg, 'ssim': ssim_meter.avg
    }
    print(f"--- Finished Training {print_prefix} ---")
    sys.stdout.flush()
    return train_stat