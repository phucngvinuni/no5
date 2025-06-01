# engine1.py
import numpy as np
import torch
import math
# import nltk # Not needed for reconstruction
import torch.nn as nn
import sys
import torch.nn.functional as F # For FIM loss if used

from utils import (
    AverageMeter, get_loss_scale_for_deepspeed,
    calc_psnr, calc_ssim # Add more metrics if needed
)
# from reg_attack import FastGradientSignUntargeted as FGSM_REG # Keep if you do adversarial training on reconstruction
# from timm.data import Mixup # Not typically used for reconstruction MAE style
# from einops import rearrange # If your decoder uses it
from typing import Iterable, Optional
# from timm.utils import accuracy # Not for reconstruction
# from nltk.translate.bleu_score import sentence_bleu # Not for reconstruction

beta = 1.0 # For FIM loss if used

# --- EVALUATE FUNCTION ---
@torch.no_grad()
def evaluate_reconstruction(net: torch.nn.Module, yolo_model: torch.nn.Module, # Pass YOLO model
                            dataloader: Iterable, device: torch.device,
                            reconstruction_criterion: torch.nn.Module,
                            args, # Pass args for configuration
                            print_freq=10):
    net.eval()
    if yolo_model: yolo_model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    map_meter = AverageMeter() # For mAP from YOLO

    # attack_module = None
    # if args.if_attack_test: # Setup attack if configured
    #     attack_module = FGSM_REG(net, epsilon=..., alpha=..., ...) # Configure properly

    for batch_idx, (data_input, targets_classification) in enumerate(dataloader): # Assume targets_classification might be for FIM or ignore
        original_imgs, bm_pos = data_input # bm_pos is the mask for the SemCom encoder
        original_imgs = original_imgs.to(device, non_blocking=True)
        bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool) if bm_pos is not None else None
        
        # targets_classification for FIM if used, or ground truth for YOLO
        # For YOLO, targets would be list of dicts with 'boxes' and 'labels'
        # This needs to come from your dataloader:
        # e.g. (data_input, (targets_classification, gt_yolo_targets)) = batch
        # For now, I'll assume gt_yolo_targets are loaded separately or part of `targets_classification`
        # if it's a multi-task dataset. This part is CRUCIAL for mAP.

        # --- Semantic Communication Part ---
        # For evaluation, you might want to test specific SNRs
        current_eval_snr = args.snr_db # Or iterate through a list of SNRs

        # Adversarial attack on input to SemCom if configured
        # if args.if_attack_test and attack_module:
        #    adv_input_imgs = attack_module.perturb(original_imgs, dummy_targets_for_attack, bm_pos_for_attack)
        #    semcom_input = adv_input_imgs
        # else:
        semcom_input = original_imgs

        outputs_dict = net(img=semcom_input, bm_pos=bm_pos, _eval=True, test_snr=current_eval_snr)
        reconstructed_image = outputs_dict['reconstructed_image']

        # Reconstruction Loss (e.g., MSE)
        rec_loss = reconstruction_criterion(reconstructed_image, original_imgs)
        if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
            rec_loss += outputs_dict['vq_loss']
        loss_meter.update(rec_loss.item(), original_imgs.size(0))

        # Reconstruction Metrics
        batch_psnr = calc_psnr(reconstructed_image.detach(), original_imgs.detach())
        batch_ssim = calc_ssim(reconstructed_image.detach(), original_imgs.detach())
        psnr_meter.update(np.mean(batch_psnr), original_imgs.size(0)) # Ensure it's a scalar average
        ssim_meter.update(np.mean(batch_ssim), original_imgs.size(0))

        # --- YOLOv11 Part ---
        if yolo_model:
            # Preprocess reconstructed_image if necessary for YOLO
            # yolo_input = preprocess_for_yolo(reconstructed_image.detach())

            yolo_predictions = yolo_model(reconstructed_image.detach()) # Feed reconstructed image

            # Post-process yolo_predictions (NMS, etc.) to get final boxes, scores, labels
            # final_preds = postprocess_yolo(yolo_predictions)

            # Calculate mAP
            # You need ground truth for the current batch: gt_boxes, gt_labels
            # mAP_val = calculate_map_for_batch(final_preds, gt_boxes_batch, gt_labels_batch, args.num_object_classes)
            # map_meter.update(mAP_val, original_imgs.size(0))
            # For now, placeholder for mAP
            pass


        if batch_idx % print_freq == 0:
            print(f'Test {batch_idx}/{len(dataloader)}: '
                  f'[Rec Loss: {loss_meter.avg:.4f}] '
                  f'[PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.4f}] '
                  f'[mAP: {map_meter.avg:.4f} (placeholder)] ' # Update with real mAP
                  f'[SNR: {current_eval_snr:.1f} dB]')

    test_stat = {
        'rec_loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'map': map_meter.avg # Will be 0 if not implemented
    }
    return test_stat

# --- TRAIN RECONSTRUCTION BATCH ---
def train_reconstruction_batch(model: torch.nn.Module,
                               input_samples: torch.Tensor, # Images that might be attacked
                               original_targets_for_loss: torch.Tensor, # Clean images for recon loss
                               bm_pos: torch.Tensor,
                               criterion: torch.nn.Module, # Reconstruction criterion (MSE/L1)
                               aux_classification_targets=None, # For FIM
                               train_type: str = 'std_train',
                               current_epoch_snr=10.0
                               ):
    # Forward pass through SemCom model
    # `targets` for model.forward() is for FIM's aux classifier, if used
    outputs_dict = model(img=input_samples, bm_pos=bm_pos, targets=aux_classification_targets, _eval=False, test_snr=current_epoch_snr)
    reconstructed_image = outputs_dict['reconstructed_image']

    # Main reconstruction loss
    loss = criterion(reconstructed_image, original_targets_for_loss)

    # Add VQ loss
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        loss += outputs_dict['vq_loss']

    # Add FIM auxiliary classification loss (if applicable and train_type is 'fim_train')
    if train_type.startswith('fim') and 'out_c' in outputs_dict and aux_classification_targets is not None:
        fim_loss = 0.
        num_fim_outputs = 0
        for extra_output in outputs_dict['out_c']:
            if extra_output is not None: # FIM_V1 might return None if dims mismatch
                fim_loss += F.cross_entropy(extra_output, aux_classification_targets)
                num_fim_outputs +=1
        if num_fim_outputs > 0:
            loss += beta * (fim_loss / num_fim_outputs)

    return loss, reconstructed_image

def train_semcom_reconstruction_batch(
    model: torch.nn.Module,
    input_samples_for_semcom: torch.Tensor,
    original_images_for_loss: torch.Tensor,
    bm_pos: torch.Tensor,
    reconstruction_criterion: torch.nn.Module,
    args # Pass full args to get SNR range
):
    # Forward pass through SemCom model
    outputs_dict = model(
        img=input_samples_for_semcom,
        bm_pos=bm_pos,
        _eval=False, # Training mode
        # Pass the SNR range for training to the model's forward method
        train_snr_db_min=args.snr_db_train_min,
        train_snr_db_max=args.snr_db_train_max
        # eval_snr_db is not needed here as _eval=False
    )
    reconstructed_image = outputs_dict['reconstructed_image']
    
    # Main reconstruction loss
    loss = reconstruction_criterion(reconstructed_image, original_images_for_loss)

    current_vq_loss = 0.0
    if 'vq_loss' in outputs_dict and outputs_dict['vq_loss'] is not None:
        loss += outputs_dict['vq_loss']
        current_vq_loss = outputs_dict['vq_loss'].item()
    
    # We need to know the actual SNR used for logging if desired, model.forward() could return it
    # For now, just return loss and reconstruction
    return loss, reconstructed_image, current_vq_loss
# --- TRAIN EPOCH ---
def train_epoch_reconstruction(model: torch.nn.Module, criterion: torch.nn.Module, # Reconstruction criterion
                               data_loader: Iterable, optimizer: torch.optim.Optimizer,
                               device: torch.device, epoch: int, loss_scaler,
                               args, # Pass full args
                               max_norm: float = 0,
                               start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                               update_freq=None, print_freq=50):
    model.train(True)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter() # Track reconstruction PSNR during training
    ssim_meter = AverageMeter()

    # attack_module = None
    # if args.if_attack_train:
    #     attack_module = FGSM_REG(model, epsilon=8./255., alpha=2./255., ...) # Configure attack for reconstruction

    if loss_scaler is None: # Should not happen with NativeScaler
        model.zero_grad()
    else:
        optimizer.zero_grad()

    for data_iter_step, (data_input, targets_classification) in enumerate(data_loader): # targets_classification for FIM
        step = data_iter_step // update_freq
        it = start_steps + step

        # LR and WD scheduling (ensure it doesn't go out of bounds)
        if lr_schedule_values is not None and it < len(lr_schedule_values):
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        if wd_schedule_values is not None and it < len(wd_schedule_values):
             for i, param_group in enumerate(optimizer.param_groups):
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        original_images, bm_pos = data_input
        original_images = original_images.to(device, non_blocking=True)
        samples_for_semcom = original_images.clone() # Start with clean images
        
        if bm_pos is not None:
            bm_pos = bm_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # Prepare targets for FIM if used (assuming targets_classification are image-level labels)
        aux_cls_targets = targets_classification.to(device, non_blocking=True) if args.train_type.startswith('fim') else None

        # Adversarial attack on input to SemCom
        # if args.if_attack_train and attack_module:
        #     # The attack needs to maximize reconstruction loss or a proxy
        #     # This requires attack_module to be adapted for reconstruction loss.
        #     # For now, let's assume it perturbs the input image directly.
        #     # `dummy_targets_for_attack` would be `original_images` if attack maximizes MSE.
        #     samples_for_semcom = attack_module.perturb(samples_for_semcom, original_images, bm_pos_for_attack)

        # Dynamic SNR for training (as in your original model.ViT_FIM_CLS.forward)
        current_epoch_snr = (torch.rand(1).item() * 25) - 5.0 # SNR from -5 to 20 dB

        with torch.cuda.amp.autocast(enabled=True if loss_scaler else False): # Enable AMP
            loss, reconstructed_batch = train_reconstruction_batch(
                model, samples_for_semcom, original_images, bm_pos, criterion,
                aux_classification_targets=aux_cls_targets,
                train_type=args.train_type,
                current_epoch_snr=current_epoch_snr
            )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler:
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), #create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
        else: # No AMP
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()


        torch.cuda.synchronize()

        loss_meter.update(loss_value, original_images.size(0))
        batch_psnr_train = calc_psnr(reconstructed_batch.detach(), original_images.detach())
        batch_ssim_train = calc_ssim(reconstructed_batch.detach(), original_images.detach())
        psnr_meter.update(np.mean(batch_psnr_train), original_images.size(0))
        ssim_meter.update(np.mean(batch_ssim_train), original_images.size(0))


        if data_iter_step % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f'Epoch:[{epoch}] {data_iter_step}/{len(data_loader)}: '
                  f'[Loss: {loss_meter.avg:.4f}] [PSNR: {psnr_meter.avg:.2f}] [SSIM: {ssim_meter.avg:.4f}] '
                  f'[LR: {lr:.3e}] [Train SNR ~: {current_epoch_snr:.1f} dB]')

    train_stat = {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
    }
    return train_stat

# train_epoch_wp can be adapted similarly if you want to do weight perturbation for reconstruction