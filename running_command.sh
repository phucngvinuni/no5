#!/bin/bash

# This sets the GPU to be used (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# This is the command to run your Python training script
python3 run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS \
    --data_set fish \
    --data_path "" \
    --num_object_classes 1 \
    --yolo_weights "best.pt" \
    --batch_size 8 \
    --input_size 224 \
    --patch_size 8 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 20 \
    --epochs 300 \
    --opt adamw \
    --weight_decay 0.05 \
    --clip_grad 1.0 \
    --save_freq 5 \
    --save_ckpt \
    --mask_ratio 0.0 \
    --snr_db_train_min 20 \
    --snr_db_train_max 25 \
    --snr_db_eval 15 \
    --num_workers 0 \
    --pin_mem \
    --fim_routing_threshold 0.6 \
    --fim_loss_weight 0.5 \
    --vq_loss_weight 0.15 \
    --lpips_loss_weight 0 \
    --yolo_conf_thres 0.4 \
    --yolo_iou_thres 0.5 \
    --resume "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS/checkpoint-89.pth" \
    --eval \
