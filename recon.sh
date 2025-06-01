#!/bin/bash

echo "Starting reconstruction script..."

# This sets the GPU to be used (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# --- DEFINE YOUR PATHS AND PARAMETERS HERE ---
PROJECT_ROOT="/mnt/c/Users/ADMIN/Downloads/siso" # Your project root
SEMCOM_CHECKPOINT_NAME="checkpoint-299.pth" # <<<---- USE YOUR BEST CHECKPOINT
SEMCOM_CHECKPOINT_FOLDER="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS" # <<<---- FOLDER OF YOUR OLD (NON-MIMO) CHECKPOINTS
ORIGINAL_DATA_ROOT="" # Path to parent of train/valid/test
RECON_DATA_ROOT="../reconstructed_yolo_train_data_SNR22_SISO" # Output for reconstructed data
SPLIT_TO_RECONSTRUCT="train"
SNR_FOR_RECONSTRUCTION="10 15 13 22.0"

# --- ARCHITECTURAL PARAMETERS OF THE MODEL IN THE CHECKPOINT ---
# !!! THESE MUST EXACTLY MATCH THE MODEL SAVED IN THE CHECKPOINT !!!
# These should match the "No FIM, weighted loss" run that got ~0.28 mAP, or your best LPIPS run.
MODEL_NAME="ViT_Reconstruction_Model_Default" # Or whatever model name produced the checkpoint
INPUT_SIZE=224
PATCH_SIZE=8
ENC_EMBED_DIM=512    # Verify this from the checkpoint's training args
ENC_DEPTH=8          # Verify
ENC_HEADS=8          # Verify
DEC_EMBED_DIM=512    # Verify
DEC_DEPTH=6          # Verify
DEC_HEADS=8          # Verify
QUANTIZER_DIM=512    # Verify
BITS_VQ_HIGH=16      # Or single bits_for_quantizer if it was a single VQ model
BITS_VQ_LOW=4        # Or single bits_for_quantizer
FIM_EMBED_DIM=256    # If checkpoint model used FIM, set these. If not, they might be ignored or cause issues if not default.
FIM_DEPTH=2          # If no FIM in checkpoint, these FIM args might not be needed by model init
FIM_HEADS=4
FIM_ROUTING_THRESHOLD=0.7
DROP_RATE=0.0
DROP_PATH_RATE=0.1

# # --- CHANNEL PARAMETERS FOR RECONSTRUCTION ---
# CHANNEL_TYPE_RECON="rayleigh" # Scalar channel since you removed MIMO from this run

# --- END USER-DEFINED PARAMETERS ---

python3 "${PROJECT_ROOT}/reconstruct_dataset.py" \
    --semcom_checkpoint_path "${PROJECT_ROOT}/${SEMCOM_CHECKPOINT_FOLDER}/${SEMCOM_CHECKPOINT_NAME}" \
    --original_dataset_root "${ORIGINAL_DATA_ROOT}" \
    --split_to_reconstruct "${SPLIT_TO_RECONSTRUCT}" \
    --reconstructed_dataset_root "${RECON_DATA_ROOT}" \
    \
    --model "${MODEL_NAME}" \
    --input_size ${INPUT_SIZE} \
    --patch_size ${PATCH_SIZE} \
    --encoder_embed_dim ${ENC_EMBED_DIM} \
    --encoder_depth ${ENC_DEPTH} \
    --encoder_num_heads ${ENC_HEADS} \
    --decoder_embed_dim ${DEC_EMBED_DIM} \
    --decoder_depth ${DEC_DEPTH} \
    --decoder_num_heads ${DEC_HEADS} \
    --quantizer_dim ${QUANTIZER_DIM} \
    --bits_vq_high ${BITS_VQ_HIGH} \
    --bits_vq_low ${BITS_VQ_LOW} \
    --fim_embed_dim ${FIM_EMBED_DIM} \
    --fim_depth ${FIM_DEPTH} \
    --fim_num_heads ${FIM_HEADS} \
    --fim_routing_threshold ${FIM_ROUTING_THRESHOLD} \
    --drop_rate ${DROP_RATE} \
    --drop_path_rate ${DROP_PATH_RATE} \
    --snr_list_for_reconstruction ${SNR_FOR_RECONSTRUCTION} \
    --device cuda \
    --batch_size_recon 16

echo "Reconstruction script finished."