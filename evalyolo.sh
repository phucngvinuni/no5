#!/bin/bash
echo "Starting evaluation script..."
PROJECT_ROOT="/mnt/c/Users/ADMIN/Downloads/New RobustSemCom - Detection downstream"
SEMCOM_CHECKPOINT_NAME="checkpoint-49.pth" # Example, use your actual best model name
SEMCOM_CHECKPOINT_DIR="${PROJECT_ROOT}/ckpt_semcom_reconstruction_yolo_fish"
DATASET_ROOT_PATH="/mnt/c/Users/ADMIN/Downloads/Mergedataset/Mergedataset/" # PARENT of train/valid/test
YOLO_WEIGHTS_PATH="${PROJECT_ROOT}/best.pt"

python3 "${PROJECT_ROOT}/evaluate_on_test_set.py" \
    --model ViT_Reconstruction_Model_Default \
    --resume "${SEMCOM_CHECKPOINT_DIR}/${SEMCOM_CHECKPOINT_NAME}" \
    --data_path "${DATASET_ROOT_PATH}" \
    --yolo_weights "${YOLO_WEIGHTS_PATH}" \
    --input_size 640 \
    --snr_db_eval 10.0 \
    --num_object_classes 1 \
    --batch_size 8 \
    --device cuda \
    --num_workers 2 \
    --seed 42
echo "Evaluation script finished."