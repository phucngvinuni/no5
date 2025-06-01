import argparse

IMGC_NUMCLASS = 1   # For the OBJECT DETECTION task (e.g., 1 for 'fish' or your single class)

def get_args():
    parser = argparse.ArgumentParser('SemCom for Reconstruction + YOLO Eval with Dual VQ', add_help=False)

    # ---------------- BASIC TRAINING & DATASET ----------------
    # ... (keep existing args like --model, --data_set, --data_path, --input_size, etc.) ...
    parser.add_argument('--model', default='ViT_Reconstruction_Model_Default', type=str, metavar='MODEL',
                        help='Name of model to train (ensure it is registered in model.py)')
    parser.add_argument('--data_set', default='fish', choices=['fish'],
                        type=str, help='Dataset type')
    parser.add_argument('--data_path', default='../yolo_fish_dataset_root/', type=str,
                        help='Root dataset path (must contain train/, valid/, test/ subdirs as per YOLODataset)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image pixel H, W for SemCom input and reconstruction target')
    parser.add_argument('--num_object_classes', default=IMGC_NUMCLASS, type=int,
                        help='Number of object classes for detection task (should match your YOLO model)')
    parser.add_argument('--output_dir', default='outputs/semcom_dualvq_fish', type=str, # Adjusted output_dir
                        help='Path to save checkpoints and logs')
    parser.add_argument('--eval_viz_output_dir', default='outputs/semcom_dualvq_fish/visualizations', type=str,
                        help='Path to save evaluation visualizations')


    # ---------------- TRAINING HYPERPARAMETERS ----------------
    # ... (keep existing: --batch_size, --epochs, --update_freq, --lr, --min_lr, --warmup_epochs, --warmup_lr) ...
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='Lower LR bound for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='Epochs to warmup LR')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='Initial LR for warmup')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', # <--- ADD THIS LINE
                        help='Number of warmup steps. Overrides warmup_epochs if > 0. Default -1 to use warmup_epochs.')

    # ---------------- OPTIMIZER ----------------
    # ... (keep existing: --opt, --weight_decay, --opt_betas, --opt_eps, --momentum, --clip_grad) ...
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (adamw, sgd)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', help='AdamW betas')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='AdamW epsilon')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--clip_grad', type=float, default=1.0, metavar='NORM', help='Gradient clipping norm')
    parser.add_argument('--weight_decay_end', type=float, default=None, # <--- ADD OR UNCOMMENT THIS LINE
                        help="Final value of weight decay for cosine scheduler. If None, WD is constant.")
    parser.add_argument('--save_ckpt', action='store_true', # <--- ADD THIS LINE (or uncomment it)
                        help="Enable saving model checkpoints during training.")
    # ---------------- SEMCOM MODEL ARCHITECTURE ----------------
    parser.add_argument('--patch_size', default=8, type=int, help="Patch size for ViT encoder")
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--encoder_depth', default=8, type=int)
    parser.add_argument('--encoder_num_heads', default=8, type=int)
    parser.add_argument('--decoder_embed_dim', default=512, type=int)
    parser.add_argument('--decoder_depth', default=6, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, metavar='PCT')

    # Quantizer Parameters (Common dimension, different bitrates/codebook sizes)
    parser.add_argument('--quantizer_dim', default=512, type=int,
                        help='Embedding dimension for VQ codebook vectors (same for high and low).')
    parser.add_argument('--bits_vq_high', default=10, type=int, # More bits for important
                        help='Number of bits for the high-importance VQ codebook size (2^bits).')
    parser.add_argument('--bits_vq_low', default=10, type=int,  # Fewer bits for less important
                        help='Number of bits for the low-importance VQ codebook size (2^bits).')
    # Remove old --bits_for_quantizer if it exists to avoid confusion

    # ---------------- FIM (Feature Importance Module) PARAMETERS ----------------
    parser.add_argument('--fim_embed_dim', default=256, type=int)
    parser.add_argument('--fim_depth', default=2, type=int)
    parser.add_argument('--fim_num_heads', default=4, type=int)
    parser.add_argument('--fim_drop_rate', default=0.1, type=float)
    parser.add_argument('--fim_routing_threshold', default=0.7, type=float, # Threshold to route to VQ_High
                        help='FIM score threshold to route tokens to high-importance VQ.')

    # ---------------- LOSS WEIGHTS ----------------
    parser.add_argument('--inside_box_loss_weight', default=1, type=float)
    parser.add_argument('--outside_box_loss_weight', default=1, type=float)
    parser.add_argument('--vq_loss_weight', default=0.15, type=float, # Might need adjustment
                        help='Weight for the combined (FIM-weighted) VQ losses.')
    parser.add_argument('--fim_loss_weight', default=0.5, type=float, # Might need adjustment
                        help='Weight for the FIM training loss.')
    parser.add_argument('--lpips_loss_weight', default=0, type=float,
                        help='Weight for LPIPS perceptual loss (0 to disable).')

    # ---------------- CHANNEL SIMULATION ----------------
    # ... (keep existing: --channel_type, --snr_db_train_min, --snr_db_train_max, --snr_db_eval) ...
    parser.add_argument('--channel_type', default='rayleigh', choices=['none', 'awgn', 'rayleigh', 'rician'], type=str)
    parser.add_argument('--snr_db_train_min', default=20, type=float)
    parser.add_argument('--snr_db_train_max', default=25, type=float)
    parser.add_argument('--snr_db_eval', default=22, type=float)

    # ---------------- YOLO EVALUATION ----------------
    # ... (keep existing: --yolo_weights, --yolo_conf_thres, --yolo_iou_thres) ...
    parser.add_argument('--yolo_weights', default='best.pt', type=str)
    parser.add_argument('--yolo_conf_thres', default=0.4, type=float)
    parser.add_argument('--yolo_iou_thres', default=0.5, type=float)

    # ---------------- MISC & RUN CONTROL ----------------
    # ... (keep existing: --mask_ratio, --save_freq, --device, --seed, --num_workers, --pin_mem, --resume, --eval, etc.)
    parser.add_argument('--mask_ratio', default=0.0, type=float)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true'); parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--model_key', default='model', type=str) # For loading checkpoints
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true'); parser.set_defaults(dist_on_itp=False)
    parser.add_argument('--dist_url', default='env://')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")