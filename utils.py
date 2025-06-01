# utils.py
from datetime import time
import datetime
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Iterable, Optional # Ensure these are imported
import math
from pathlib import Path
import os
import io
import json
import torch.distributed as dist # For distributed utils
from collections import OrderedDict, defaultdict, deque
from pytorch_msssim import ssim

from typing import Iterable, Optional # For type hinting

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"
    def __init__(self):
        self.scaler_enabled = torch.cuda.is_available()
        # Use the constructor that is compatible with PyTorch 2.1.0
        # The `enabled` flag correctly controls whether it operates.
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.scaler_enabled)

    def __call__(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                 parameters: Iterable[torch.nn.Parameter], clip_grad: Optional[float] = None,
                 update_grad: bool = True, create_graph: bool = False):
        
        if self._scaler.is_enabled(): # Check if scaling should happen
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else: # If not enabled (e.g., on CPU), just do a normal backward pass
            loss.backward(create_graph=create_graph)

        norm = None
        if update_grad:
            if self._scaler.is_enabled():
                self._scaler.unscale_(optimizer) # Must be called before clip_grad_norm_

            if clip_grad is not None and clip_grad > 0:
                valid_parameters = [p for p in parameters if p.grad is not None]
                if valid_parameters:
                    norm = torch.nn.utils.clip_grad_norm_(valid_parameters, clip_grad)
                else:
                    norm = torch.tensor(0.0, device=loss.device) if parameters else None # Keep this line

            if self._scaler.is_enabled():
                self._scaler.step(optimizer)
                self._scaler.update()
            else: # If not enabled, just step the optimizer
                optimizer.step()
            
            optimizer.zero_grad() # Zero grad after step
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        if self.scaler_enabled: # Only load if scaler was enabled during init
             self._scaler.load_state_dict(state_dict)


# --- Distributed Training Utilities ---
def init_distributed_mode(args):
    if args.dist_on_itp: # Specific for some cluster environment
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK']) # GPU for this specific process
    elif 'SLURM_PROCID' in os.environ: # Slurm cluster
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else: # Not in a distributed environment
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0 # Default to GPU 0 if not distributed and CUDA available
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl' # Standard backend for NVIDIA GPUs
    print('| Distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier() # Wait for all processes to sync up
    setup_for_distributed(args.rank == 0) # Disable print for non-master processes

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
# --- End Distributed Training Utilities ---


def sel_criterion(args):
    criterion = nn.L1Loss(reduction='none')
    print(f"Base Reconstruction Criterion (for per-pixel loss) = {str(criterion)}")
    return criterion

def get_model(args):
    print(f"Creating model: {args.model} with input_size: {args.input_size}")
    from timm.models import create_model as timm_create_model

    model_kwargs = {
        'input_size': args.input_size,
        'img_size': args.input_size, # Some models might use this
        'patch_size': args.patch_size,
        'encoder_embed_dim': args.encoder_embed_dim,
        'encoder_depth': args.encoder_depth,
        'encoder_num_heads': args.encoder_num_heads,
        'decoder_embed_dim': args.decoder_embed_dim, # For ViT blocks in decoder
        'decoder_depth': args.decoder_depth,
        'decoder_num_heads': args.decoder_num_heads,
        'quantizer_dim': args.quantizer_dim,           # Common dim for VQs
        'bits_vq_high': args.bits_vq_high,             # Specific to this new setup
        'bits_vq_low': args.bits_vq_low,               # Specific to this new setup
        # 'bits_for_quantizer' might be removed from args if only _high/_low are used
        'quantizer_commitment_cost': getattr(args, 'quantizer_commitment_cost', 0.25), # If you add this arg
        'drop_rate': args.drop_rate,
        'drop_path_rate': args.drop_path_rate,
        'fim_embed_dim': getattr(args, 'fim_embed_dim', 128),
        'fim_depth': getattr(args, 'fim_depth', 2),
        'fim_num_heads': getattr(args, 'fim_num_heads', 4),
        'fim_drop_rate': getattr(args, 'fim_drop_rate', 0.1),
        'fim_routing_threshold': getattr(args, 'fim_routing_threshold', 0.6) # Pass new arg
    }
    # Remove 'bits_for_quantizer' if it's no longer a direct arg to the model __init__
    # if 'bits_for_quantizer' in model_kwargs:
    #     del model_kwargs['bits_for_quantizer']

    print(f"  kwargs for timm.create_model: {model_kwargs}")

    model = timm_create_model(
        args.model,
        pretrained=False,
        **model_kwargs
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'=> Number of params: {n_parameters / 1e6:.2f} M')
    return model

def create_bbox_weight_map(image_shape: Tuple[int, int],
                           gt_boxes_xyxy: torch.Tensor,
                           inside_box_weight: float = 5.0,
                           outside_box_weight: float = 1.0,
                           target_device: torch.device = torch.device('cpu')
                           ) -> torch.Tensor:
    H, W = image_shape
    weight_map = torch.full((H, W), outside_box_weight, dtype=torch.float32, device=target_device)
    if gt_boxes_xyxy.numel() > 0:
        gt_boxes_on_device = gt_boxes_xyxy.to(target_device)
        for box in gt_boxes_on_device:
            x1, y1, x2, y2 = box.long() # Ensure integer coordinates for slicing
            # Clamp coordinates to be within image boundaries
            x1_c = torch.clamp(x1, 0, W -1) # Max index is W-1
            y1_c = torch.clamp(y1, 0, H -1) # Max index is H-1
            x2_c = torch.clamp(x2, x1_c + 1, W) # Min value x1_c + 1 ensures width > 0, max is W
            y2_c = torch.clamp(y2, y1_c + 1, H) # Min value y1_c + 1 ensures height > 0, max is H
            
            if x2_c > x1_c and y2_c > y1_c: # Check if box has valid area after clamping
                weight_map[y1_c:y2_c, x1_c:x2_c] = inside_box_weight
    return weight_map


def as_img_array(image: torch.Tensor) -> torch.Tensor:
    if image.dtype != torch.float32: image = image.float()
    image = torch.clamp(image * 255.0, 0, 255) # Ensure clamping before rounding
    return torch.round(image)

def calc_psnr(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)
    # Ensure tensors are on the same device and float type for calculations
    predictions = predictions.to(targets.device, dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    pred_arr = as_img_array(predictions)
    targ_arr = as_img_array(targets)
    mse = torch.mean((pred_arr - targ_arr) ** 2.0, dim=(1, 2, 3))
    # Handle perfect match (mse=0) to avoid log(0) -> -inf and sqrt(0) -> 0 in denominator
    psnr_val = torch.where(
        mse == 0,
        torch.tensor(100.0, device=mse.device, dtype=torch.float32), # Or a large finite number like 100.0
        20 * torch.log10(255.0 / torch.sqrt(mse.clamp(min=1e-8))) # Clamp mse for stability if very small but not zero
    )
    return psnr_val.tolist()

def calc_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> list:
    if predictions.ndim == 3: predictions = predictions.unsqueeze(0)
    if targets.ndim == 3: targets = targets.unsqueeze(0)
    # Ensure tensors are on the same device and float type for calculations
    predictions = predictions.to(targets.device, dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    pred_for_ssim = as_img_array(predictions)
    targ_for_ssim = as_img_array(targets)
    try:
        # size_average=False returns SSIM per image in batch
        ssim_val = ssim(pred_for_ssim, targ_for_ssim, data_range=255.0, size_average=False, nonnegative_ssim=True)
    except RuntimeError as e:
        print(f"RuntimeError during SSIM calculation: {e}. Check win_size vs image_size.");
        ssim_val = torch.zeros(predictions.shape[0], device=predictions.device) # Return zeros on error
    return ssim_val.tolist()

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_name = str(epoch) 
    checkpoint_path = output_dir / f'checkpoint-{epoch_name}.pth'
    print(f"  DEBUG (save_model): Constructed checkpoint path: {checkpoint_path}") # Add this

    to_save = { # ... (as before)
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch, 
        'args': args,
    }
    if loss_scaler is not None:
        to_save['scaler'] = loss_scaler.state_dict()

    try:
        torch.save(to_save, checkpoint_path)
        print(f"  INFO (save_model): Checkpoint successfully saved to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"  ERROR (save_model): Failed to save checkpoint to {checkpoint_path}. Error: {e}")
        return None

# In utils.py (example of what it should do)
# In utils.py

import os
import torch
from collections import OrderedDict

def load_custom_checkpoint(model_to_load, checkpoint_path, model_key_in_ckpt,
                           optimizer_to_load=None, loss_scaler_to_load=None,
                           args_for_epoch_resume=None):
    print(f"  DEBUG (load_custom_checkpoint): Attempting to load '{checkpoint_path}'")
    
    if not os.path.isfile(checkpoint_path):
        print(f"  ERROR (load_custom_checkpoint): Checkpoint file NOT FOUND: {checkpoint_path}")
        return False

    try:
        # Load checkpoint to CPU first to avoid GPU memory issues if checkpoint is large
        # or if the saving device was different.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  DEBUG (load_custom_checkpoint): Checkpoint loaded to CPU. Keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"  ERROR (load_custom_checkpoint): Failed to load checkpoint file with torch.load(): {e}")
        return False

    # --- Model State Loading ---
    # ... (your existing model state loading logic - seems fine) ...
    state_dict_to_load = None
    possible_model_keys = model_key_in_ckpt.split('|')
    for key_try in possible_model_keys:
        if key_try in checkpoint:
            state_dict_to_load = checkpoint[key_try]
            print(f"  DEBUG (load_custom_checkpoint): Found model state_dict with key '{key_try}'.")
            break
    if state_dict_to_load is None:
        print(f"  ERROR (load_custom_checkpoint): Could not find model state_dict in checkpoint with keys: {model_key_in_ckpt}")
        return False
    new_state_dict = OrderedDict()
    for k, v in state_dict_to_load.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    try:
        missing_keys, unexpected_keys = model_to_load.load_state_dict(new_state_dict, strict=False)
        if missing_keys: print(f"    Warning (load_custom_checkpoint): Missing keys in model: {missing_keys}")
        if unexpected_keys: print(f"    Warning (load_custom_checkpoint): Unexpected keys in checkpoint: {unexpected_keys}")
        print(f"  INFO (load_custom_checkpoint): Model weights loaded into {model_to_load.__class__.__name__}.")
    except Exception as e:
        print(f"  ERROR (load_custom_checkpoint): Failed to load model state_dict: {e}")
        return False
    # --- End Model State Loading ---

    # --- Optimizer State Loading ---
    if optimizer_to_load:
        if 'optimizer' in checkpoint:
            try:
                optimizer_to_load.load_state_dict(checkpoint['optimizer'])
                print("  INFO (load_custom_checkpoint): Optimizer state loaded successfully (initially to CPU).")
                
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # ADD THIS BLOCK TO MOVE OPTIMIZER STATE TO DEVICE
                # This needs to be done *after* model.to(device) has been called in the main script
                # However, we can do it here assuming the device for optimizer should match model.
                # A safer place is in run_class_main.py *after* model.to(device) AND optimizer state load.
                # For now, let's try moving it here, assuming model_to_load.device is set.
                # It's better if `device` is passed to this function.
                
                # Let's assume device should be taken from model_to_load if it's already on a device
                # Or from args_for_epoch_resume if available
                target_device = None
                if args_for_epoch_resume and hasattr(args_for_epoch_resume, 'device'):
                    target_device = torch.device(args_for_epoch_resume.device)
                elif hasattr(model_to_load, 'device'): # This might not exist or be reliable
                    target_device = model_to_load.device
                
                if target_device and target_device.type != 'cpu': # Only move if target is CUDA
                    print(f"  DEBUG (load_custom_checkpoint): Moving optimizer state to device: {target_device}")
                    for state in optimizer_to_load.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(target_device)
                    print(f"  INFO (load_custom_checkpoint): Optimizer state moved to {target_device}.")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            except Exception as e:
                print(f"  Warning (load_custom_checkpoint): Could not load/move optimizer state: {e}. Optimizer will start fresh.")
        else:
            print("  Warning (load_custom_checkpoint): Optimizer state not found in checkpoint.")
    
    # --- Loss Scaler State Loading ---
    if loss_scaler_to_load:
        # ... (your existing loss scaler loading logic - seems fine as GradScaler handles device internally) ...
        if 'scaler' in checkpoint and hasattr(loss_scaler_to_load, 'load_state_dict') and loss_scaler_to_load.scaler_enabled:
            try:
                loss_scaler_to_load.load_state_dict(checkpoint['scaler'])
                print("  INFO (load_custom_checkpoint): Loss scaler state loaded successfully.")
            except Exception as e:
                print(f"  Warning (load_custom_checkpoint): Could not load loss scaler state: {e}. Scaler will start fresh.")
        elif 'scaler' not in checkpoint:
            print("  Warning (load_custom_checkpoint): Loss scaler state not found in checkpoint.")
        elif not (hasattr(loss_scaler_to_load, 'load_state_dict') and loss_scaler_to_load.scaler_enabled):
             print("  INFO (load_custom_checkpoint): Loss scaler not active or does not support loading state. Skipping scaler load.")


    # --- Epoch Resuming ---
    if args_for_epoch_resume and 'epoch' in checkpoint:
        previous_epoch = checkpoint['epoch']
        if args_for_epoch_resume.start_epoch == 0 or args_for_epoch_resume.resume == checkpoint_path:
            args_for_epoch_resume.start_epoch = previous_epoch + 1
            print(f"  INFO (load_custom_checkpoint): Resuming training from epoch {args_for_epoch_resume.start_epoch} (checkpoint was from epoch {previous_epoch}).")
        else:
            print(f"  INFO (load_custom_checkpoint): Checkpoint epoch is {previous_epoch}, but args.start_epoch is {args_for_epoch_resume.start_epoch}. Using args.start_epoch.")
    elif 'epoch' not in checkpoint:
        print("  Warning (load_custom_checkpoint): 'epoch' not found in checkpoint. args.start_epoch will not be updated.")
        
    return True


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep) # Ensure integer

    if warmup_steps > 0: # Override if specific warmup_steps are given
        warmup_iters = int(warmup_steps)
    
    # Handle cases where niter_per_ep might be 0 for very small datasets
    if niter_per_ep == 0 and warmup_epochs > 0:
        print(f"Warning: niter_per_ep is 0, but warmup_epochs is {warmup_epochs}. Setting warmup_iters to 0 for safety.")
        warmup_iters = 0
    
    # print("Set warmup steps = %d" % warmup_iters) # Moved print for clarity
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    total_iters = int(epochs * niter_per_ep)
    if total_iters <= 0 : # Handle case with no iterations (e.g. 0 epochs or 0 niter_per_ep)
         print(f"Warning: Total iterations is {total_iters}. Returning schedule with base_value.")
         return np.array([base_value])

    if total_iters < warmup_iters:
        print(f"Warning: Total iterations ({total_iters}) < warmup iterations ({warmup_iters}). Schedule will be only warmup phase to final value.")
        return np.linspace(start_warmup_value, final_value, total_iters)

    main_phase_iters = total_iters - warmup_iters
    iters_array = np.arange(main_phase_iters)

    if main_phase_iters <= 0: # Only warmup phase
        schedule = warmup_schedule
        # Ensure schedule length matches total_iters if only warmup and total_iters > 0
        if len(schedule) > total_iters and total_iters > 0: schedule = schedule[:total_iters]
        elif len(schedule) < total_iters and total_iters > 0: schedule = np.pad(schedule, (0, total_iters - len(schedule)), 'edge')

    else:
        # For cosine part, avoid division by zero if main_phase_iters is 1
        denominator = (main_phase_iters - 1) if main_phase_iters > 1 else 1
        main_schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(math.pi * iters_array / denominator))
        schedule = np.concatenate((warmup_schedule, main_schedule))

    # Final length check and adjustment
    if len(schedule) != total_iters:
        # This can happen due to float to int conversions or edge cases
        print(f"Scheduler length ({len(schedule)}) mismatch with total_iters ({total_iters}). Adjusting...")
        if len(schedule) < total_iters:
            schedule = np.pad(schedule, (0, total_iters - len(schedule)), 'edge') # Pad with last value
        else:
            schedule = schedule[:total_iters] # Truncate

    if len(schedule) == 0 and total_iters == 0 : # Ensure at least one value if total_iters becomes 0 somehow
        return np.array([base_value])
    elif len(schedule) == 0 and total_iters > 0: # Should not happen with current logic but defensive
        print(f"Warning: Schedule is empty but total_iters is {total_iters}. Returning linspace.")
        return np.linspace(start_warmup_value if warmup_iters > 0 else base_value, final_value, total_iters)

    return schedule

# --- Add MetricLogger if used by other scripts like genr_noise.py, or remove if not needed by main path ---
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))