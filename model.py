# model.py
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
from functools import partial
from timm.models.registry import register_model

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, Channels, FeatureImportanceTransformer, _cfg
)

class ViT_Reconstruction_Model(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 encoder_in_chans: int = 3,
                 encoder_embed_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 init_values: float = 0.0,
                 use_learnable_pos_emb: bool = False,
                 # Quantizer Args - Common dimension, different codebook sizes
                 quantizer_dim: int = 512, # Common embedding dimension for both VQs
                 bits_vq_high: int = 12,   # For high importance tokens
                 bits_vq_low: int = 8,     # For low importance tokens
                 quantizer_commitment_cost: float = 0.25,
                 # FIM Args
                 fim_embed_dim: int = 128,
                 fim_depth: int = 2,
                 fim_num_heads: int = 4,
                 fim_drop_rate: float = 0.1,
                 fim_routing_threshold: float = 0.6, # Threshold for routing to VQ_High
                 **kwargs):
        super().__init__()

        self.fim_routing_threshold = fim_routing_threshold
        effective_patch_size = patch_size

        self.img_encoder = ViTEncoder_Van(
            img_size=img_size, patch_size=effective_patch_size, in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.full_image_num_patches_h = self.img_encoder.patch_embed.patch_shape[0]
        self.full_image_num_patches_w = self.img_encoder.patch_embed.patch_shape[1]
        num_total_patches_in_image = self.img_encoder.patch_embed.num_patches

        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=effective_patch_size, num_total_patches=num_total_patches_in_image,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, init_values=init_values
        )

        self.encoder_to_channel_proj = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.norm_before_quantizer = nn.LayerNorm(quantizer_dim)
        
        self.quantizer_high = HierarchicalQuantizer(
            num_embeddings=2**bits_vq_high,
            embedding_dim=quantizer_dim, # Using common quantizer_dim
            commitment_cost=quantizer_commitment_cost
        )
        self.quantizer_low = HierarchicalQuantizer(
            num_embeddings=2**bits_vq_low,
            embedding_dim=quantizer_dim, # Using common quantizer_dim
            commitment_cost=quantizer_commitment_cost
        )
        
        self.channel_simulator = Channels()
        self.channel_to_decoder_proj = nn.Linear(quantizer_dim, decoder_embed_dim) # Single projection

        self.fim_module = FeatureImportanceTransformer(
            input_dim=encoder_embed_dim, fim_embed_dim=fim_embed_dim, fim_depth=fim_depth,
            fim_num_heads=fim_num_heads, drop_rate=fim_drop_rate,
            norm_layer=partial(norm_layer, eps=1e-6)
        )
        self.current_vq_loss = torch.tensor(0.0, dtype=torch.float32)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0); nn.init.constant_(module.weight, 1.0)

    def forward(self,
                img: torch.Tensor,
                bm_pos: Optional[torch.Tensor] = None,
                _eval: bool = False,
                eval_snr_db: float = 30.0,
                train_snr_db_min: float = 10.0,
                train_snr_db_max: float = 25.0,
                **kwargs # Catch other args like 'targets' if passed by some wrapper
               ) -> Dict[str, torch.Tensor]:

        # Clean up kwargs print if desired, or make more specific
        # if kwargs: print(f"Warning: ViT_Reconstruction_Model.forward() received unexpected kwargs: {kwargs}")

        B = img.shape[0]
        is_currently_training = self.training and not _eval

        if bm_pos is None:
            encoder_input_mask_bool = torch.zeros(
                B, self.img_encoder.patch_embed.num_patches, dtype=torch.bool, device=img.device
            )
        else:
            encoder_input_mask_bool = bm_pos

        # 1. Encoder
        x_encoded_tokens = self.img_encoder(img, encoder_input_mask_bool)
        
        # 2. FIM -> raw logits -> scores for routing & VQ loss weighting
        fim_raw_logits = self.fim_module(x_encoded_tokens)
        fim_scores_01_for_routing_and_loss_weighting = torch.sigmoid(fim_raw_logits) # [B, NumVisPatches, 1]

        # 3. Project features for quantizers/channel
        x_proj_for_channel = self.encoder_to_channel_proj(x_encoded_tokens)
        x_proj_normalized = self.norm_before_quantizer(x_proj_for_channel) # [B, NumVisPatches, QuantizerDim]

        # --- FIM-Gated Dual Quantization Logic (Iterating over batch) ---
        # This assumes NumVisPatches is the same for all items if mask_ratio > 0 (due to RandomMaskingGenerator)
        # If mask_ratio = 0, NumVisPatches = TotalPatches.
        
        # Squeeze FIM scores for easier masking
        fim_scores_squeezed = fim_scores_01_for_routing_and_loss_weighting.squeeze(-1) # [B, NumVisPatches]
        route_to_high_mask = (fim_scores_squeezed > self.fim_routing_threshold) # [B, NumVisPatches] boolean

        final_tokens_for_channel = torch.zeros_like(x_proj_normalized)
        
        accumulated_weighted_vq_loss_value = 0.0
        num_tokens_actually_quantized_total = 0

        for i in range(B): # Iterate over batch samples
            sample_x_proj_norm = x_proj_normalized[i] # [NumVisPatches, QuantizerDim]
            sample_route_high = route_to_high_mask[i]   # [NumVisPatches] boolean
            sample_fim_scores = fim_scores_squeezed[i]  # [NumVisPatches] float scores

            # --- High Importance Tokens ---
            tokens_for_high_vq_i = sample_x_proj_norm[sample_route_high] # [N_high_i, QuantizerDim]
            if tokens_for_high_vq_i.numel() > 0:
                fim_scores_high_i = sample_fim_scores[sample_route_high] # [N_high_i]
                
                q_high_st_i, pt_vq_loss_high_i, _, _ = self.quantizer_high(
                    tokens_for_high_vq_i.unsqueeze(0), return_per_token_loss=True # Pass as batch of 1
                ) # pt_vq_loss_high_i: [1, N_high_i]
                
                weighted_vq_loss_high_i = (fim_scores_high_i * pt_vq_loss_high_i.squeeze(0)).sum()
                accumulated_weighted_vq_loss_value += weighted_vq_loss_high_i.item()
                num_tokens_actually_quantized_total += tokens_for_high_vq_i.size(0)
                final_tokens_for_channel[i, sample_route_high] = q_high_st_i.squeeze(0)

            # --- Low Importance Tokens ---
            tokens_for_low_vq_i = sample_x_proj_norm[~sample_route_high] # [N_low_i, QuantizerDim]
            if tokens_for_low_vq_i.numel() > 0:
                fim_scores_low_i = sample_fim_scores[~sample_route_high] # [N_low_i]

                q_low_st_i, pt_vq_loss_low_i, _, _ = self.quantizer_low(
                    tokens_for_low_vq_i.unsqueeze(0), return_per_token_loss=True # Pass as batch of 1
                ) # pt_vq_loss_low_i: [1, N_low_i]

                weighted_vq_loss_low_i = (fim_scores_low_i * pt_vq_loss_low_i.squeeze(0)).sum()
                accumulated_weighted_vq_loss_value += weighted_vq_loss_low_i.item()
                num_tokens_actually_quantized_total += tokens_for_low_vq_i.size(0)
                final_tokens_for_channel[i, ~sample_route_high] = q_low_st_i.squeeze(0)
        
        if num_tokens_actually_quantized_total > 0:
            self.current_vq_loss = torch.tensor(accumulated_weighted_vq_loss_value / num_tokens_actually_quantized_total, device=img.device)
        else:
            self.current_vq_loss = torch.tensor(0.0, device=img.device)
        # --- End Dual VQ Logic ---

        # 4. Channel Simulation
        if is_currently_training:
            current_snr_db_tensor = torch.rand(1, device=img.device) * (train_snr_db_max - train_snr_db_min) + train_snr_db_min
        else:
            current_snr_db_tensor = torch.tensor(eval_snr_db, device=img.device)
        noise_power_variance = 10**(-current_snr_db_tensor / 10.0)
        tokens_after_channel = self.channel_simulator.Rayleigh(final_tokens_for_channel, noise_power_variance.item())

        # 5. Decoder
        x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
        
        reconstructed_image = self.img_decoder(
            x_vis_tokens=x_for_decoder_input, # These are features for visible encoder patches
            encoder_mask_boolean=encoder_input_mask_bool, # Mask indicating which original patches were fed to encoder
            full_image_num_patches_h=self.full_image_num_patches_h,
            full_image_num_patches_w=self.full_image_num_patches_w,
            ids_restore_if_mae=None # For mask_ratio=0, this is fine.
                                    # If mask_ratio > 0, MAE-style decoder would need ids_restore.
        )

        output_dict = {
            'reconstructed_image': reconstructed_image,
            'vq_loss': self.current_vq_loss,
            'fim_importance_scores': fim_raw_logits # Pass RAW LOGITS for FIM training loss
        }
        return output_dict

@register_model
def ViT_Reconstruction_Model_Default(pretrained: bool = False, **kwargs) -> ViT_Reconstruction_Model:
    model_defaults = dict(
        patch_size=16, encoder_in_chans=3,
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3, # Decoder ViT blocks
        mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantizer_dim=256, # Common dimension for both VQs
        bits_vq_high=10,   # Default bits for high-importance VQ
        bits_vq_low=6,     # Default bits for low-importance VQ
        quantizer_commitment_cost=0.25,
        init_values=0.0, use_learnable_pos_emb=False,
        drop_rate=0.0, drop_path_rate=0.1,
        fim_embed_dim=128, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
        fim_routing_threshold=0.6, # Default routing threshold
    )
    
    final_model_constructor_args = model_defaults.copy()

    # Resolve img_size
    if 'input_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['input_size']
    elif 'img_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['img_size']
    else: final_model_constructor_args['img_size'] = 224

    # Override defaults with any relevant keys from kwargs
    for key in final_model_constructor_args.keys():
        if key in kwargs:
            final_model_constructor_args[key] = kwargs[key]
    
    model = ViT_Reconstruction_Model(**final_model_constructor_args)
    model.default_cfg = _cfg() # Define or import _cfg appropriately
    if pretrained:
        print("Warning: `pretrained=True` but no pretrained weight loading implemented in factory.")
    return model