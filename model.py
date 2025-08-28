# model.py
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.registry import register_model

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, Channels, FeatureImportanceTransformer, _cfg
)
# --- BƯỚC 1: IMPORT MODULE KÊNH KỸ THUẬT SỐ ---
# Giả định bạn đã tạo file digital_channel.py
from digital_channel import transmit_and_receive_indices_batch

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
                 quantizer_dim: int = 512,
                 bits_vq_high: int = 12,
                 bits_vq_low: int = 8,
                 quantizer_commitment_cost: float = 0.25,
                 fim_embed_dim: int = 128,
                 fim_depth: int = 2,
                 fim_num_heads: int = 4,
                 fim_drop_rate: float = 0.1,
                 fim_routing_threshold: float = 0.6,
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
            embedding_dim=quantizer_dim,
            commitment_cost=quantizer_commitment_cost
        )
        self.quantizer_low = HierarchicalQuantizer(
            num_embeddings=2**bits_vq_low,
            embedding_dim=quantizer_dim,
            commitment_cost=quantizer_commitment_cost
        )
        
        # Channel simulator này giờ chỉ được dùng cho "analog proxy" trong lúc training
        self.channel_simulator = Channels() 
        self.channel_to_decoder_proj = nn.Linear(quantizer_dim, decoder_embed_dim)

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
                **kwargs
               ) -> Dict[str, torch.Tensor]:

        B = img.shape[0]
        is_training = self.training and not _eval
        
        if is_training:
            current_snr_db = (torch.rand(1).item() * (train_snr_db_max - train_snr_db_min)) + train_snr_db_min
        else:
            current_snr_db = eval_snr_db

        if bm_pos is None:
            encoder_input_mask_bool = torch.zeros(
                B, self.img_encoder.patch_embed.num_patches, dtype=torch.bool, device=img.device
            )
        else:
            encoder_input_mask_bool = bm_pos

        # 1. Encoder
        x_encoded_tokens = self.img_encoder(img, encoder_input_mask_bool)
        
        # 2. FIM -> Routing Mask
        fim_raw_logits = self.fim_module(x_encoded_tokens)
        fim_scores = torch.sigmoid(fim_raw_logits)
        route_to_high_mask = (fim_scores.squeeze(-1) > self.fim_routing_threshold) # [B, Np]

        # 3. Project features
        x_proj_normalized = self.norm_before_quantizer(self.encoder_to_channel_proj(x_encoded_tokens))

        # --- BƯỚC 4: LƯỢNG TỬ HÓA VÀ TRUYỀN TIN ---
        if is_training:
            # --- TRAINING PATH (ANALOG PROXY) ---
            final_tokens_for_channel = torch.zeros_like(x_proj_normalized)
            total_vq_loss = torch.tensor(0.0, device=img.device)
            
            # Xử lý các token quan trọng
            high_tokens_mask = route_to_high_mask
            if high_tokens_mask.any():
                q_high, vq_loss_high, _, _ = self.quantizer_high(x_proj_normalized[high_tokens_mask])
                final_tokens_for_channel[high_tokens_mask] = q_high
                # Weight the VQ loss by FIM scores
                total_vq_loss += (vq_loss_high * fim_scores.squeeze(-1)[high_tokens_mask]).mean()

            # Xử lý các token không quan trọng
            low_tokens_mask = ~route_to_high_mask
            if low_tokens_mask.any():
                q_low, vq_loss_low, _, _ = self.quantizer_low(x_proj_normalized[low_tokens_mask])
                final_tokens_for_channel[low_tokens_mask] = q_low
                # Weight the VQ loss by (1 - FIM scores) could be an option, or just the scores
                total_vq_loss += (vq_loss_low * fim_scores.squeeze(-1)[low_tokens_mask]).mean()
            
            self.current_vq_loss = total_vq_loss / B
            
            # Truyền VECTORS qua kênh ANALOG mô phỏng
            noise_power_variance = 10**(-current_snr_db / 10.0)
            tokens_after_channel = self.channel_simulator.Rayleigh(final_tokens_for_channel, noise_power_variance)

        else:
            # --- EVALUATION PATH (DIGITAL PIPELINE) ---
            # Chỉ hoạt động ở batch size = 1 để đơn giản hóa, hoặc cần sửa transmit_and_receive_indices_batch
            if B > 1:
                raise NotImplementedError("Digital evaluation path currently supports batch size of 1.")

            # Lấy chỉ số và số bit cho từng token
            indices_high, indices_low = self.quantizer_high.get_indices(x_proj_normalized[route_to_high_mask]), self.quantizer_low.get_indices(x_proj_normalized[~route_to_high_mask])
            
            # Tạo một tensor chứa chỉ số và một tensor chứa số bit tương ứng
            all_indices = torch.zeros(B, x_proj_normalized.size(1), dtype=torch.long, device=img.device)
            bits_per_index = torch.zeros(B, x_proj_normalized.size(1), dtype=torch.long, device=img.device)
            
            bits_high = int(np.log2(self.quantizer_high.num_embeddings))
            bits_low = int(np.log2(self.quantizer_low.num_embeddings))

            all_indices[route_to_high_mask] = indices_high
            all_indices[~route_to_high_mask] = indices_low
            bits_per_index[route_to_high_mask] = bits_high
            bits_per_index[~route_to_high_mask] = bits_low

            # Truyền INDICES qua kênh DIGITAL mô phỏng
            recovered_indices = transmit_and_receive_indices_batch(all_indices, bits_per_index, snr_db=current_snr_db)

            # Tra cứu sổ mã để lấy lại vectors
            final_tokens_for_channel = torch.zeros_like(x_proj_normalized)
            for i in range(B):
                rec_idx_i = recovered_indices[i]
                route_high_i = route_to_high_mask[i]
                final_tokens_for_channel[i, route_high_i] = self.quantizer_high.embedding(rec_idx_i[route_high_i])
                final_tokens_for_channel[i, ~route_high_i] = self.quantizer_low.embedding(rec_idx_i[~route_high_i])

            tokens_after_channel = final_tokens_for_channel
            self.current_vq_loss = torch.tensor(0.0, device=img.device) # Không có VQ loss khi eval

        # 5. Decoder (chung cho cả hai luồng)
        x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)
        
        reconstructed_image = self.img_decoder(
            x_vis_tokens=x_for_decoder_input,
            encoder_mask_boolean=encoder_input_mask_bool,
            full_image_num_patches_h=self.full_image_num_patches_h,
            full_image_num_patches_w=self.full_image_num_patches_w,
        )

        output_dict = {
            'reconstructed_image': reconstructed_image,
            'vq_loss': self.current_vq_loss,
            'fim_importance_scores': fim_raw_logits
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
