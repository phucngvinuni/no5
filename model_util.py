# model_util.py
import math
from typing import Tuple, Optional
import numpy as np
# from timm.models.registry import register_model # Not registering here, but in model.py
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

# --- (DropPath, Mlp, Attention, Block, PatchEmbed, get_sinusoid_encoding_table, ViTEncoder_Van
#       HierarchicalQuantizer, Channels - should be the same as the last correct versions) ---
# For brevity, I'm only showing ViTDecoder_ImageReconstruction with the CNN head.
# Ensure other classes are correctly defined as in the previous "full code for model_util.py"
# where HierarchicalQuantizer used nn.Embedding.

def _cfg(url='', **kwargs): # Keep _cfg if model.py uses it
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs
    }

class DropPath(nn.Module): # ... (as before) ...
    def __init__(self, drop_prob=None): super(DropPath, self).__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self) -> str: return f'p={self.drop_prob}'

class Mlp(nn.Module): # ... (as before) ...
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(); out_features = out_features or in_features; hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features); self.act = act_layer(); self.fc2 = nn.Linear(hidden_features, out_features); self.drop = nn.Dropout(drop)
    def forward(self, x): x = self.fc1(x); x = self.act(x); x = self.fc2(x); x = self.drop(x); return x

class Attention(nn.Module): # ... (as before, with self.all_head_dim fix) ...
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__(); self.num_heads = num_heads; head_dim = dim // num_heads
        if attn_head_dim is not None: head_dim = attn_head_dim
        self.all_head_dim = head_dim * self.num_heads; self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop); self.proj = nn.Linear(self.all_head_dim, dim); self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape; head_dim_calc = self.all_head_dim // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim_calc).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0); q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.all_head_dim)
        x = self.proj(x); x = self.proj_drop(x); return x

class Block(nn.Module): # ... (as before) ...
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., init_values=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__(); self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(); self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio); self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0: self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)); self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else: self.gamma_1, self.gamma_2 = None, None
    def forward(self, x):
        if self.gamma_1 is None: x = x + self.drop_path(self.attn(self.norm1(x))); x = x + self.drop_path(self.mlp(self.norm2(x)))
        else: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x))); x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module): # ... (as before) ...
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(); img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.img_size = img_size; self.patch_size = patch_size
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]): raise ValueError(f"Input image size ({H}*{W}) doesn't match model PatchEmbed size ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x).flatten(2).transpose(1, 2); return x

def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False): # ... (as before) ...
    if cls_token: n_position = n_position + 1
    def get_position_angle_vec(position): return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]); sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ViTEncoder_Van(nn.Module): # ... (as before) ...
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.0, use_learnable_pos_emb=False):
        super().__init__(); self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim); self.num_patches = self.patch_embed.num_patches
        if use_learnable_pos_emb: self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim)); trunc_normal_(self.pos_embed, std=.02)
        else: self.register_buffer('pos_embed', get_sinusoid_encoding_table(self.num_patches, embed_dim), persistent=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]; self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        self.norm = norm_layer(embed_dim); self.apply(self._init_weights_vit)
    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def forward(self, x_img: torch.Tensor, encoder_boolean_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_tokens = self.patch_embed(x_img); x_tokens = x_tokens + self.pos_embed
        x_proc = x_tokens
        if encoder_boolean_mask is not None and encoder_boolean_mask.any(): # True in mask means MASKED for encoder
            # Assuming B, L, D for x_tokens and B, L for mask
            # This simple selection requires all items in batch to have same number of visible tokens
            # or for x_tokens[~encoder_boolean_mask] to be followed by a padding/unpadding mechanism
            # For mask_ratio = 0, encoder_boolean_mask is all False, ~encoder_boolean_mask is all True, all tokens are kept.
            x_proc = x_tokens[~encoder_boolean_mask].view(x_tokens.shape[0], -1, self.embed_dim)
        for blk in self.blocks: x_proc = blk(x_proc)
        x_proc = self.norm(x_proc); return x_proc


class ViTDecoder_ImageReconstruction(nn.Module):
    def __init__(self, patch_size: int = 16, num_total_patches: int = 196,
                 embed_dim: int = 192, # Input token dimension for decoder's ViT blocks
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, norm_layer=nn.LayerNorm, init_values: float = 0.0,
                 out_chans: int = 3):
        super().__init__()
        self.num_total_patches = num_total_patches
        self.patch_size_h, self.patch_size_w = to_2tuple(patch_size)
        self.embed_dim = embed_dim # For ViT blocks

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed_decoder', get_sinusoid_encoding_table(self.num_total_patches, embed_dim), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer, init_values=init_values, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # CNN Head
        # Calculate number of upsampling stages (assuming 2x upsampling per stage)
        # e.g., patch_size 16 -> 16x upsample -> log2(16) = 4 stages
        if patch_size == 0: raise ValueError("patch_size cannot be zero.")
        num_upsample_stages = int(math.log2(patch_size))
        if 2**num_upsample_stages != patch_size:
            raise ValueError(f"patch_size ({patch_size}) must be a power of 2 for this simple CNN head design.")

        cnn_layers = []
        current_channels = embed_dim
        
        for i in range(num_upsample_stages):
            out_c = current_channels // 2 if i < num_upsample_stages - 1 else out_chans
            if out_c < out_chans and i < num_upsample_stages -1 : # Ensure intermediate channels don't go below out_chans
                out_c = max(out_chans, current_channels // 2, 16) # Ensure some minimum channels
            
            cnn_layers.append(
                nn.ConvTranspose2d(current_channels, out_c, kernel_size=4, stride=2, padding=1)
            )
            if i < num_upsample_stages - 1: # No activation/norm before final output if it's sigmoid/tanh
                cnn_layers.append(nn.BatchNorm2d(out_c)) # Using BatchNorm for CNNs
                cnn_layers.append(nn.GELU())
            current_channels = out_c
        
        cnn_layers.append(nn.Sigmoid())
        self.cnn_pixel_head = nn.Sequential(*cnn_layers)
        
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_vit)

    def _init_weights_vit(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
             trunc_normal_(m.weight, std=.02)
             if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self,
                x_vis_tokens: torch.Tensor,
                encoder_mask_boolean: torch.Tensor,
                full_image_num_patches_h: int,
                full_image_num_patches_w: int,
                ids_restore_if_mae: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        B, N_processed_tokens, D_in = x_vis_tokens.shape
        assert D_in == self.embed_dim, f"Decoder input embed_dim mismatch. Expected {self.embed_dim}, got {D_in}"

        x_full_sequence: torch.Tensor
        pos_embed_expanded = self.pos_embed_decoder.expand(B, -1, -1).to(x_vis_tokens.device, x_vis_tokens.dtype) # Ensure pos_embed matches x_vis_tokens dtype

        if not encoder_mask_boolean.any(): # Case 1: No masking by encoder (mask_ratio=0.0)
            # ... (this part was likely okay, but ensure pos_embed dtype matches) ...
            if N_processed_tokens != self.num_total_patches:
                raise ValueError(f"Decoder received {N_processed_tokens} tokens but encoder_mask indicates no masking. Expected {self.num_total_patches} tokens.")
            x_full_sequence = x_vis_tokens + pos_embed_expanded # Add after ensuring pos_embed_expanded has same dtype
        
        else: # Case 2: MAE-style masking was done by encoder
            if ids_restore_if_mae is None:
                # Fallback path if ids_restore is not provided
                # Initialize full sequence with mask_token + pos_embed
                # Ensure mask_token is cast to the correct dtype before adding to pos_embed_expanded
                # and before assigning to x_full_sequence if x_full_sequence is half.
                
                # Create the placeholder with the correct dtype from the start
                x_full_sequence = self.mask_token.expand(B, self.num_total_patches, -1).to(x_vis_tokens.dtype) + \
                                  pos_embed_expanded # pos_embed_expanded now matches x_vis_tokens.dtype

                # Place visible tokens (already with their pos_embeds from encoder if that's the MAE design)
                # This path needs careful re-evaluation if ViTEncoder_Van doesn't add pos_embed before returning visible tokens
                # Assuming x_vis_tokens are *just* the features and need pos_embed added here
                
                # A common MAE decoder pattern:
                # 1. Initialize all N_total positions with mask_token + its_pos_embed
                # 2. Scatter/place visible_tokens + their_pos_embeds into these positions.

                # Let's refine this MAE path (without ids_restore, relying on boolean mask)
                # This assumes x_vis_tokens are just the features of visible patches, WITHOUT pos_embed yet.
                
                # Create indices for scattering
                # ~encoder_mask_boolean gives True for visible patches
                visible_indices_flat = (~encoder_mask_boolean).nonzero(as_tuple=False) # [TotalVisible, 2] (batch_idx, patch_idx_in_seq)
                
                # Ensure x_vis_tokens is [TotalVisibleAcrossBatch, D_in] if it comes flattened from encoder logic
                # Or if it's [B, N_vis_per_item, D_in], we need to handle it.
                # Your ViTEncoder_Van's current forward: x_proc = x_tokens[~encoder_boolean_mask].view(B, -1, D)
                # This means x_vis_tokens IS [B, N_vis_per_item, D] where N_vis_per_item can vary.
                # This makes direct scatter hard without padding or looping.
                # For MAE, it's typical that N_vis_per_item IS fixed by sampling a fixed number of visible patches.

                # --- Simpler logic for mask_ratio > 0 if ids_restore is NOT used ---
                # This requires that x_vis_tokens contains features for *all* patches,
                # where masked patches might have a zeroed-out or placeholder representation
                # from the encoder, or that the encoder always returns a fixed number of visible tokens.
                # Given your current ViTEncoder_Van, if masking is active, it returns variable N_vis.
                # This is the core difficulty without ids_restore.

                # FOR NOW, LET'S ASSUME A SCENARIO WHERE ids_restore_if_mae IS PROVIDED FOR MASKING
                # OR that the mask_ratio = 0.0 (which makes encoder_mask_boolean.any() False)
                # The error specifically happens in the ids_restore_if_mae=None path when encoder_mask_boolean.any() is True.
                
                # If ids_restore_if_mae is None AND encoder_mask_boolean.any() is True:
                # This path is inherently complex to make batch-efficient without ids_restore
                # or fixed N_vis. The print statement for NotImplementedError was good.
                # For the dtype error, if we reach here:
                # x_full_sequence needs to be initialized with the target dtype.
                # self.mask_token needs to be cast.

                # print(f"Warning: MAE-style decoding (mask_ratio > 0) without 'ids_restore_if_mae'. "
                #       "Attempting to place visible tokens based on 'encoder_mask_boolean'. "
                #       "This can be inefficient or error-prone if num_visible_tokens varies per batch item.")

                # Initialize full sequence with the dtype of x_vis_tokens (which could be float16 due to AMP)
                x_full_sequence = torch.zeros(B, self.num_total_patches, D_in,
                                              dtype=x_vis_tokens.dtype, device=x_vis_tokens.device)
                
                # Add positional embeddings to all positions first
                x_full_sequence += pos_embed_expanded # pos_embed_expanded is now same dtype as x_vis_tokens

                # Place mask_tokens (cast to correct dtype) at MASKED positions
                # self.mask_token is nn.Parameter, default float32
                mask_token_casted = self.mask_token.to(x_full_sequence.dtype)
                # Correctly expand mask_token_casted for broadcasting to masked positions
                # This assignment needs to be careful if encoder_mask_boolean refers to the full sequence
                # but x_full_sequence was just init with pos_embeds.
                # Better: init with mask_token+pos_embed for masked, then fill visible.

                # Re-initialize x_full_sequence properly:
                x_full_sequence = mask_token_casted.expand(B, self.num_total_patches, -1) + pos_embed_expanded
                
                # Place visible tokens. x_vis_tokens are features. They also need their pos_embeds added.
                # This assumes ViTEncoder_Van did NOT add pos_embed to the x_vis_tokens it returned.
                # If ViTEncoder_Van *did* add pos_embed, then we should only add pos_embed to mask_tokens.
                
                # Assuming x_vis_tokens from encoder does NOT have pos_embed yet for this MAE path.
                # Let's reconstruct assuming x_vis_tokens need their pos_embeds.
                # And that N_processed_tokens is the number of visible tokens *per image*.
                
                # This scatter is hard without ids_restore or fixed N_vis.
                # The error `RuntimeError: Index put requires the source and destination dtypes match`
                # occurs at `x_full_sequence[encoder_mask_boolean] = self.mask_token...`
                # because x_full_sequence might be float16 and self.mask_token float32.

                # Revised MAE path when ids_restore is None (simpler scatter, less robust for variable N_vis):
                # This path is entered if encoder_mask_boolean.any() is True and ids_restore_if_mae is None.
                
                # 1. Create a tensor of mask tokens + their positional embeddings
                masked_token_representation = self.mask_token.to(x_vis_tokens.dtype) + pos_embed_expanded
                
                # 2. Create a tensor of visible tokens + their positional embeddings
                # We need to select the correct positional embeddings for the visible tokens.
                # This is tricky if x_vis_tokens is already a packed representation of visible tokens.
                # Let's assume ViTEncoder_Van returns features of visible tokens *and these tokens already had pos_embed added*.
                # If so, x_vis_tokens are ready.
                
                # Initialize x_full_sequence with the dtype of x_vis_tokens
                x_full_sequence = torch.empty(B, self.num_total_patches, D_in,
                                              dtype=x_vis_tokens.dtype, device=x_vis_tokens.device)
                
                # For each item in batch, fill x_full_sequence
                for i in range(B):
                    # Get the boolean mask for the current item: True for masked positions
                    item_encoder_mask = encoder_mask_boolean[i]
                    # Get positional embeddings for the current item
                    item_pos_embed = pos_embed_expanded[i]
                    
                    # Fill masked positions
                    x_full_sequence[i, item_encoder_mask] = self.mask_token.to(x_vis_tokens.dtype) + item_pos_embed[item_encoder_mask]
                    
                    # Fill visible positions
                    # This assumes x_vis_tokens[i] corresponds to the ~item_encoder_mask positions.
                    # If x_vis_tokens is a packed tensor [TotalVisAcrossBatch, D], this needs global indexing.
                    # Given ViTEncoder_Van returns [B, N_vis_per_item, D], this is per item.
                    x_full_sequence[i, ~item_encoder_mask] = x_vis_tokens[i] # Assumes x_vis_tokens[i] already has pos_embed from encoder
                                                                          # or that pos_embed was added to x_vis_tokens before this function

            else: # Proper MAE un-shuffling with ids_restore
                num_masked_patches = self.num_total_patches - N_processed_tokens
                if num_masked_patches < 0: raise ValueError("More visible tokens than total patches for ids_restore.")

                mask_tokens_to_append = self.mask_token.repeat(B, num_masked_patches, 1).to(x_vis_tokens.dtype) # Cast to match
                
                # x_vis_tokens are already processed by encoder. Add pos_embed after unshuffle.
                x_temp_shuffled_or_partial = torch.cat([x_vis_tokens, mask_tokens_to_append], dim=1)
                
                x_unshuffled = torch.gather(x_temp_shuffled_or_partial, dim=1,
                                            index=ids_restore_if_mae.unsqueeze(-1).expand(-1, -1, D_in))
                x_full_sequence = x_unshuffled + pos_embed_expanded # Add pos_embed to full, unshuffled sequence

        # Pass the full sequence through ViT decoder blocks
        decoded_tokens = x_full_sequence
        for blk in self.blocks:
            decoded_tokens = blk(decoded_tokens)
        decoded_tokens = self.norm(decoded_tokens)

        # Reshape for CNN head and generate pixels
        x_feat_map = decoded_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, full_image_num_patches_h, full_image_num_patches_w
        )
        reconstructed_image = self.cnn_pixel_head(x_feat_map)
        return reconstructed_image


# In model_util.py
class HierarchicalQuantizer(nn.Module):
    # ... (init as before) ...
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25,
                 hier_init: bool = True, linkage: str = 'ward'):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.hier_init_requested = hier_init
        self.hier_init_done = False
        self.linkage = linkage

    def _initialize_centroids_hierarchical(self, x_flat_cpu_np: np.ndarray):
        if self.hier_init_done: return
        try:
            from scipy.cluster.hierarchy import linkage as scipy_linkage_func, fcluster
            import numpy as np # Ensure numpy is imported here if not globally in model_util
        except ImportError:
            print("Warning: Scipy not found for HierarchicalQuantizer init. Skipping hierarchical init.");
            self.hier_init_done = True; return

        num_samples = x_flat_cpu_np.shape[0]
        if num_samples < self.num_embeddings and num_samples > 1 : # Need at least n_clusters samples for fcluster if criterion='maxclust'
             print(f"HierarchicalQuantizer: Warning - num_samples ({num_samples}) < num_embeddings ({self.num_embeddings}). Hierarchical init might be suboptimal.")
             # Continue, but be aware. Or, can implement a fallback to random if num_samples is too low.
        elif num_samples <= 1: # Cannot perform linkage
            print(f"HierarchicalQuantizer: Not enough samples ({num_samples}) for hierarchical clustering. Skipping hierarchical init.")
            self.hier_init_done = True; return


        # print(f"HierarchicalQuantizer: Initializing with {num_samples} samples, targeting {self.num_embeddings} embeddings via fcluster.")
        Z = scipy_linkage_func(x_flat_cpu_np, method=self.linkage)
        
        # Ensure n_clusters for fcluster is valid
        num_clusters_for_fcluster = min(num_samples -1 if num_samples > 1 else 1, self.num_embeddings) # maxclust must be < n_samples
        num_clusters_for_fcluster = max(1, num_clusters_for_fcluster)

        cluster_labels_from_fcluster = fcluster(Z, num_clusters_for_fcluster, criterion='maxclust')
        
        centroids = np.zeros((self.num_embeddings, self.embedding_dim), dtype=x_flat_cpu_np.dtype)
        unique_fcluster_labels = np.unique(cluster_labels_from_fcluster)
        num_actual_clusters_formed = len(unique_fcluster_labels)

        for i in range(num_actual_clusters_formed):
            centroid_idx = i # Fill the first num_actual_clusters_formed centroids
            current_fcluster_label_val = unique_fcluster_labels[i]
            current_cluster_mask = (cluster_labels_from_fcluster == current_fcluster_label_val)
            if np.any(current_cluster_mask):
                centroids[centroid_idx] = x_flat_cpu_np[current_cluster_mask].mean(axis=0)
            elif num_samples > 0: # Fallback if a cluster is unexpectedly empty
                centroids[centroid_idx] = x_flat_cpu_np[np.random.randint(num_samples)]

        if num_actual_clusters_formed < self.num_embeddings:
            # print(f"  Hierarchical clustering formed {num_actual_clusters_formed} distinct centroids. Expected {self.num_embeddings}. Filling remaining...")
            num_to_fill_additionally = self.num_embeddings - num_actual_clusters_formed
            if num_samples > 0:
                fill_indices = np.random.choice(num_samples, num_to_fill_additionally, replace=(num_samples < num_to_fill_additionally))
                centroids[num_actual_clusters_formed:] = x_flat_cpu_np[fill_indices]
            elif num_actual_clusters_formed == 0 and self.num_embeddings > 0 : # No clusters formed, fill all randomly
                 centroids = np.random.randn(self.num_embeddings, self.embedding_dim).astype(x_flat_cpu_np.dtype) * 0.1 # Small random noise


        self.embedding.weight.data.copy_(torch.from_numpy(centroids).to(self.embedding.weight.device))
        print("HierarchicalQuantizer: Centroids initialized using hierarchical clustering.")
        self.hier_init_done = True


    def forward(self, x: torch.Tensor, return_per_token_loss: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        B, N_tokens, D = original_shape
        assert D == self.embedding_dim, f"Input dim {D} != embedding_dim {self.embedding_dim}"
        x_flat = x.reshape(-1, self.embedding_dim)

        if self.hier_init_requested and not self.hier_init_done and self.training and x_flat.shape[0] > 1:
            self._initialize_centroids_hierarchical(x_flat.detach().cpu().numpy())

        distances_sq = (torch.sum(x_flat**2, dim=1, keepdim=True) +
                        torch.sum(self.embedding.weight**2, dim=1) -
                        2 * torch.matmul(x_flat, self.embedding.weight.t()))
        distances_sq = torch.clamp(distances_sq, min=0.0) # Ensure non-negative distances
        encoding_indices = torch.argmin(distances_sq, dim=1)

        quantized_flat = self.embedding(encoding_indices)
        quantized_output_tokens = quantized_flat.view(original_shape)

        # --- VQ Loss Calculation & Debugging ---
        # Term 1 (often called codebook loss or e_k in some papers): moves embedding vectors
        codebook_loss_per_token = F.mse_loss(quantized_flat, x_flat.detach(), reduction='none').mean(dim=-1).view(B, N_tokens)
        
        # Term 2 (often called commitment loss or e_c in some papers): moves encoder output
        commitment_loss_per_token = F.mse_loss(x_flat, quantized_flat.detach(), reduction='none').mean(dim=-1).view(B, N_tokens)

        if self.training and torch.rand(1).item() < 0.005: # Print for ~0.5% of training batches
            print(f"VQ_DEBUG (HierarchicalQuantizer):")
            print(f"  x_flat min: {x_flat.min().item():.4f}, max: {x_flat.max().item():.4f}, mean: {x_flat.mean().item():.4f}")
            print(f"  quantized_flat min: {quantized_flat.min().item():.4f}, max: {quantized_flat.max().item():.4f}, mean: {quantized_flat.mean().item():.4f}")
            print(f"  codebook_loss_per_token (raw) min: {codebook_loss_per_token.min().item():.4f}, max: {codebook_loss_per_token.max().item():.4f}, mean: {codebook_loss_per_token.mean().item():.4f}")
            print(f"  commitment_loss_per_token (raw) min: {commitment_loss_per_token.min().item():.4f}, max: {commitment_loss_per_token.max().item():.4f}, mean: {commitment_loss_per_token.mean().item():.4f}")
            if codebook_loss_per_token.min().item() < -1e-6 or commitment_loss_per_token.min().item() < -1e-6: # Check with a small tolerance for float errors
                print("  ERROR: Raw VQ loss component is negative!")
        # --- End VQ Loss Debugging ---

        # Standard VQ-VAE loss: L_codebook + beta * L_commitment
        per_token_total_vq_loss = codebook_loss_per_token + self.commitment_cost * commitment_loss_per_token
        
        mean_vq_total_loss = per_token_total_vq_loss.mean()
        quantized_output_tokens_st = x + (quantized_output_tokens - x).detach() # Straight-through

        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings_one_hot.view(-1, self.num_embeddings), dim=0) # Ensure flat before mean for perplexity
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if return_per_token_loss:
            return quantized_output_tokens_st, per_token_total_vq_loss, perplexity, encoding_indices
        else:
            return quantized_output_tokens_st, mean_vq_total_loss, perplexity, encoding_indices
class Channels(nn.Module):
    # ... (Keep the Channels class exactly as it was in the previous full code response) ...
    def __init__(self): super().__init__()
    def _get_noise_std(self, total_noise_variance: float) -> float: return math.sqrt(max(0.0, total_noise_variance))
    def AWGN(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        device = Tx_sig.device; noise_std = self._get_noise_std(total_noise_variance)
        if torch.is_complex(Tx_sig): noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0); noise = torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device) + 1j * torch.normal(0, noise_std_per_component, size=Tx_sig.shape, device=device)
        else: noise = torch.normal(0, noise_std, size=Tx_sig.shape, device=device)
        return Tx_sig + noise
    def _apply_flat_fading_channel(self, Tx_sig: torch.Tensor, H_complex: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        is_input_real = not torch.is_complex(Tx_sig); original_shape = Tx_sig.shape; Tx_as_complex = Tx_sig
        if is_input_real:
            if Tx_sig.shape[-1] % 2 != 0: Tx_sig = F.pad(Tx_sig, (0,1)); Tx_as_complex = torch.complex(Tx_sig[...,:Tx_sig.shape[-1]//2], Tx_sig[...,Tx_sig.shape[-1]//2:]) # Pad if odd for complex conv
            else: Tx_as_complex = torch.complex(Tx_sig[..., :Tx_sig.shape[-1]//2], Tx_sig[..., Tx_sig.shape[-1]//2:])
        faded_signal = Tx_as_complex * H_complex
        noise_std_per_component = math.sqrt(max(0.0, total_noise_variance) / 2.0)
        noise = torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device) + 1j * torch.normal(0, noise_std_per_component, faded_signal.shape, device=Tx_sig.device)
        received_noisy_faded = faded_signal + noise
        equalized_signal = received_noisy_faded / (H_complex + 1e-8)
        Rx_sig_out = equalized_signal
        if is_input_real: Rx_sig_out = torch.cat((equalized_signal.real, equalized_signal.imag), dim=-1)
        return Rx_sig_out.view(original_shape) # Ensure original shape, esp if padding was done
    def Rayleigh(self, Tx_sig: torch.Tensor, total_noise_variance: float) -> torch.Tensor:
        B = Tx_sig.shape[0]; device = Tx_sig.device; H_real = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device); H_imag = torch.normal(0, math.sqrt(1/2), size=(B, 1, 1), device=device)
        H_complex = torch.complex(H_real, H_imag); return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)
    def Rician(self, Tx_sig: torch.Tensor, total_noise_variance: float, K_factor: float = 1.0) -> torch.Tensor:
        B = Tx_sig.shape[0]; device = Tx_sig.device; H_los_real = math.sqrt(K_factor / (K_factor + 1.0)); H_los_imag = 0.0
        std_nlos_per_component = math.sqrt(1.0 / (2.0 * (K_factor + 1.0)))
        H_nlos_real = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device); H_nlos_imag = torch.normal(0, std_nlos_per_component, size=(B, 1, 1), device=device) # Centered at 0
        H_complex = torch.complex(H_los_real + H_nlos_real, H_los_imag + H_nlos_imag); return self._apply_flat_fading_channel(Tx_sig, H_complex, total_noise_variance)
        
        
class FeatureImportanceTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 fim_embed_dim: int,
                 fim_depth: int,
                 fim_num_heads: int,
                 mlp_ratio: float = 2.0,
                 norm_layer=nn.LayerNorm, # This comes from nn
                 drop_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.fim_embed_dim = fim_embed_dim

        if input_dim != fim_embed_dim:
            self.input_proj = nn.Linear(input_dim, fim_embed_dim)
        else:
            self.input_proj = nn.Identity()

        self.blocks = nn.ModuleList([
            Block(dim=fim_embed_dim, num_heads=fim_num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer, # norm_layer is passed as arg
                  drop=drop_rate, attn_drop=drop_rate)
            for _ in range(fim_depth)
        ])
        self.norm = norm_layer(fim_embed_dim) # Use the passed norm_layer

        self.output_head = nn.Sequential(
            nn.Linear(fim_embed_dim, fim_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(fim_embed_dim // 2, 1),
            # nn.Sigmoid() cause error in training
        )
        self.apply(self._init_weights) # Assuming _init_weights is defined

    def _init_weights(self, m): # Make sure this method exists or is inherited
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x_encoder_tokens: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_encoder_tokens.shape
        x = self.input_proj(x_encoder_tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        importance_scores = self.output_head(x)
        return importance_scores