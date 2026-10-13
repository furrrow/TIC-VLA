import math
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any, Tuple

from vision_utils import (
    build_transform,
    dynamic_preprocess,
    load_image,
    CrossAttentionTransformer,
)
from ticvla_vlm import TICVLA_VLM


class ActionExpert(nn.Module):
    """Produces a short-horizon chunked waypoint plan from model-state context."""

    def __init__(
        self,
        input_dim: int,  # Dimension of image embeddings (hidden_size from VLM)
        hidden_dim: int,
        action_dim: int,
        num_layers: int = 3,
        num_chunks: int = 5,
        max_state_tokens: int = 16,
        kv_cache_feat_dim: int = 128,  # num_heads * head_dim (typically 2 * 64 = 128 for InternVL3-1B)
    ) -> None:
        super().__init__()

        self.num_chunks = num_chunks
        self.action_dim = action_dim
        self.max_state_tokens = max_state_tokens
        self.hidden_dim = hidden_dim

        # Down-project visual tokens (they are usually larger than 512)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        
        # Process model state: project directly from model state dimension to hidden_dim
        # model state from VLM: (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        # Initialize kv_cache_proj here (not dynamically) for easier model loading
        self.kv_cache_proj = nn.Linear(kv_cache_feat_dim, hidden_dim)
        # Token dropout rate for model state (0.1 = drop 10% of tokens)
        self.kv_cache_token_dropout_rate = 0.1

        # discrete chunk embedding table 
        self.action_chunk_embed = nn.Embedding(num_chunks, hidden_dim)

        # Transformer cross-attention stack
        self.attention_layers = nn.ModuleList(
            [
                CrossAttentionTransformer(
                    input_dim=hidden_dim,
                    hidden_dim=4 * hidden_dim,
                    num_heads=8,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )

        # State encoder for robot low-dimensional states
        self.state_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Learnable positional embeddings for scalar state sequence
        self.state_pos_embedding = nn.Embedding(max_state_tokens, hidden_dim)

        # MLP head → predict each atomic action component
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def _process_kv_cache(self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> torch.Tensor:
        """
        Process model state from VLM into a sequence of embeddings.

        Single path, assuming standard InternVL layout:
        values are shaped (B, num_heads, seq_len, head_dim).
        """
        if not past_key_values:
            # Return empty tensor with hidden_dim (will be set when kv_cache_proj is created)
            return torch.zeros(
                1,
                0,
                self.hidden_dim,
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )

        # Take values from the last layer
        _, last_layer_value = past_key_values[-1]
        if last_layer_value.dim() != 4:
            raise ValueError(
                f"Expected model state values of shape (B, num_heads, seq_len, head_dim), "
                f"got {last_layer_value.shape}. "
                f"If this triggers, print the shape once and adjust _process_kv_cache."
            )

        B, num_heads, seq_len, head_dim = last_layer_value.shape

        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads * head_dim)
        value_reshaped = last_layer_value.permute(0, 2, 1, 3).contiguous()
        value_reshaped = value_reshaped.view(B, seq_len, num_heads * head_dim)

        # Verify dimension matches (kv_cache_proj is now initialized in __init__)
        feat_dim = value_reshaped.size(-1)  # num_heads * head_dim
        if feat_dim != self.kv_cache_proj.in_features:
            raise ValueError(
                f"model state feature dimension mismatch: expected {self.kv_cache_proj.in_features}, "
                f"got {feat_dim} (num_heads={num_heads}, head_dim={head_dim}). "
                f"Please update kv_cache_feat_dim in ActionExpert.__init__ to {feat_dim}."
            )
        
        
        # Project directly to hidden_dim
        value_reshaped = self.kv_cache_proj(value_reshaped)
        

        # Token-level dropout on model state
        if self.training and self.kv_cache_token_dropout_rate > 0:
            keep_prob = 1.0 - self.kv_cache_token_dropout_rate
            token_mask = torch.rand(
                B, seq_len, 1, device=value_reshaped.device, dtype=value_reshaped.dtype
            )
            token_mask = (token_mask < keep_prob).to(value_reshaped.dtype)
            value_reshaped = value_reshaped * token_mask

        return value_reshaped  # (B, seq_len, input_dim)

    def forward(
        self,
        image_embeds: torch.Tensor,
        state: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        """
        Return (B, T, 3) tensor - predicted waypoints relative to previous frame.
        
        Predicts waypoints (dx, dy, dtheta) relative to previous frame.
        
        Args:
            image_embeds: (B, L_img, H) image embeddings from current frame
            state: (B, S, 1) robot state tokens including velocity, displacement, and time delay
            kv_cache: Optional model state from VLM processing delayed frame context
        
        Returns:
            waypoints: (B, T, 3) tensor of waypoints [dx, dy, dtheta] relative to previous frame
        """
        B = image_embeds.size(0)

        # 1) Token preparation ------------------------------------------------
        image_embeds = self.down_proj(image_embeds)  # (B, L_img, hidden_dim)
        
        # Process model state if provided
        kv_embeds = None
        if kv_cache is not None:
            # _process_kv_cache already projects directly to hidden_dim (128 -> 512)
            kv_embeds = self._process_kv_cache(kv_cache)  # (B, L_kv, hidden_dim)
            
        state_embeds = self.state_encoder(state)  # (B, S, hidden_dim)
        state_len = state_embeds.size(1)
        if state_len > 0:
            pos_ids = torch.arange(state_len, device=state_embeds.device)
            pos_ids = pos_ids.unsqueeze(0).expand(B, -1)
            pos_emb = self.state_pos_embedding(pos_ids).to(dtype=state_embeds.dtype)
            state_embeds = state_embeds + pos_emb  # No dropout on state embeddings
        
        # Concatenate: model state (if available) + current image + state embeddings
        if kv_embeds is not None:
            hidden_states = torch.cat([kv_embeds, image_embeds, state_embeds], dim=1)  # (B, L_kv + L_img + S, hidden_dim)
        else:
            hidden_states = torch.cat([image_embeds, state_embeds], dim=1)  # (B, L_img + S, hidden_dim)

        # 2) Build action queries -------------------------------------------
        chunk_ids = torch.arange(self.num_chunks, device=hidden_states.device)
        chunk_emb = self.action_chunk_embed(chunk_ids)[None, ...].expand(B, -1, -1)  # (B, T, H)
        waypoint_queries = chunk_emb  # (B, T, H)

        # 3) Cross-attention stack ------------------------------------------
        for layer in self.attention_layers:
            waypoint_queries = layer(waypoint_queries, hidden_states)

        # 4) Directly predict waypoints relative to previous frame ------------
        waypoints = self.mlp(waypoint_queries)  # (B, T, 3) where 3 = [dx, dy, dtheta] relative to previous frame

        return waypoints


# -----------------------------------------------------------------------------
# TIC-VLA with TIC-VLA
# -----------------------------------------------------------------------------

class TICVLA(nn.Module):
    def __init__(
        self,
        model_path: str = 'InternVL3-1B',
        device: str = 'cuda:0',
        num_action_chunks: int = 30,
    ) -> None:
        """
        Initialize TIC-VLA model.
        
        Args:
            model_path: Path to pretrained VLM model
            device: Device to use
            num_action_chunks: Number of decoded action chunks per prediction.
        """
        super().__init__()
        
        # Handle device: when CUDA_VISIBLE_DEVICES is set, PyTorch only sees cuda:0
        # Store the correct device as torch.device for all .to() calls
        if device.startswith('cuda:') and 'CUDA_VISIBLE_DEVICES' in os.environ:
            # When CUDA_VISIBLE_DEVICES is set, PyTorch only sees cuda:0
            self.device = torch.device('cuda:0')
            device_str_for_vlm = 'cuda:0'
        else:
            self.device = torch.device(device)
            device_str_for_vlm = device

        # Async infrastructure for background model state generation
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._kv_cache_future: Future | None = None
        self._kv_cache_start_time: float | None = None
        
        # Cache for latest model state (use previous while new one is generating)
        self._latest_kv_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self._last_response: str | None = None  # Latest VLM response text (for reasoning)
        # Track when model state was generated in SIMULATION STEPS (for logging/debugging only)
        # Note: Delay calculation is done in behavior scripts, not here
        self._kv_cache_generation_step: int | None = None  # Simulation step/frame when current model state generation was STARTED
        self._kv_cache_generation_pose: dict | None = None  # Robot pose when model state generation started {'position': [...], 'quaternion': [...]}
        self._kv_cache_completion_step: int | None = None  # Step when the current generation COMPLETED (set when polling detects completion)
        self._has_ever_had_kv_cache: bool = False  # Track if we've ever successfully extracted model state

        # Use VLM-only model for VLM component (imported from ticvla_vlm)
        self.vlm_model = TICVLA_VLM(model_path=model_path, device=device_str_for_vlm)
        # Access VLM, tokenizer, and processor from the VLM model
        self.vlm = self.vlm_model.vlm
        self.tokenizer = self.vlm_model.tokenizer
        self.processor = self.vlm_model.processor
        
        # Freeze ALL VLM parameters (not just vision backbone)
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Action expert that uses model state (following waypoint pattern).
        # Allow overriding chunk count for checkpoints trained with different horizons.
        env_num_chunks = os.getenv("TICVLA_NUM_ACTION_CHUNKS")
        if env_num_chunks is not None:
            try:
                num_action_chunks = int(env_num_chunks)
            except ValueError as e:
                raise ValueError(
                    f"Invalid TICVLA_NUM_ACTION_CHUNKS='{env_num_chunks}'; expected integer"
                ) from e
        if num_action_chunks <= 0:
            raise ValueError(f"num_action_chunks must be > 0, got {num_action_chunks}")
        self.num_action_chunks = num_action_chunks

        # num_chunks controls how many future waypoints are decoded.
        # The actual model state has shape (B, num_heads=2, seq_len, head_dim=64)
        # So feat_dim = 2 * 64 = 128
        kv_num_heads = 2  # Fixed value for InternVL3-1B model state (NOT from VLM config)
        kv_head_dim = 64  # Fixed value for InternVL3-1B model state
        kv_cache_feat_dim = kv_num_heads * kv_head_dim  # 2 * 64 = 128
        print(f"[INIT] Initializing kv_cache_proj with feat_dim={kv_cache_feat_dim} (kv_num_heads={kv_num_heads}, kv_head_dim={kv_head_dim})")
        logging.info(f"Initializing kv_cache_proj with feat_dim={kv_cache_feat_dim} (kv_num_heads={kv_num_heads}, kv_head_dim={kv_head_dim})")
        
        self.action_expert = ActionExpert(
            input_dim=self.vlm.config.llm_config.hidden_size,
            hidden_dim=512,
            action_dim=2,  # Predict (dx, dy, dtheta) relative to previous frame
            num_layers=3,
            num_chunks=self.num_action_chunks,
            kv_cache_feat_dim=kv_cache_feat_dim,  # Initialize kv_cache_proj in __init__
        ).to(self.device).to(torch.bfloat16)
        print(f"[INIT] Using num_action_chunks={self.num_action_chunks}")
        logging.info(f"Using num_action_chunks={self.num_action_chunks}")


    def forward(self, batch: dict) -> torch.Tensor:
        """
        Training forward pass with state extraction from delayed frame context.

        Expects batch keys (from `ticvla.data.policy_data.TICVLACollator`):
        - delayed_input_ids: (B, N) input IDs for delayed frame context (past images, COT, instruction)
        - delayed_attention_mask: (B, N)
        - delayed_pixel_values: (sum_tiles, C, H, W) images from delayed frame
        - delayed_image_flags: (sum_tiles, 1)
        - delayed_num_tiles_per_sample: (B,) number of tiles per sample for delayed frame
        - current_image_paths: List[List[str]] paths to current frame's single closest image
        - robot_state: (B, 5) compact state tokens [vx, vy, yaw_speed, dx, dy]
                      where dx, dy is displacement from delayed time to current time
        - time_delay: (B,) scalar delay in seconds
        - current_image_paths: List[List[str]] paths to current frame's single closest image (optional)

        Returns:
        - predicted_waypoints: (B, T, 3) where 3 = [dx, dy, dtheta] relative to current frame (after cumsum conversion), T=num_action_chunks
        """
        # Ensure IMG_CONTEXT token id is set for InternVL forward() token replacement
        if getattr(self.vlm, 'img_context_token_id', None) is None:
            try:
                self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            except Exception:
                pass

        # Get delayed frame inputs for VLM (from collator: delayed_input_ids, delayed_pixel_values, etc.)
        delayed_input_ids = batch.get('delayed_input_ids', batch.get('input_ids')).to(self.device)
        delayed_attention_mask = batch.get('delayed_attention_mask', batch.get('attention_mask')).to(self.device)
        delayed_pixel_values = batch.get('delayed_pixel_values', batch.get('pixel_values', None))
        delayed_image_flags = batch.get('delayed_image_flags', batch.get('image_flags', None))
        delayed_num_tiles_per_sample = batch.get('delayed_num_tiles_per_sample', batch.get('num_tiles_per_sample', None))
        robot_state = batch.get('robot_state', None)
        current_image_paths = batch.get('current_image_paths', [])  # List of lists for current frame images
        
        # Print messages sent to VLM elegantly (replace image tokens with paths) - only every 50 iterations
        if not hasattr(self, '_print_counter'):
            self._print_counter = 0
        self._print_counter += 1
        should_print = self._print_counter % 200 == 0
        
        if should_print and delayed_input_ids.shape[0] > 0:
            delayed_image_paths = batch.get('delayed_image_paths', [])
            if delayed_image_paths and len(delayed_image_paths) > 0:
                # Get first sample's delayed image paths
                first_sample_images = delayed_image_paths[0]
                
                # Decode text and replace image tokens with actual paths
                first_sample_ids = delayed_input_ids[0]
                non_pad_mask = first_sample_ids != 0
                if non_pad_mask.any():
                    non_pad_ids = first_sample_ids[non_pad_mask]
                    decoded_text = self.tokenizer.decode(non_pad_ids, skip_special_tokens=False)
                    
                    # Replace image token patterns with actual image paths
                    # Pattern: <img><IMG_CONTEXT>...<IMG_CONTEXT></img> -> <image: filename.jpg>
                    # Find all <img>...</img> blocks (non-greedy match)
                    img_pattern = r'<img>.*?</img>'
                    
                    image_idx = [0]  # Use list to allow modification in nested function
                    def replace_with_path(match):
                        if image_idx[0] < len(first_sample_images):
                            path = first_sample_images[image_idx[0]]
                            image_idx[0] += 1
                            # Extract just the filename for cleaner output
                            path_display = Path(path).name if path else '<missing>'
                            return f'<image: {path_display}>'
                        return '<image: missing>'
                    
                    decoded_text_clean = re.sub(img_pattern, replace_with_path, decoded_text, flags=re.DOTALL)
                    
                    print("\n" + "="*80)
                    print("PROMPT SENT TO VLM (matching training format):")
                    # print(decoded_text_clean)
                    print(decoded_text)
                    print("="*80)
                    print(f"Delayed images ({len(first_sample_images)}):")
                    for i, img_path in enumerate(first_sample_images):
                        print(f"  Image {i+1}: {img_path}")
                    # Don't print the full decoded prompt text (too verbose with tokens)
                    print("="*80 + "\n")

        # Check if delayed_pixel_values exists and is non-empty
        # This happens when dataset's image_paths is empty (no delayed frame images collected)
        # Cause: dataset __getitem__ may not populate image_paths from history/current frames
        if delayed_pixel_values is None:
            raise RuntimeError(
                "delayed_pixel_values is None. This happens when the dataset has no delayed frame images. "
                "Check dataset __getitem__: image_paths should be populated from history frames. "
                "Each sample should have at least one delayed frame image (not just current frame)."
            )
        
        delayed_pixel_values = delayed_pixel_values.to(self.device).to(torch.bfloat16)
        
        # Check if tensor is empty (0 patches) - this shouldn't happen if dataset is correct
        if delayed_pixel_values.shape[0] == 0:
            raise RuntimeError(
                f"delayed_pixel_values is an empty tensor (shape: {delayed_pixel_values.shape}). "
                "This happens when delayed_image_paths exists but all images fail to load or produce 0 tiles. "
                "Check dataset: ensure delayed frame images exist and can be loaded."
            )
        
        # delayed_image_flags MUST exist if delayed_pixel_values exists (created by collator)
        if delayed_image_flags is None:
            raise RuntimeError(
                "delayed_image_flags is None but delayed_pixel_values exists. "
                "This is a collator bug - image_flags should always be created when pixel_values exist."
            )
        delayed_image_flags = delayed_image_flags.to(self.device)

        # Forward pass through VLM to get model state (delayed frame context)
        # VLM processes: delayed frame's past images + COT + instruction + trajectory
        with torch.no_grad():  # VLM is frozen, so no gradients needed
            vlm_kwargs = {
                'input_ids': delayed_input_ids,
                'attention_mask': delayed_attention_mask,
                'pixel_values': delayed_pixel_values,
                'image_flags': delayed_image_flags,
                'return_dict': True,
                'use_cache': True,  # Enable model state
            }
            vlm_outputs = self.vlm(**vlm_kwargs)
        
        # Extract past_key_values (model state) from VLM output
        past_key_values = getattr(vlm_outputs, 'past_key_values', None)
        
        # Debug: print VLM output info
        if should_print:
            print(f"\n[VLM OUTPUT DEBUG]")
            print(f"  - past_key_values type: {type(past_key_values)}")
            if past_key_values is not None:
                if hasattr(past_key_values, 'layers'):
                    print(f"  - DynamicCache with {len(past_key_values.layers)} layers")
                    if len(past_key_values.layers) > 0:
                        last_layer = past_key_values.layers[-1]
                        if hasattr(last_layer, 'keys') and hasattr(last_layer, 'values'):
                            print(f"  - Last layer keys shape: {last_layer.keys.shape if isinstance(last_layer.keys, torch.Tensor) else 'N/A'}")
                            print(f"  - Last layer values shape: {last_layer.values.shape if isinstance(last_layer.values, torch.Tensor) else 'N/A'}")
                elif isinstance(past_key_values, tuple):
                    print(f"  - Tuple with {len(past_key_values)} layers")
                    if len(past_key_values) > 0:
                        print(f"  - First layer key shape: {past_key_values[0][0].shape}")
                        print(f"  - First layer value shape: {past_key_values[0][1].shape}")
                        print(f"  - Last layer key shape: {past_key_values[-1][0].shape}")
                        print(f"  - Last layer value shape: {past_key_values[-1][1].shape}")
        
        # Convert DynamicCache to tuple format (single path, no fallbacks)
        if past_key_values is not None:
            cache_type = type(past_key_values).__name__

            if cache_type == 'DynamicCache':
                # DynamicCache: extract from layers attribute
                past_key_values = tuple(
                    (layer.keys, layer.values)
                    for layer in past_key_values.layers
                )
            elif not isinstance(past_key_values, tuple):
                # Unknown format - raise error instead of fallback
                raise ValueError(
                    f"Unsupported model state type: {cache_type}. "
                    f"Expected DynamicCache or tuple, got {type(past_key_values)}"
                )
            # If already tuple, use as-is
        
        # Visual features → per-sample token sequences (current frame image)
        # Extract visual features from current frame images (for action expert)
        if current_image_paths and len(current_image_paths) > 0:
            # Extract embeddings for each sample's current image
            current_image_embeds_list = []
            for paths in current_image_paths:
                if paths and len(paths) > 0:
                    # Load and extract features for the first (closest) image
                    pixel_values_current = load_image(paths[0], input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
                    with torch.no_grad():
                        img_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
                        img_embeds = img_embeds.reshape(-1, img_embeds.shape[-1])  # (total_tokens, H)
                    current_image_embeds_list.append(img_embeds)
                else:
                    # Empty embeddings if no image
                    current_image_embeds_list.append(torch.zeros(0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16))
            
            # Pad to same length
            max_tokens = max(e.shape[0] for e in current_image_embeds_list) if current_image_embeds_list else 0
            if max_tokens > 0:
                padded_embeds = []
                for e in current_image_embeds_list:
                    if e.shape[0] < max_tokens:
                        pad = torch.zeros(max_tokens - e.shape[0], e.shape[1], device=e.device, dtype=e.dtype)
                        e = torch.cat([pad, e], dim=0) # pad to the front
                    padded_embeds.append(e)
                image_embeds = torch.stack(padded_embeds, dim=0)  # (B, L_img, H)
            else:
                batch_size = delayed_input_ids.shape[0]
                image_embeds = torch.zeros(batch_size, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)
        else:
            # No images available; create empty visual tokens
            batch_size = delayed_input_ids.size(0)
            image_embeds = torch.zeros(batch_size, 0, self.vlm.config.llm_config.hidden_size, device=self.device, dtype=torch.bfloat16)

        # State features: robot_state (5) + time_delay (1) = (6,)
        # robot_state: (B, 5) = [vx, vy, yaw_speed, dx, dy] from dataset
        robot_state = robot_state.to(self.device).to(torch.bfloat16)  # (B, 5)

        # time delay (robot input)
        time_delay = (batch["time_delay"].to(self.device).to(torch.bfloat16)).unsqueeze(1)  # (B, 1)

        # concat robot state and time delay → (B, 6)
        state = torch.cat([robot_state, time_delay], dim=1).unsqueeze(-1)  # (B, 6, 1)

        # Debug: print action expert inputs (matching the shared vision utility format)
        if not hasattr(self, '_action_expert_debug_printed'):
            self._action_expert_debug_printed = True
            print("\n" + "="*80)
            print("ACTION POLICY INPUT (forward - model-state mode):")
            print("="*80)
            print(f"robot_state: {robot_state[0].float().cpu().numpy()}")
            print(f"time_delay: {time_delay[0].float().cpu().numpy()}")
            if past_key_values and len(past_key_values) > 0:
                if isinstance(past_key_values, tuple):
                    print(f"model state last layer value shape: {past_key_values[-1][1].shape}")
                elif hasattr(past_key_values, 'layers'):
                    last_layer = past_key_values.layers[-1]
                    if hasattr(last_layer, 'values'):
                        print(f"model state last layer value shape: {last_layer.values.shape if isinstance(last_layer.values, torch.Tensor) else 'N/A'}")
            print("="*80 + "\n")

        # Decode action waypoints (relative to previous frame)
        # Uses model state (from delayed frame) + current image + state
        waypoints = self.action_expert(image_embeds, state, kv_cache=past_key_values)  # (B, T, 3) where 3 = [dx, dy, dtheta] relative to previous frame
        
        return waypoints

    @torch.inference_mode()
    def load_images(self, image_paths: list[str], input_size: int = 448, max_num: int = 1) -> tuple[list[torch.Tensor], list[int]]:
        pixel_values_list = []
        num_patches_list = []
        for p in image_paths:
            try:
                pv = load_image(p, input_size=input_size, max_num=max_num).to(torch.bfloat16).to(self.device)
                if pv is not None and pv.numel() > 0:
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.shape[0])
                # Skip empty tensors
                pass
            except Exception as e:
                print(f"Warning: Failed to load image {p}: {e}")

        return pixel_values_list, num_patches_list

    @torch.inference_mode()
    def predict(
        self,
        image_paths: list[str],
        instruction: str | None = None,
        robot_state: torch.Tensor | None = None,
        current_step: int | None = None,
        current_robot_pose: dict | None = None,
        time_delay: float = 0.0,
        previous_waypoints_text: str = "",
        delayed_image_paths: list[str] | None = None,
        robot_type: str = "mobile robot",
    ) -> tuple[Optional[str], torch.Tensor, Optional[int], bool, Optional[dict]]:
        """
        Synchronous inference pipeline using model state.

        Blocks until VLM generation completes and returns fresh waypoints every call.
        """
        instruction = instruction or "Move forward safely and efficiently."

        if not image_paths:
            raise ValueError("[TIC-VLA] ERROR: image_paths is empty.")
        delayed_img_paths = delayed_image_paths if delayed_image_paths is not None else image_paths
        if not delayed_img_paths:
            raise ValueError("[TIC-VLA] ERROR: delayed_image_paths is empty.")

        # Track generation start metadata (synchronous, so start = now)
        self._kv_cache_start_time = time.perf_counter()
        if current_step is not None:
            self._kv_cache_generation_step = current_step
            if current_robot_pose is not None:
                pos = current_robot_pose.get('position', None)
                quat = current_robot_pose.get('quaternion', None)
                pos = np.array(pos).copy() if pos is not None else None
                quat = np.array(quat).copy() if quat is not None else None
                self._kv_cache_generation_pose = {'position': pos, 'quaternion': quat}

        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.1, top_p=0.1, top_k=10)
        response, kv_cache = self.generate_and_extract_kv_cache(
            delayed_img_paths,
            instruction,
            generation_config,
            previous_waypoints_text,
            robot_type,
        )

        if current_step is not None:
            self._kv_cache_completion_step = current_step

        if kv_cache is None:
            raise RuntimeError("[TIC-VLA] state extraction failed during synchronous predict.")

        self._latest_kv_cache = kv_cache
        self._last_response = response
        self._has_ever_had_kv_cache = True

        if self._kv_cache_start_time is not None:
            latency = time.perf_counter() - self._kv_cache_start_time
            gen_start = self._kv_cache_generation_step
            gen_end = self._kv_cache_completion_step
            step_delay = (gen_end - gen_start) if gen_start is not None and gen_end is not None else None
            delay_str = f", step_delay={step_delay}" if step_delay is not None else ""
            print(f"[TIC-VLA] ✓ Sync generation completed: started_at_step={gen_start}, completed_at_step={gen_end}{delay_str}, latency={latency:.3f}s")
        self._kv_cache_start_time = None

        # Current frame image for action expert
        current_image_path = image_paths[-1]
        pixel_values_current = load_image(current_image_path, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
        image_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1]).unsqueeze(0)  # (1, L_img, H)

        # Build state with robot_state and time_delay
        if robot_state is None:
            robot_state = torch.zeros(6, device=self.device, dtype=image_embeds.dtype)
        elif not torch.is_tensor(robot_state):
            robot_state = torch.tensor(robot_state, device=self.device, dtype=image_embeds.dtype)
        else:
            robot_state = robot_state.to(self.device, dtype=image_embeds.dtype)

        robot_state_flat = robot_state.view(-1)
        if robot_state_flat.shape[0] == 6:
            robot_state_tensor = torch.stack([
                robot_state_flat[0],  # vx
                robot_state_flat[1],  # vy
                robot_state_flat[3],  # yaw_speed
                robot_state_flat[4],  # dx
                robot_state_flat[5],  # dy
            ])
        elif robot_state_flat.shape[0] == 5:
            robot_state_tensor = robot_state_flat
        else:
            raise ValueError(f"[TIC-VLA] ERROR: robot_state must have 5 or 6 values, got {robot_state_flat.shape[0]}.")

        time_delay_tensor = torch.tensor([time_delay], device=self.device, dtype=image_embeds.dtype)
        state = torch.cat([robot_state_tensor, time_delay_tensor], dim=0).view(1, -1, 1)

        waypoints = self.action_expert(image_embeds, state, kv_cache=kv_cache)

        vlm_generation_start_step = self._kv_cache_generation_step if current_step is not None else None
        vlm_generation_start_pose = self._kv_cache_generation_pose if current_step is not None else None
        kv_cache_available = kv_cache is not None

        return response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose

    def generate_and_extract_kv_cache(
        self,
        delayed_img_paths: list[str],
        instruction: str,
        generation_config: dict,
        previous_waypoints_text: str = "",
        robot_type: str = "mobile robot",
    ) -> tuple[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...] | None]:
        """
        Generate text (for reasoning) and extract model state from delayed frame context (runs in background thread).
        Similar to predict(), but also extracts model state for async use.
        
        Args:
            delayed_img_paths: List of image paths for delayed frame context
            instruction: Navigation instruction text
            generation_config: Generation config for text generation
            previous_waypoints_text: Optional text describing previous waypoints
            robot_type: Type of robot (e.g., "legged robot", "wheeled robot")
        
        Returns:
            Tuple of (response_text, kv_cache) or (response_text, None) if extraction fails
        """
        # Load delayed images
        delayed_pixel_values_list, delayed_num_patches_list = self.load_images(
            delayed_img_paths, input_size=448, max_num=1
        )
        if not delayed_pixel_values_list:
            return "", None
        
        # Build prompt for VLM (same format as training - matching ticvla.data.policy_data)
        system_text = f"You are a {robot_type} assigned to perform navigation tasks.\n" + \
                      "You are provided with a video consisting of visual observations, including historical and current frames.\n"
        self.vlm.system_message = system_text
        
        user_text = f"The navigation instruction is: {instruction}"
        if previous_waypoints_text:
            user_text += f"\n{previous_waypoints_text}"
        user_text += "\nUse reasoning to predict future target waypoints for the next 3s, 6s, and 9s in format: (x, y, theta). " + \
                     "Each waypoint represents the cumulative offset from the current position (total displacement over 3s, 6s, or 9s), " + \
                     "where x is positive for forward, y is positive for left, and theta is the heading angle in radians."
        
        generation_prompt = ''.join([f'Frame {i}: <image>\n' for i in range(len(delayed_num_patches_list))]) + user_text
        print(f"[TIC-VLA] Full Prompt:\n{user_text}\n[TIC-VLA] End of Prompt")

        # Generate text response (for reasoning) - this computes model state internally
        response = self.vlm.chat(
            self.tokenizer, 
            torch.cat(delayed_pixel_values_list, dim=0), 
            generation_prompt, 
            generation_config,
            history=None, 
            return_history=False, 
            num_patches_list=delayed_num_patches_list
        )
        
        # Extract model state from FULL conversation (input + generated response)
        # Build full conversation messages including generated assistant response
        messages = [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': [{'type': 'image', 'image': p} for p in delayed_img_paths] + [{'type': 'text', 'text': user_text}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]}
        ]
        
        text_batch = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # No generation prompt needed - assistant response already included
        )
        
        if isinstance(text_batch, list):
            text_batch = text_batch[0] if len(text_batch) > 0 else ""
        
        # Build queries with image tokens
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
        num_image_token = 256  # Default for InternVL
        
        query = text_batch
        if '<image>' in query:
            for tiles_for_image in delayed_num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN
                    query = query.replace('<image>', image_tokens, 1)
        else:
            prepend = []
            for tiles_for_image in delayed_num_patches_list:
                if tiles_for_image > 0:
                    tokens_per_image = num_image_token * tiles_for_image
                    prepend.append(IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * tokens_per_image) + IMG_END_TOKEN)
            if len(prepend) > 0:
                query = '\n'.join(prepend) + '\n' + query

        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer([query], return_tensors='pt', padding=True)
        delayed_input_ids = tokenized['input_ids'].to(self.device)
        delayed_attention_mask = tokenized['attention_mask'].to(self.device)
        
        pixel_values = torch.cat(delayed_pixel_values_list, dim=0)
        total_patches = pixel_values.shape[0]
        delayed_image_flags = torch.ones(total_patches, 1, dtype=torch.long, device=self.device)
        
        # Extract model state from FULL conversation (input + generated response)
        # This includes the VLM's reasoning in the model state
        with torch.no_grad():
            vlm_outputs = self.vlm(
                pixel_values=pixel_values,
                input_ids=delayed_input_ids,
                attention_mask=delayed_attention_mask,
                image_flags=delayed_image_flags,
                return_dict=True,
                use_cache=True,
            )
        
        past_key_values = vlm_outputs.past_key_values
        
        # Handle both cache formats:
        # - DynamicCache (has `.layers`)
        # - legacy tuple((k, v), ...)
        kv_cache = None
        if past_key_values is not None:
            if hasattr(past_key_values, 'layers'):
                kv_cache = tuple(
                    (layer.keys, layer.values)
                    for layer in past_key_values.layers
                )
            elif isinstance(past_key_values, tuple):
                kv_cache = past_key_values
            else:
                print(
                    f"[TIC-VLA] WARNING: Unsupported past_key_values type in "
                    f"generate_and_extract_kv_cache: {type(past_key_values)}"
                )
        
        return response, kv_cache

    @torch.inference_mode()
    def _start_kv_cache_generation(
        self,
        delayed_img_paths: list[str],
        instruction: str,
        generation_config: dict,
        current_step: int | None = None,
        previous_waypoints_text: str = "",
        robot_type: str = "mobile robot",
        current_robot_pose: dict | None = None,
    ) -> None:
        """Start text generation and state extraction in background thread if not already running.
        
        Starts a new generation if no generation is currently running.
        First call can use any number of images (even just 1).
        
        Args:
            current_robot_pose: Robot pose at the moment generation starts {'position': [...], 'quaternion': [...]}
                                Captured and stored when a new generation actually starts.
        """
        # Check if a generation is currently running
        # A generation is running if _kv_cache_future exists and is not done
        is_generation_running = (
            self._kv_cache_future is not None and 
            not self._kv_cache_future.done()
        )
        
        # Only start a new generation if no generation is currently running
        if not is_generation_running:
            self._kv_cache_start_time = time.perf_counter()
            # Track the step and pose when generation STARTS (for delay calculation and dx/dy)
            if current_step is not None:
                self._kv_cache_generation_step = current_step
                # Capture pose at the exact moment generation starts (matches the generation step)
                if current_robot_pose is not None:
                    pos = current_robot_pose.get('position', None)
                    quat = current_robot_pose.get('quaternion', None)
                    # Copy the pose values (they might be numpy arrays or lists)
                    if pos is not None:
                        pos = np.array(pos).copy() if not isinstance(pos, np.ndarray) else pos.copy()
                    if quat is not None:
                        quat = np.array(quat).copy() if not isinstance(quat, np.ndarray) else quat.copy()
                    self._kv_cache_generation_pose = {
                        'position': pos,
                        'quaternion': quat
                    }
                print(f"[TIC-VLA] Starting vlm generation at step={current_step} with {delayed_img_paths}.")
            self._kv_cache_future = self._executor.submit(
                self.generate_and_extract_kv_cache,
                delayed_img_paths,
                instruction,
                generation_config,
                previous_waypoints_text,
                robot_type,
            )
        
    @torch.inference_mode()
    def _poll_kv_cache_future(self, current_step: int | None = None) -> None:
        """Poll for text generation and state extraction completion, update latest cache and response.
        
        Args:
            current_step: Current simulation step/frame when model state is generated.
                         Used to track when the model state was created for delay calculation.
        """
        # Check if generation is running and completed
        if self._kv_cache_future is not None:
            if self._kv_cache_future.done():
                try:
                    response, kv_cache = self._kv_cache_future.result()
                    self._last_response = response
                    if response:
                        print(f"[TIC-VLA] Generated response: {response}")  # Print first 200 chars
                    
                    # Update completion step when generation completes (even if state extraction failed)
                    # IMPORTANT: Track completion step regardless of model state success, so delay calculation works
                    if current_step is not None:
                        # Save the step when this generation COMPLETED (when we got a response)
                        self._kv_cache_completion_step = current_step
                    
                    if kv_cache is not None:
                        self._latest_kv_cache = kv_cache
                        self._has_ever_had_kv_cache = True  # Mark that we've successfully extracted model state at least once
                    else:
                        print(f"[TIC-VLA] WARNING: Generation completed but state extraction returned None")
                    
                    # Log completion
                    if current_step is not None and self._kv_cache_generation_step is not None:
                        # Also log real-world latency for debugging
                        if self._kv_cache_start_time is not None:
                            latency = time.perf_counter() - self._kv_cache_start_time
                            step_delay = current_step - self._kv_cache_generation_step
                            kv_status = "with model state" if kv_cache is not None else "without model state"
                            print(f"[TIC-VLA] ✓ Text generation completed {kv_status}: started_at_step={self._kv_cache_generation_step}, completed_at_step={current_step}, step_delay={step_delay}, latency={latency:.3f}s")
                except Exception as e:
                    import traceback
                    print(f"[TIC-VLA] ERROR in text generation/state extraction: {e}")
                    print(f"[TIC-VLA] Traceback:")
                    traceback.print_exc()
                    # Don't set completion step if generation failed
                    # This ensures we don't track failed generations as completed
                finally:
                    self._kv_cache_future = None
                    self._kv_cache_start_time = None

    @torch.inference_mode()
    def predict_async(
        self,
        image_paths: list[str],
        instruction: str | None = None,
        robot_state: torch.Tensor | None = None,
        current_step: int | None = None,
        current_robot_pose: dict | None = None,
        time_delay: float = 0.0,
        previous_waypoints_text: str = "",
        delayed_image_paths: list[str] | None = None,
        robot_type: str = "mobile robot",
    ) -> tuple[Optional[str], torch.Tensor, Optional[int], bool, Optional[dict]]:
        """
        Non-blocking, polling-based inference step using model state.

        - Starts text generation (for reasoning) and state extraction in background from delayed frame context (non-blocking)
        - Polls for completion and updates latest model state and response text when ready
        - Always uses latest available model state (from previous call if new one not ready)
        - Decodes actions using model state, current image embeddings, robot state, and time delay

        Args:
            image_paths: List of image file paths for current frame
            instruction: Navigation instruction text
            robot_state: Robot state tensor [vx, vy, vz, yaw_speed, dx, dy]
                        where dx, dy is displacement from delayed time to current time
            current_step: Current simulation step/frame number.
                         Used for model state generation tracking and fallback delay calculation.
            current_robot_pose: Current robot pose (for compatibility with waypoint version, not used in model state version).
            time_delay: Time delay in seconds between current frame and second-to-last inference start frame.
                       Should be provided by behavior scripts: delay_time = (current_frame - ref_inference_start_frame) * (1.0 / 30.0).
                       Calculated from frame difference, not simulation time, for consistency.
            previous_waypoints_text: Optional text describing previous waypoints (for prompt context).
            delayed_image_paths: List of image paths for delayed frame context (past 4 images).
                                If None, uses current images as fallback.
            robot_type: Type of robot (e.g., "legged robot", "wheeled robot")

        Returns (response_or_none, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose):
        - response_or_none: Latest VLM response text (with reasoning) if one has completed, else None
        - vlm_generation_start_step: Frame number when VLM generation started (None if no new generation started this call)
        - kv_cache_available: True if model state is available (at least one generation has completed), False otherwise
        - vlm_generation_start_pose: Robot pose when generation started {'position': [...], 'quaternion': [...]} (None if no new generation started)
        """
        if instruction is None:
            instruction = "Move forward safely and efficiently."

        # 1) Poll for previous model state completion FIRST (before starting new generation)
        self._poll_kv_cache_future(current_step)
        
        # 2) Start text generation and state extraction in background (non-blocking)
        # VLM uses 4 images from generation time: current frame + 3 earlier frames (each ~3s apart)
        # These are the "delayed" images - the context when model state was generated
        # image_paths contains: [oldest (-9s), -6s, -3s, current] after reversal in behavior script
        # If delayed_image_paths is provided, use it; otherwise use image_paths (which contains 4 sampled images)
        delayed_img_paths = delayed_image_paths if delayed_image_paths is not None else image_paths
        generation_config = dict(max_new_tokens=200, do_sample=True, temperature=0.1, top_p=0.1, top_k=10)
        
        # Track VLM generation start step before calling (to detect if new generation started)
        prev_vlm_generation_step = self._kv_cache_generation_step
        
        # Pass current_robot_pose so it can be captured when generation actually starts
        self._start_kv_cache_generation(
            delayed_img_paths,
            instruction,
            generation_config,
            current_step,
            previous_waypoints_text,
            robot_type,
            current_robot_pose=current_robot_pose
        )
        
        # Check if a new VLM generation started (for behavior script tracking)
        vlm_generation_start_step = None
        vlm_generation_start_pose = None
        if self._kv_cache_generation_step != prev_vlm_generation_step:
            # New VLM generation started - return the step and pose that were captured
            vlm_generation_start_step = self._kv_cache_generation_step
            vlm_generation_start_pose = self._kv_cache_generation_pose
        # Note: vlm_generation_start_step is None if no new generation started (previous one still running)
        
        # Debug: print delay information
        if current_step is not None:
            # gen_status = f"new_gen_at={vlm_generation_start_step}" if vlm_generation_start_step is not None else f"no_new_gen(prev_running_at={self._kv_cache_generation_step})"
            # print(f"[TIC-VLA] Delay: current_step={current_step}, time_delay={time_delay:.3f}s (from behavior script), "
            #       f"vlm_gen_step={vlm_generation_start_step} ({gen_status})")

        # 5) Use latest model state (from previous call if new one not ready yet)
        # For first generation, wait until model state is available (don't use empty cache)
        kv_cache_to_use = self._latest_kv_cache
        if kv_cache_to_use is None:
            # On first generation, wait for model state to be available
            # But if we've had model state before and now it's None, that's an error
            if self._has_ever_had_kv_cache:
                # ERROR: We've had model state before, but now it's None - extraction must be failing
                raise RuntimeError(f"[TIC-VLA] ERROR: No model state available at step={current_step} "
                                 f"but we've had model state before. Generation started at step={self._kv_cache_generation_step}. "
                                 f"This indicates state extraction is failing. Check error logs above.")
            else:
                # First generation - wait for model state to be available (blocking)
                # Don't use empty cache - wait until generation completes
                if self._kv_cache_future is not None:
                    if not self._kv_cache_future.done():
                        print(f"[TIC-VLA] First generation: Waiting for model state to be available. "
                              f"Generation started at step={self._kv_cache_generation_step}, current_step={current_step}")
                    # Wait for generation to complete (blocking - returns immediately if already done)
                    try:
                        response, kv_cache = self._kv_cache_future.result(timeout=30.0)  # 30 second timeout
                        self._last_response = response
                        # Track completion step/time for clearer logging (avoids confusion that it "finished" at the same frame)
                        if current_step is not None:
                            self._kv_cache_completion_step = current_step
                        latency = None
                        if self._kv_cache_start_time is not None:
                            latency = time.perf_counter() - self._kv_cache_start_time

                        if kv_cache is not None:
                            self._latest_kv_cache = kv_cache
                            self._has_ever_had_kv_cache = True
                            kv_cache_to_use = kv_cache
                            # Log with start/completion frames and measured latency for transparency
                            gen_start = self._kv_cache_generation_step
                            gen_end = self._kv_cache_completion_step
                            step_delay = None
                            if gen_start is not None and gen_end is not None:
                                step_delay = gen_end - gen_start
                            latency_str = f", latency={latency:.3f}s" if latency is not None else ""
                            step_str = (
                                f" started_at_step={gen_start}, completed_at_step={gen_end}"
                                if gen_start is not None and gen_end is not None
                                else ""
                            )
                            delay_str = f", step_delay={step_delay}" if step_delay is not None else ""
                            print(f"[TIC-VLA] ✓ First generation completed, model state available{step_str}{delay_str}{latency_str}")
                        else:
                            raise RuntimeError(f"[TIC-VLA] First generation completed but state extraction returned None")
                    except Exception as e:
                        import traceback
                        print(f"[TIC-VLA] ERROR waiting for first generation: {e}")
                        traceback.print_exc()
                        raise RuntimeError(f"[TIC-VLA] Failed to get model state from first generation: {e}")
                    finally:
                        self._kv_cache_future = None
                        self._kv_cache_start_time = None
                else:
                    # Generation should be running, but future is None - this shouldn't happen
                    raise RuntimeError(f"[TIC-VLA] First generation: No model state and no generation future. "
                                     f"This should not happen - generation should have been started.")

        # 6) Extract current frame image embeddings (use current frame image, not delayed images)
        # Action expert uses the current frame's single closest image (latest observation)
        # image_paths contains 4 images: [oldest, -6s, -3s, current]
        # So the current frame is the LAST image (most recent)
        current_image_path = image_paths[-1]  # Use last image (current/most recent frame)
        pixel_values_current = load_image(current_image_path, input_size=448, max_num=1).to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            image_embeds = self.vlm.extract_feature(pixel_values_current)  # (num_tiles, num_img_tokens, H)
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])  # (total_tokens, H)
            image_embeds = image_embeds.unsqueeze(0)  # (1, L_img, H)

        # 7) Build state with robot state and time delay
        time_delay_tensor = torch.tensor([time_delay], device=self.device, dtype=image_embeds.dtype)

        robot_state = robot_state.to(self.device, dtype=image_embeds.dtype)
        
        # Model expects robot_state to have 5 values: [vx, vy, yaw_speed, dx, dy]
        # But behavior scripts may send 6 values: [vx, vy, vz, yaw_speed, dx, dy]
        robot_state_flat = robot_state.view(-1)
        if robot_state_flat.shape[0] == 6:
            # Extract [vx, vy, yaw_speed, dx, dy] from [vx, vy, vz, yaw_speed, dx, dy]
            robot_state_tensor = torch.stack([
                robot_state_flat[0],  # vx
                robot_state_flat[1],  # vy
                robot_state_flat[3],  # yaw_speed (skip vz at index 2)
                robot_state_flat[4],  # dx
                robot_state_flat[5]   # dy
            ])
        else:
            raise ValueError(f"[TIC-VLA] ERROR: robot_state must have 6 values [vx, vy, vz, yaw_speed, dx, dy], got {robot_state_flat.shape[0]}: {robot_state_flat}")
        
        # print(f"[TIC-VLA] robot_state (input): {robot_state}, robot_state (model input): {robot_state_tensor}, time_delay: {time_delay}")

        # model state version uses robot_state (5 dims: vx, vy, yaw_speed, dx, dy) + time_delay (1 dim) = 6 dims total
        state = torch.cat([robot_state_tensor, time_delay_tensor], dim=0).view(1, -1, 1)  # (1, 6, 1)

        waypoints = self.action_expert(image_embeds, state, kv_cache=kv_cache_to_use)

        # Return whether model state is available (for first inference completion check)
        kv_cache_available = self._has_ever_had_kv_cache and self._latest_kv_cache is not None

        return self._last_response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose

    def reset_episode_state(self):
        """
        Reset all episode-specific state (model state, generation tracking, etc.).
        Call this at the start of each new episode to prevent stale state from affecting new episodes.
        """
        # Cancel any ongoing model state generation and wait for it to complete
        if self._kv_cache_future is not None:
            if not self._kv_cache_future.done():
                # Cancel the future
                cancelled = self._kv_cache_future.cancel()
                if not cancelled:
                    # If cancellation failed, wait for it to complete (with timeout)
                    try:
                        self._kv_cache_future.result(timeout=5.0)
                    except Exception as e:
                        print(f"[TIC-VLA] Warning: Exception while waiting for future completion during reset: {e}")
                else:
                    # Successfully cancelled, wait a bit for thread to clean up
                    time.sleep(0.1)
            # Clear the future reference
            self._kv_cache_future = None
        
        # Shutdown and recreate executor to ensure clean thread state
        # This prevents threads from previous episodes interfering with new ones
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            print(f"[TIC-VLA] Warning: Exception during executor shutdown: {e}")
        
        # Recreate executor with fresh thread
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # Clear all state
        self._kv_cache_start_time = None
        self._latest_kv_cache = None
        self._last_response = None
        self._kv_cache_generation_step = None
        self._kv_cache_generation_pose = None
        self._kv_cache_completion_step = None
        self._has_ever_had_kv_cache = False
        
        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset VLM model's internal state if possible
        # Some models cache attention states or other internal buffers
        if hasattr(self.vlm, 'reset_cache'):
            try:
                self.vlm.reset_cache()
            except Exception:
                pass
        
        print("[TIC-VLA] Episode state reset complete (executor recreated, CUDA cache cleared)")
    
    def cleanup(self):
        """
        Clean up resources when model is no longer needed.
        Should be called before deleting the model instance.
        """
        # Cancel any ongoing generation
        if self._kv_cache_future is not None:
            if not self._kv_cache_future.done():
                self._kv_cache_future.cancel()
            try:
                self._kv_cache_future.result(timeout=2.0)
            except Exception:
                pass
        
        # Shutdown executor
        try:
            self._executor.shutdown(wait=True)
        except Exception as e:
            print(f"[TIC-VLA] Warning during cleanup: {e}")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[TIC-VLA] Cleanup complete")


# Alias for compatibility with behavior scripts
TICVLA = TICVLA

