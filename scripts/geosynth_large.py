import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..ControlNet.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ..ControlNet.ldm.modules.attention import SpatialTransformer
from ..ControlNet.ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ..ControlNet.ldm.models.diffusion.ddpm import LatentDiffusion
from ..ControlNet.ldm.util import log_txt_as_img, exists, instantiate_from_config
from ..ControlNet.ldm.models.diffusion.ddim import DDIMSampler


class EnhancedLocationEncoder(nn.Module):
    def __init__(self, embed_dim=256, out_dim=256, num_heads=4, use_global_pos=True):
        super().__init__()
        self.use_global_pos = use_global_pos
        
        # Original location encoding
        self.query_embed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_embed = nn.Linear(1280, embed_dim, bias=False)
        self.value_embed = nn.Linear(1280, embed_dim, bias=False)
        
        # Global position encoding (x, y, width, height relative to full image)
        if use_global_pos:
            self.global_pos_embed = nn.Linear(4, embed_dim, bias=False)
            self.pos_combine = nn.Linear(embed_dim * 2, embed_dim, bias=False)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ff_zero = zero_module(nn.Linear(embed_dim, out_dim, bias=False))

    def forward(self, loc_emb, time_emb, global_pos=None):
        # Combine local and global position information if available
        if self.use_global_pos and global_pos is not None:
            # global_pos should be tensor of shape [B, 4] with normalized x, y, w, h
            global_pos_embed = self.global_pos_embed(global_pos.float())
            combined_loc = self.pos_combine(torch.cat([loc_emb, global_pos_embed.unsqueeze(1)], dim=-1))
        else:
            combined_loc = loc_emb
            
        q, k, v = (
            self.query_embed(combined_loc),
            self.key_embed(time_emb),
            self.value_embed(time_emb),
        )
        x, _ = self.cross_attn(q, k, v)
        x = self.norm1(x) + combined_loc
        x = self.norm2(self.ff(x)) + x
        return x, self.ff_zero(x)


class TileAwareControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        location=None,
        adjacent_features=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        # import code; code.interact(local=locals());
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        # Apply control signal at bottleneck
        if control is not None:
            h += control.pop()

        # Apply location signal at bottleneck
        if location is not None:
            loc = location.pop().squeeze(1)
            loc = repeat(loc, "b d -> b d h w", h=h.shape[-2], w=h.shape[-1])
            h += loc
            
        # Apply adjacent tile features at bottleneck if available
        if adjacent_features is not None and len(adjacent_features) > 0:
            h += adjacent_features.pop()

        # Process through output blocks with skip connections
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or (control is None and location is None and adjacent_features is None):
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # Prepare skip connection with control and location signals
                skip_connection = hs.pop()
                
                if control is not None and len(control) > 0:
                    skip_connection = skip_connection + control.pop()
                    
                if location is not None and len(location) > 0:
                    loc = location.pop().squeeze(1)
                    loc = repeat(loc, "b d -> b d h w", h=skip_connection.shape[-2], w=skip_connection.shape[-1])
                    skip_connection = skip_connection + loc
                
                if adjacent_features is not None and len(adjacent_features) > 0:
                    skip_connection = skip_connection + adjacent_features.pop()
                
                h = torch.cat([h, skip_connection], dim=1)
                
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class TileAwareControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        use_global_position=True,
        adjacent_tiles_encoder=True,
    ):
        super().__init__()
        # Most initialization is the same as original ControlNet
        if use_spatial_transformer:
            assert context_dim is not None, "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert use_spatial_transformer, "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.use_global_position = use_global_position
        self.adjacent_tiles_encoder = adjacent_tiles_encoder
        
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
            
        # Rest of initialization (similar to original)
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # Enhanced hint block that can also process adjacent tile information
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )
        
        # Enhanced location encoder with global position awareness
        self.loc_blocks = nn.ModuleList([
            EnhancedLocationEncoder(
                out_dim=model_channels, 
                use_global_pos=use_global_position
            )
        ])
        
        # Adjacent tiles encoder (if enabled)
        if adjacent_tiles_encoder:
            self.adjacent_encoder = TimestepEmbedSequential(
                conv_nd(dims, hint_channels * 4, 32, 3, padding=1),  # 4 directions (N,S,E,W)
                nn.SiLU(),
                conv_nd(dims, 32, 64, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 64, 128, 3, padding=1, stride=2),
                nn.SiLU(),
                zero_module(conv_nd(dims, 128, model_channels, 3, padding=1)),
            )
            self.adjacent_zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # Rest of model structure (similar to original)
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self.loc_blocks.append(
                    EnhancedLocationEncoder(
                        out_dim=ch, 
                        use_global_pos=use_global_position
                    )
                )
                if adjacent_tiles_encoder:
                    self.adjacent_zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                if adjacent_tiles_encoder:
                    self.adjacent_zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch
                self.loc_blocks.append(
                    EnhancedLocationEncoder(
                        out_dim=ch,
                        use_global_pos=use_global_position
                    )
                )

        # Middle block (similar to original)
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
            
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        if adjacent_tiles_encoder:
            self.adjacent_middle_zero_conv = self.make_zero_conv(ch)
        self.loc_middle_block = EnhancedLocationEncoder(
            out_dim=ch,
            use_global_pos=use_global_position
        )
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )
        
    def process_adjacent_tiles(self, adjacent_tiles, emb, context):
        """Process adjacent tiles and extract features at different scales"""
        if adjacent_tiles is None or not self.adjacent_tiles_encoder:
            return None
            
        # Process adjacent_tiles through the encoder
        adj_features = self.adjacent_encoder(adjacent_tiles, emb, context)
        
        # Create a list of features for each resolution level
        adj_outs = [self.adjacent_zero_convs[0](adj_features, emb, context)]
        
        # Downsample for each level
        curr_feat = adj_features
        for i in range(1, len(self.adjacent_zero_convs)):
            # Simple downsampling
            if i % self.num_res_blocks[0] == 0 and i > 1:
                curr_feat = F.avg_pool2d(curr_feat, kernel_size=2, stride=2)
            
            # Apply zero conv to get features at this level
            adj_outs.append(self.adjacent_zero_convs[i](curr_feat, emb, context))
            
        # Middle block feature
        if hasattr(self, "adjacent_middle_zero_conv"):
            adj_outs.append(self.adjacent_middle_zero_conv(curr_feat, emb, context))
            
        return adj_outs

    def forward(self, x, hint, timesteps, context, location, global_position=None, adjacent_tiles=None, **kwargs):
        """
        Forward pass with support for adjacent tiles and global position
        
        Args:
            x: Input latent
            hint: Conditioning image
            timesteps: Diffusion timesteps
            context: Text embedding
            location: Local position encoding
            global_position: Global position encoding [B, 4] with normalized x, y, w, h
            adjacent_tiles: Adjacent tile features [B, C*4, H, W] (4 directions concatenated)
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # Process the hint (conditioning image)
        guided_hint = self.input_hint_block(hint, emb, context)
        
        # Process adjacent tiles if available
        adjacent_features = self.process_adjacent_tiles(adjacent_tiles, emb, context) if adjacent_tiles is not None else None

        outs = []
        locs = []

        # Initialize local position embedding
        loc_input = location.float().unsqueeze(1)

        # Process through input blocks
        h = x.type(self.dtype)
        for i, (module, zero_conv, loc_module) in enumerate(zip(
            self.input_blocks, self.zero_convs, self.loc_blocks
        )):
            # Apply guided hint only at the first block
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
                
            # Process location with global position
            loc_input, loc_zero = loc_module(loc_input, emb.unsqueeze(1), global_position)
            locs.append(loc_zero)
            
            # Store features for later use
            outs.append(zero_conv(h, emb, context))

        # Process through middle block
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        locs.append(self.loc_middle_block(loc_input, emb.unsqueeze(1), global_position)[1])

        return outs, locs, adjacent_features


class TileAwareControlLDM(LatentDiffusion):
    def __init__(
        self, 
        control_stage_config, 
        control_key, 
        only_mid_control, 
        tile_overlap=0.25,  # 25% overlap between tiles
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.loc_scales = [1.0] * 13
        self.adjacent_scales = [1.0] * 13
        self.tile_overlap = tile_overlap

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        location = batch["location"]
        
        # Handle global position if available
        global_position = batch.get("global_position", None)
        
        # Handle adjacent tiles if available
        adjacent_tiles = batch.get("adjacent_tiles", None)
        
        if bs is not None:
            control = control[:bs]
            if adjacent_tiles is not None:
                adjacent_tiles = adjacent_tiles[:bs]
            if global_position is not None:
                global_position = global_position[:bs]
                
        control = control.to(self.device)
        location = location.to(self.device)
        
        if global_position is not None:
            global_position = global_position.to(self.device)
            
        if adjacent_tiles is not None:
            adjacent_tiles = adjacent_tiles.to(self.device)
            adjacent_tiles = einops.rearrange(adjacent_tiles, "b h w c -> b c h w")
            adjacent_tiles = adjacent_tiles.to(memory_format=torch.contiguous_format).float()
        
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        
        return x, dict(
            c_crossattn=[c], 
            c_concat=[control], 
            c_loc=location,
            c_global_pos=global_position,
            c_adjacent=adjacent_tiles
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                location=None,
                adjacent_features=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            # Get control and location signals
            control, locs, adjacent_features = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                location=cond["c_loc"],
                global_position=cond.get("c_global_pos", None),
                adjacent_tiles=cond.get("c_adjacent", None),
                timesteps=t,
                context=cond_txt,
            )
            
            # Apply scaling factors
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            locs = [c * scale for c, scale in zip(locs, self.loc_scales)]
            
            if adjacent_features is not None:
                adjacent_features = [c * scale for c, scale in zip(adjacent_features, self.adjacent_scales)]
            
            # Apply to diffusion model
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                location=locs,
                adjacent_features=adjacent_features,
                only_mid_control=self.only_mid_control,
            )

        return eps
        
    @torch.no_grad()
    def process_tiles(self, full_image, prompt, tile_size=512, batch_size=1):
        """
        Process a large image by tiling it and generating each tile with awareness of its neighbors
        
        Args:
            full_image: Full input image to process [C, H, W]
            prompt: Text prompt for conditioning
            tile_size: Size of each tile
            batch_size: Batch size for processing tiles
            
        Returns:
            Processed full image
        """
        _, h, w = full_image.shape
        
        # Calculate overlap in pixels
        overlap_px = int(tile_size * self.tile_overlap)
        stride = tile_size - overlap_px
        
        # Calculate number of tiles in each dimension
        num_tiles_h = max(1, (h - overlap_px) // stride)
        num_tiles_w = max(1, (w - overlap_px) // stride)
        
        # Adjust stride to cover the image fully
        stride_h = (h - tile_size) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
        stride_w = (w - tile_size) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
        
        # Create output tensor for the full processed image
        output_image = torch.zeros_like(full_image)
        weight_map = torch.zeros((1, h, w), device=full_image.device)
        
        # Create a weight mask for blending (higher weight in the center, lower at the edges)
        def create_weight_mask(size):
            weight = torch.ones((size, size), device=full_image.device)
            for i in range(overlap_px):
                # Apply linear ramp at the borders
                weight[i, :] *= (i / overlap_px)
                weight[size-i-1, :] *= (i / overlap_px)
                weight[:, i] *= (i / overlap_px)
                weight[:, size-i-1] *= (i / overlap_px)
            return weight
            
        tile_weight = create_weight_mask(tile_size)
        
        # Process each tile
        tile_data = []
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate tile position
                y = int(i * stride_h)
                x = int(j * stride_w)
                
                # Ensure we don't go out of bounds
                y = min(y, h - tile_size)
                x = min(x, w - tile_size)
                
                # Extract tile
                tile = full_image[:, y:y+tile_size, x:x+tile_size]
                
                # Calculate global position (normalized)
                global_pos = torch.tensor([
                    x / w,                  # x position
                    y / h,                  # y position
                    tile_size / w,          # width
                    tile_size / h           # height
                ], device=full_image.device).unsqueeze(0)
                
                # Get adjacent tiles (if they exist)
                adjacent = {}
                # North
                if i > 0:
                    y_north = int((i-1) * stride_h)
                    y_north = min(y_north, h - tile_size)
                    adjacent["north"] = full_image[:, y_north:y_north+tile_size, x:x+tile_size]
                # South
                if i < num_tiles_h - 1:
                    y_south = int((i+1) * stride_h)
                    y_south = min(y_south, h - tile_size)
                    adjacent["south"] = full_image[:, y_south:y_south+tile_size, x:x+tile_size]
                # West
                if j > 0:
                    x_west = int((j-1) * stride_w)
                    x_west = min(x_west, w - tile_size)
                    adjacent["west"] = full_image[:, y:y+tile_size, x_west:x_west+tile_size]
                # East
                if j < num_tiles_w - 1:
                    x_east = int((j+1) * stride_w)
                    x_east = min(x_east, w - tile_size)
                    adjacent["east"] = full_image[:, y:y+tile_size, x_east:x_east+tile_size]
                
                # Store tile data
                tile_data.append({
                    "tile": tile,
                    "pos": (y, x),
                    "global_pos": global_pos,
                    "adjacent": adjacent
                })
                
        # Process tiles in batches
        for batch_idx in range(0, len(tile_data), batch_size):
            batch_tiles = tile_data[batch_idx:batch_idx + batch_size]
            
            # Prepare batch inputs
            batch_input = []
            for tile_info in batch_tiles:
                # Create input dictionary for each tile
                tile_batch = {
                    self.first_stage_key: tile_info["tile"].unsqueeze(0),
                    self.cond_stage_key: prompt,
                    self.control_key: tile_info["tile"].unsqueeze(0),
                    "location": torch.zeros(1, 256, device=self.device),  # Will be replaced with appropriate embedding
                    "global_position": tile_info["global_pos"]
                }
                
                # Add adjacent tiles if available (concatenate in channel dimension)
                if len(tile_info["adjacent"]) > 0:
                    # Create a tensor with all adjacent tiles (N, E, S, W)
                    # If a direction doesn't exist, use zeros
                    directions = ["north", "east", "south", "west"]
                    adjacent_tensor = []
                    
                    for direction in directions:
                        if direction in tile_info["adjacent"]:
                            adjacent_tensor.append(tile_info["adjacent"][direction])
                        else:
                            adjacent_tensor.append(torch.zeros_like(tile_info["tile"]))
                            
                    # Concatenate in channel dimension
                    adjacent_tensor = torch.cat(adjacent_tensor, dim=0).unsqueeze(0)
                    tile_batch["adjacent_tiles"] = adjacent_tensor
                
                batch_input.append(tile_batch)
            
            # Process this batch of tiles
            processed_tiles = []
            for tile_batch in batch_input:
                # Generate the tile with conditional guidance
                processed_tile = self.generate_tile(tile_batch)
                processed_tiles.append(processed_tile)
            
            # Blend tiles into the output image
            for tile_info, processed_tile in zip(batch_tiles, processed_tiles):
                y, x = tile_info["pos"]
                
                # Apply weight mask for blending
                weighted_tile = processed_tile * tile_weight.unsqueeze(0)
                
                # Add weighted tile to output
                output_image[:, y:y+tile_size, x:x+tile_size] += weighted_tile
                weight_map[:, y:y+tile_size, x:x+tile_size] += tile_weight
        
        # Normalize by the accumulated weights
        output_image = output_image / (weight_map + 1e-8)
        
        return output_image
        
    @torch.no_grad()
    def generate_tile(self, tile_batch, ddim_steps=50, ddim_eta=0.0, unconditional_guidance_scale=7.5):
        """Generate a single tile with the diffusion model"""
        # Setup sampler
        ddim_sampler = DDIMSampler(self)
        
        # Get model input
        x, conditioning = self.get_input(tile_batch, self.first_stage_key)
        
        # Encode to latent space
        c = conditioning
        
        # Generate unconditional conditioning
        uc = None
        if unconditional_guidance_scale != 1.0:
            uc_cross = self.get_unconditional_conditioning(x.shape[0])
            uc_loc = c["c_loc"]
            uc_concat = c["c_concat"] 
            uc_global_pos = c.get("c_global_pos", None)
            uc_adjacent = c.get("c_adjacent", None)
            
            uc = {
                "c_crossattn": [uc_cross], 
                "c_concat": uc_concat, 
                "c_loc": uc_loc,
                "c_global_pos": uc_global_pos,
                "c_adjacent": uc_adjacent
            }
        
        # Setup shape
        shape = (self.channels, x.shape[2] // 8, x.shape[3] // 8)
        
        # Sample
        samples, _ = ddim_sampler.sample(
            ddim_steps,
            x.shape[0],
            shape,
            c,
            verbose=False,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
        )
        
        # Decode
        x_samples = self.decode_first_stage(samples)
        
        return x_samples[0]  # Return as [C, H, W]

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()