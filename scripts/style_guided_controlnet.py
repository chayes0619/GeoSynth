import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ControlNet.ldm.modules.diffusionmodules.util import conv_nd, timestep_embedding
from ..ControlNet.ldm.modules.attention import SpatialTransformer
from ..scripts.geosynth import ControlNet


class StyleGuidedControlNet(ControlNet):
    """
    Extended ControlNet that incorporates style guidance using cross-attention.
    A simpler approach to style consistency than full feature integration.
    """
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
        style_attention_layer=3,  # Layer to apply style attention
        style_attention_strength=1.0,  # Strength of style guidance
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            hint_channels=hint_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
            disable_self_attentions=disable_self_attentions,
            num_attention_blocks=num_attention_blocks,
            disable_middle_self_attn=disable_middle_self_attn,
            use_linear_in_transformer=use_linear_in_transformer,
        )
        
        # Save style attention parameters
        self.style_attention_layer = style_attention_layer
        self.style_attention_strength = style_attention_strength
        
        # Simple style encoder
        self.style_encoder = nn.Sequential(
            conv_nd(2, 3, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 128, model_channels, 3, padding=1, stride=2)
        )
        
        # Calculate the feature dimensions at the target layer
        layer_channels = model_channels
        for i in range(min(style_attention_layer, len(channel_mult))):
            if i < len(channel_mult):
                layer_channels = model_channels * channel_mult[i]
            
        # Cross-attention for style guidance
        if num_head_channels == -1:
            dim_head = layer_channels // num_heads if num_heads != -1 else layer_channels
        else:
            dim_head = num_head_channels
            
        # Create style attention module
        self.style_attention = SpatialTransformer(
            layer_channels,
            num_heads if num_heads != -1 else 8,
            dim_head,
            depth=1,
            context_dim=model_channels,
            use_checkpoint=use_checkpoint,
            disable_self_attn=False,
            use_linear=use_linear_in_transformer
        )
        
    def forward(self, x, hint, timesteps, context, location, style_ref=None, **kwargs):
        """
        Forward pass with optional style guidance
        
        Args:
            x: Input tensor
            hint: Control input 
            timesteps: Diffusion timesteps
            context: Text context
            location: Location embedding
            style_ref: Optional style reference image
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        # Process hint as usual
        guided_hint = self.input_hint_block(hint, emb, context)
        
        # Extract style features if style reference is provided
        style_features = None
        if style_ref is not None:
            style_features = self.style_encoder(style_ref)
        
        outs = []
        locs = []
        
        loc_input = location.float().unsqueeze(1)
        
        h = x.type(self.dtype)
        for idx, (module, zero_conv, loc_module) in enumerate(zip(
            self.input_blocks, self.zero_convs, self.loc_blocks
        )):
            # Regular processing
            if guided_hint is not None and idx == 0:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            
            # Apply style guidance at specified layer
            if style_features is not None and idx == self.style_attention_layer:
                # Resize style features to match current feature map size
                if style_features.shape[2:] != h.shape[2:]:
                    style_features = F.interpolate(
                        style_features, 
                        size=h.shape[2:], 
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Adjust style features channel dimension if needed
                if style_features.shape[1] != h.shape[1]:
                    style_features = nn.Conv2d(
                        style_features.shape[1], 
                        h.shape[1], 
                        kernel_size=1
                    ).to(h.device)(style_features)
                
                # Apply cross-attention for style
                style_guided = self.style_attention(h, context=style_features.flatten(2).transpose(1, 2))
                
                # Mix with original features based on style strength
                h = h + self.style_attention_strength * (style_guided - h)
            
            # Standard processing continues
            loc_input, loc_zero = loc_module(loc_input, emb.unsqueeze(1))
            locs.append(loc_zero)
            outs.append(zero_conv(h, emb, context))
        
        # Process middle block normally
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        locs.append(self.loc_middle_block(loc_input, emb.unsqueeze(1))[1])
        
        return outs, locs
