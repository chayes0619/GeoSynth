import torch
import einops
from ..scripts.geosynth import ControlLDM


class StyleGuidedControlLDM(ControlLDM):
    """
    Extended ControlLDM that accepts a style reference image.
    """
    def __init__(
        self,
        control_stage_config,
        control_key,
        only_mid_control,
        style_key=None,  # Key for style reference
        *args, **kwargs
    ):
        super().__init__(
            control_stage_config=control_stage_config,
            control_key=control_key,
            only_mid_control=only_mid_control,
            *args, **kwargs
        )
        self.style_key = style_key  # Store key for style reference
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        """
        Extended input processing to handle style reference
        """
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        # Get control input 
        control = batch[self.control_key]
        
        # Get style reference if available
        style_ref = None
        if self.style_key is not None and self.style_key in batch:
            style_ref = batch[self.style_key]
        
        # Get location embedding
        location = batch["location"]
        
        # Apply batch size limit if specified
        if bs is not None:
            control = control[:bs]
            if style_ref is not None:
                style_ref = style_ref[:bs]
        
        # Move to device
        control = control.to(self.device)
        location = location.to(self.device)
        
        # Process style reference if available
        if style_ref is not None:
            style_ref = style_ref.to(self.device)
            style_ref = einops.rearrange(style_ref, "b h w c -> b c h w")
            style_ref = style_ref.to(memory_format=torch.contiguous_format).float()
        
        # Process control
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        
        return x, dict(
            c_crossattn=[c],      # Text conditioning
            c_concat=[control],   # Control input
            c_style=style_ref,    # Style reference
            c_loc=location        # Location embedding
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        Extended apply_model with style reference
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        
        # Get text conditioning
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        
        # Forward pass through control model
        control, locs = self.control_model(
            x=x_noisy,
            hint=torch.cat(cond["c_concat"], 1),
            timesteps=t,
            context=cond_txt,
            location=cond["c_loc"],
            style_ref=cond["c_style"]  # Pass style reference
        )
        
        # Apply scales
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        locs = [c * scale for c, scale in zip(locs, self.loc_scales)]
        
        # Forward pass through diffusion model
        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=control,
            location=locs,
            only_mid_control=self.only_mid_control,
        )
        
        return eps
    
    @torch.no_grad()
    def log_images(self, batch, *args, **kwargs):
        """
        Extended log_images to include style reference
        """
        log = super().log_images(batch, *args, **kwargs)
        
        # Add style reference to log if available
        if self.style_key is not None and self.style_key in batch:
            style_ref = batch[self.style_key]
            if len(style_ref) > 0:
                log["style_reference"] = style_ref * 2.0 - 1.0  # Convert to [-1, 1] range
        
        return log
