import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from GeoSynth.scripts.style_guided_ldm import StyleGuidedControlLDM
from GeoSynth.ControlNet.ldm.models.diffusion.ddim import DDIMSampler


def load_model(config_path, checkpoint_path):
    """Load the style-guided GeoSyth model"""
    config = OmegaConf.load(config_path)
    model = StyleGuidedControlLDM(**config.model.params)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])
    model = model.cuda().eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess an image for the model"""
    if isinstance(image_path, str):
        image = np.array(Image.open(image_path))
    else:
        image = np.array(image_path)
    
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    # Ensure it's RGB (not RGBA)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Convert to tensor [C, H, W] with values in [0, 1]
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor


def generate_with_style_guidance(
    model,
    input_image_path,
    style_image_path=None,
    prompt="",
    num_samples=1,
    ddim_steps=50,
    ddim_eta=0.0,
    scale=9.0
):
    """
    Generate an image with style guidance
    
    Args:
        model: The StyleGuidedControlLDM model
        input_image_path: Path to the input image (hint)
        style_image_path: Optional path to the style reference image
        prompt: Text prompt for conditioning
        num_samples: Number of samples to generate
        ddim_steps: Number of DDIM sampling steps
        ddim_eta: DDIM eta parameter
        scale: Unconditional guidance scale
        
    Returns:
        Generated images
    """
    # Initialize sampler
    sampler = DDIMSampler(model)
    
    # Load and preprocess input image
    control = preprocess_image(input_image_path).cuda()
    
    # Load and preprocess style reference if provided
    style_ref = None
    if style_image_path:
        style_ref = preprocess_image(style_image_path).cuda()
    
    # Generate conditioning
    c = model.get_learned_conditioning([prompt] * num_samples)
    
    # Create dummy location
    h, w = control.shape[2], control.shape[3]
    location = torch.zeros((num_samples, 256), device=model.device)
    
    # Create conditioning dictionary
    cond = {
        "c_crossattn": [c],
        "c_concat": [control],
        "c_style": style_ref,
        "c_loc": location
    }
    
    # Create unconditional conditioning
    uc = model.get_unconditional_conditioning(num_samples)
    uc_full = {
        "c_crossattn": [uc],
        "c_concat": [control],
        "c_style": style_ref,
        "c_loc": location
    }
    
    # Get latent shape
    shape = (model.channels, h // 8, w // 8)
    
    # Sample
    samples, _ = sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_full,
        eta=ddim_eta
    )
    
    # Decode
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    return x_samples


def generate_tile_grid(
    model,
    input_images,
    style_image=None,
    prompt="",
    rows=3,
    cols=3,
    generation_order="left-to-right"
):
    """
    Generate a grid of consistent tiles
    
    Args:
        model: The StyleGuidedControlLDM model
        input_images: List of input images for each tile
        style_image: Optional initial style reference
        prompt: Text prompt
        rows, cols: Grid dimensions
        generation_order: Generation strategy
        
    Returns:
        Generated image grid
    """
    generated_tiles = []
    
    # Determine generation order
    if generation_order == "left-to-right":
        order = [(r, c) for r in range(rows) for c in range(cols)]
    elif generation_order == "spiral":
        # Complex spiral order logic...
        order = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        order = order[:rows*cols]  # Truncate if needed
    else:
        raise ValueError(f"Unknown generation order: {generation_order}")
    
    # Initialize grid to track generated tiles
    tile_grid = [[None for _ in range(cols)] for _ in range(rows)]
    
    # Generate tiles in specified order
    for r, c in order:
        idx = r * cols + c
        input_image = input_images[idx] if idx < len(input_images) else input_images[0]
        
        # Determine which adjacent tile to use for style reference
        current_style = None
        
        if c > 0 and tile_grid[r][c-1] is not None:  # Use left tile
            current_style = tile_grid[r][c-1].cpu()
        elif r > 0 and tile_grid[r-1][c] is not None:  # Use top tile
            current_style = tile_grid[r-1][c].cpu()
        elif style_image is not None and len(generated_tiles) == 0:
            # Use provided initial style for first tile
            current_style = style_image
        
        # Generate the tile
        result = generate_with_style_guidance(
            model=model,
            input_image_path=input_image,
            style_image_path=current_style,
            prompt=prompt
        )
        
        # Store the generated tile
        generated_tiles.append(result)
        tile_grid[r][c] = result
    
    # Visualize the grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c] if rows > 1 and cols > 1 else axes[c] if rows == 1 else axes[r]
            tile = tile_grid[r][c][0].cpu().permute(1, 2, 0).numpy()
            ax.imshow(tile)
            ax.axis('off')
    
    plt.tight_layout()
    return fig, tile_grid


if __name__ == "__main__":
    # Example usage
    model = load_model(
        config_path="style_guided_config.yaml",
        checkpoint_path="path/to/checkpoint.ckpt"
    )
    
    # Generate a single image with style guidance
    result = generate_with_style_guidance(
        model=model,
        input_image_path="example_input.png",
        style_image_path="example_style.png",
        prompt="satellite imagery of urban area"
    )
    
    # Convert to image and save
    result_np = result[0].cpu().permute(1, 2, 0).numpy()
    result_np = (result_np * 255).astype(np.uint8)
    Image.fromarray(result_np).save("style_guided_result.png")
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(result_np)
    plt.axis('off')
    plt.title("Generated Image with Style Guidance")
    plt.show()
