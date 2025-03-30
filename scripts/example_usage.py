import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from GeoSynth.scripts.geosynth import TileAwareControlLDM

def load_model(config_path, checkpoint_path):
    """Load the TileAwareControlLDM model from config and checkpoint"""
    from omegaconf import OmegaConf
    from GeoSynth.ControlNet.ldm.util import instantiate_from_config
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Update model class in config to use TileAwareControlLDM
    config.model.target = "GeoSynth.scripts.geosynth.TileAwareControlLDM"
    
    # Create model
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    
    return model

def process_large_image(model, image_path, prompt, output_path, tile_size=512, overlap=0.25):
    """Process a large image using tiled generation"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms.ToTensor()(image)
    
    # Move to device
    model = model.to("cuda")
    image_tensor = image_tensor.to("cuda")
    
    # Set tile overlap
    model.tile_overlap = overlap
    
    # Process image with tiled generation
    with torch.no_grad():
        processed_image = model.process_tiles(
            image_tensor,
            prompt,
            tile_size=tile_size,
            batch_size=1  # Adjust based on GPU memory
        )
    
    # Convert to PIL and save
    processed_pil = transforms.ToPILImage()(processed_image.cpu())
    processed_pil.save(output_path)
    
    print(f"Processed image saved to {output_path}")
    
    return processed_pil

def main():
    # Paths
    config_path = "configs/geosynth.yaml"
    checkpoint_path = "checkpoints/geosynth.ckpt"
    image_path = "large_image.png"
    output_path = "processed_large_image.png"
    
    # Prompt
    prompt = "Create a detailed topographic map from this input"
    
    # Load model
    model = load_model(config_path, checkpoint_path)
    
    # Process large image
    process_large_image(model, image_path, prompt, output_path, tile_size=1024, overlap=0.25)

if __name__ == "__main__":
    main()