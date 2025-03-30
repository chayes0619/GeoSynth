# Style-Guided GeoSyth

This extension adds style guidance to the GeoSyth model using cross-attention, allowing for consistent style across adjacent tiles.

## Overview

The style-guided extension adds two simple components:

1. **Style Encoder**: A lightweight encoder that extracts style features from reference images
2. **Style Cross-Attention**: A cross-attention layer that guides the generation process to match the reference style

This approach is much simpler than full feature integration while still providing good style consistency between adjacent tiles.

## Files

- `style_guided_controlnet.py`: The extended ControlNet with style guidance
- `style_guided_ldm.py`: The extended ControlLDM that handles style references
- `style_guided_config.yaml`: Configuration file for the style-guided model
- `inference_example.py`: Example code for inference with style guidance

## Installation

1. Place the Python files in your GeoSyth project's `scripts` directory:
   ```
   GeoSynth/
   └── scripts/
       ├── style_guided_controlnet.py
       ├── style_guided_ldm.py
       └── inference_example.py
   ```

2. Use the provided configuration file or update your existing config to use the style-guided modules.

## Usage

### Basic Generation with Style Guidance

```python
from GeoSynth.scripts.style_guided_ldm import StyleGuidedControlLDM
from GeoSynth.ControlNet.ldm.models.diffusion.ddim import DDIMSampler
import torch
from PIL import Image
import numpy as np

# Load model
model = StyleGuidedControlLDM.load_from_checkpoint("path/to/checkpoint.ckpt")
model = model.cuda().eval()

# Load input and style images
input_image = torch.from_numpy(np.array(Image.open("input.png"))).float() / 255.0
input_image = input_image.permute(2, 0, 1).unsqueeze(0).cuda()

style_image = torch.from_numpy(np.array(Image.open("style.png"))).float() / 255.0
style_image = style_image.permute(2, 0, 1).unsqueeze(0).cuda()

# Initialize sampler
sampler = DDIMSampler(model)

# Create conditioning
c = model.get_learned_conditioning(["satellite imagery of urban area"])
location = torch.zeros((1, 256), device=model.device)

# Create conditioning dict
cond = {
    "c_crossattn": [c],
    "c_concat": [input_image],
    "c_style": style_image,
    "c_loc": location
}

# Generate
shape = (model.channels, input_image.shape[2] // 8, input_image.shape[3] // 8)
samples, _ = sampler.sample(50, 1, shape, cond, verbose=False)
x_samples = model.decode_first_stage(samples)
result = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

# Save result
result_np = result[0].cpu().permute(1, 2, 0).numpy() * 255
Image.fromarray(result_np.astype(np.uint8)).save("result.png")
```

### Generating Multiple Consistent Tiles

For multiple tiles, use the previously generated tile as the style reference for the next tile:

```python
# Generate first tile
first_tile = generate_tile(model, input_image_1, style_reference=None)

# Generate second tile using first tile as style reference
second_tile = generate_tile(model, input_image_2, style_reference=first_tile)

# Continue for additional tiles...
```

## Adjusting Style Influence

You can control how strongly the style is applied by adjusting the `style_attention_strength` parameter in the config file:

- Higher values (e.g., 2.0) will make the style more prominent
- Lower values (e.g., 0.5) will make the style more subtle

## Fine-tuning

You can start with your existing GeoSyth checkpoint and fine-tune with style references:

```python
# Load existing model
model = StyleGuidedControlLDM.load_from_checkpoint("original_checkpoint.ckpt", strict=False)

# Fine-tune with style references
# Your training loop here...
```

During training, provide both the control input and a style reference in your dataset.
