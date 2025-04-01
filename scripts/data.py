import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, prompt_path, location_embeds_path, adjacent_satellite_dir=None):
        self.data = json.load(open(prompt_path, "rt"))
        self.loc_embeds = np.load(location_embeds_path)
        self.adjacent_satellite_dir = adjacent_satellite_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]
        loc_ind = item["class"]

        source = cv2.imread("../" + source_filename)
        target = cv2.imread("../" + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        result = dict(
            jpg=target, txt=prompt, hint=source, location=self.loc_embeds[loc_ind]
        )
        
        # Load adjacent satellite image if directory is provided
        if self.adjacent_satellite_dir is not None and "adjacent_satellite" in item:
            # Use the path already in the JSON data
            adjacent_filename = "../" + item["adjacent_satellite"]
            
            # Check if the adjacent image exists
            if os.path.exists(adjacent_filename):
                adjacent = cv2.imread(adjacent_filename)
                adjacent = cv2.cvtColor(adjacent, cv2.COLOR_BGR2RGB)
                
                # Normalize adjacent images to [0, 1] like source images
                adjacent = adjacent.astype(np.float32) / 255.0
                
                result["adjacent_satellite"] = adjacent
            else:
                # If no adjacent image exists, create a blank image
                result["adjacent_satellite"] = np.zeros_like(source)
                
                # Log missing adjacent images for debugging
                print(f"Warning: Adjacent satellite image not found: {adjacent_filename}")
        
        return result