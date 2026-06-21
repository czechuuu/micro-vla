import torch
import numpy as np
from typing import Union, List
from transformers import AutoImageProcessor, AutoModel

class DinoV3Wrapper():
    def __init__(self, config: "Config"):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.processor.size.height = config.image_size
        self.processor.size.width = config.image_size

        self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.model.config.image_size = config.image_size

        # Cleaned up the variable assignment and saved to self.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set the model to evaluation mode (critical for on-the-fly inference)
        self.model.eval()

    def __call__(self, images: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """
        Processes images on the fly into embeddings, matching the offline HDF5 dataset structure.
        """
        # 1. Process the raw images into PyTorch tensors
        processed_obs = self.processor(images=images, return_tensors="pt")

        # 2. Move the pixel values to the same device as the model
        pixel_values = processed_obs["pixel_values"].to(self.device)

        # 3. Perform the forward pass without tracking gradients
        with torch.no_grad():
            output = self.model(pixel_values)

            # 4. Extract the pooler output and convert back to a numpy array
            embeddings = output["pooler_output"].cpu().numpy()

        return embeddings