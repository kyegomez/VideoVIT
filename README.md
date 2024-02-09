[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Video Vit
Open source implementation of a vision transformer that can understand Videos using max vit as a foundation.

## Installation
`$ pip install video-vit`

## Usage
```python
import torch
from video_vit.main import VideoViT


# Instantiate the model
model = VideoViT(
    num_classes=10, 
    dim=64, 
    depth=(2, 2, 2), 
    dim_head=32, 
    window_size=7, 
    mbconv_expansion_rate=4, 
    mbconv_shrinkage_rate=0.25, 
    dropout=0.1, 
    channels=3
)

# Create a random tensor with shape (batch_size, channels, frames, height, width)
x = torch.randn(1, 3, 10, 224, 224)

# Forward pass
output = model(x)

# Print the output
print(output.shape)


```


# License
MIT
