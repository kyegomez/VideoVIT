[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Video Vit
Open source implementation of a vision transformer that can understand Videos using max vit as a foundation. This uses max vit as the backbone vit and then packs the video tensor into a 4d tensor which is the input to the maxvit model. Implementing this because the new McVit came out and I need more practice. This is fully ready to train and I believe would perform amazingly.

## Installation
`$ pip install video-vit`

## Usage
```python
import torch
from video_vit.main import VideoViT

# Instantiate the VideoViT model with the specified parameters
model = VideoViT(
    num_classes=10,                 # Number of output classes
    dim=64,                         # Dimension of the token embeddings
    depth=(2, 2, 2),                # Depth of each stage in the model
    dim_head=32,                    # Dimension of the attention head
    window_size=7,                  # Size of the attention window
    mbconv_expansion_rate=4,        # Expansion rate of the Mobile Inverted Bottleneck block
    mbconv_shrinkage_rate=0.25,     # Shrinkage rate of the Mobile Inverted Bottleneck block
    dropout=0.1,                    # Dropout rate
    channels=3,                     # Number of input channels
)

# Create a random tensor with shape (batch_size, channels, frames, height, width)
x = torch.randn(1, 3, 10, 224, 224)

# Perform a forward pass through the model
output = model(x)

# Print the shape of the output tensor
print(output.shape)


```


# License
MIT
