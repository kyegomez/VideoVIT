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
