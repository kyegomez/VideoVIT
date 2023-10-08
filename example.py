import torch
from video_vit.main import VideoVit

# Actual testing
x = torch.randn(1, 3, 224, 224)

model = VideoVit(
    num_classes=1000, 
    dim=512, 
    depth=(2, 2, 2, 2), 
    dim_head=64
)

logits = model(x)  # (1, 1000)
print(logits)