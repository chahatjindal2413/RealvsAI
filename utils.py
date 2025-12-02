import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ----- MODIFY If your model architecture is custom -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*24*24, 128),
            nn.ReLU(),
            nn.Linear(128, 2)   # 2 classes: Real vs Fake
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------

def load_model(model_path="model.pth"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

def preprocess(image):
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)
