import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as transforms

# -----------------------------
# 1. Load DINOv2 + processor
# -----------------------------
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base",
    do_rescale=False
)
dinov2 = AutoModel.from_pretrained("facebook/dinov2-base")
dinov2.eval()

# -----------------------------
# 2. Classifier Architecture
# -----------------------------
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 64),   # DINOv2-base output size
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

# -----------------------------
# 3. Load classifier weights
# -----------------------------
def load_classifier(path="model.pth"):
    model = Classifier()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

classifier = load_classifier()

# -----------------------------
# 4. Preprocess for DINOv2
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def preprocess(image):
    image = Image.open(image).convert("RGB")
    image = test_transform(image)
    return image

# -----------------------------
# 5. Predict function
# -----------------------------
def predict(image):
    # image → tensor shape [3,128,128]
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        feats = dinov2(**inputs).pooler_output  # [1,768]
        out = classifier(feats)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()
