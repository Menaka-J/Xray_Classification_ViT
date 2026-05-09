
import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image


class SPT(nn.Module):
    
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class TokenLearner(nn.Module):
   
    def __init__(self, dim, num_tokens=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x):
      
        attn = self.attention(x)                
        attn = torch.softmax(attn, dim=1)       
    
        tokens = torch.einsum('bnk,bnc->bkc', attn, x)
        return tokens


class ProposedViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.spt = SPT()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )
        dim = self.backbone.num_features
        self.token_learner = TokenLearner(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.spt(x)
        x = self.backbone.forward_features(x)
        x = x[:,1:,:]
        tokens = self.token_learner(x)
        x = tokens.mean(dim=1)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


metadata = torch.load("dataset_metadata.pth", map_location="cpu")
class_names = metadata["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_plain_vit(num_classes):
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load("plain_vit_best.pth", map_location=device))
    model.eval().to(device)
    return model

def load_proposed_vit(num_classes):
    model = ProposedViT(num_classes=num_classes)
    state = torch.load("proposed_vit_best.pth", map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])




st.title("X-ray Disease Classification — Plain vs Proposed ViT")

model_choice = st.selectbox("Choose model", ["Plain ViT", "Proposed ViT"])

model = None
load_error = None
try:
    if model_choice == "Plain ViT":
        model = load_plain_vit(num_classes=len(class_names))
    else:
        model = load_proposed_vit(num_classes=len(class_names))
except Exception as e:
    load_error = str(e)

if load_error:
    st.error("Error loading model:\n" + load_error)
    st.info("If this persists, copy the exact SPT and TokenLearner class definitions from your Kaggle notebook into this file.")
else:
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        st.write(f"### Predicted Disease: **{class_names[pred_idx]}**")
        #  bar chart
        st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})
