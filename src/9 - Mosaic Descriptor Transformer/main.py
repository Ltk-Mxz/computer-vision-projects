import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class MosaicPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class MosaicDescriptorTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = MosaicPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 
            (img_size // patch_size) ** 2, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        # Descriptor MLP
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Ajouter position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Générer descripteurs
        descriptors = self.descriptor_mlp(x)
        
        return descriptors

def visualize_mosaic_patches(image_path, patch_size=16):
    # Charger et transformer l'image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image)
    
    # Créer la visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Image originale
    ax1.imshow(image)
    ax1.set_title("Image originale")
    ax1.axis('off')
    
    # Image avec grille de patches
    ax2.imshow(image)
    ax2.set_title(f"Mosaïque (patches {patch_size}x{patch_size})")
    
    # Dessiner la grille
    for i in range(0, 224, patch_size):
        ax2.axhline(y=i, color='r', linestyle='-', alpha=0.3)
        ax2.axvline(x=i, color='r', linestyle='-', alpha=0.3)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialiser le modèle
    model = MosaicDescriptorTransformer()
    model.eval()
    
    # Préparer la transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Charger une image
    image_path = "../../assets/11.jpg"
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0)
    
    # Visualiser la mosaïque
    visualize_mosaic_patches(image_path)
    
    # Extraire les descripteurs
    with torch.no_grad():
        descriptors = model(img_tensor)
    
    print(f"Forme des descripteurs: {descriptors.shape}")
    print("Descripteurs extraits avec succès!")

if __name__ == "__main__":
    main() 