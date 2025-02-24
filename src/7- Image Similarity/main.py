import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch.nn.functional as F

# Charger le modèle ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Définir la transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    # Charger et transformer l'image
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0)
    
    # Extraire les features
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        # Prendre la moyenne des features pour obtenir un vecteur
        features = torch.mean(features, dim=1)
    return features

def compare_images(image1_path, image2_path):
    # Extraire les features
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    
    # Calculer la similarité cosinus
    similarity = F.cosine_similarity(features1, features2)
    return similarity.item()

# Chemins des images à comparer
image1_path = '../../assets/8.jpg'
image2_path = '../../assets/10.jpg'

# Calculer le score de similarité
similarity_score = compare_images(image1_path, image2_path)

# Afficher les images et le score
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(Image.open(image1_path))
plt.title("Image 1")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Image.open(image2_path))
plt.title("Image 2")
plt.axis('off')

# Afficher le score de similarité
plt.subplot(1, 3, 3)
plt.text(0.5, 0.5, f'Score de similarité:\n{similarity_score:.2f}', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Score de similarité entre les images : {similarity_score:.2f}")
# Le score varie de -1 (très différent) à 1 (identique)
