import torch
from PIL import Image
import requests
from torchvision import transforms
import timm

# Charger le modèle ViT pré-entraîné
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Charger les étiquettes des classes ImageNet
try:
    imagenet_labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.splitlines()
except requests.exceptions.RequestException as e:
    print(f"Erreur lors du téléchargement des étiquettes : {e}")
    exit(1)

# Transformer l'image pour l'adapter au modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner l'image
    transforms.ToTensor(),          # Convertir en tenseur
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliser l'image
])

# Charger l'image à classer
image_path = '../../assets/1.jfif'
image = Image.open(image_path)

# Appliquer les transformations
img_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension batch

# Effectuer la prédiction
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Trouver la classe prédite
top5_prob, top5_catid = torch.topk(probabilities, 5)  # Récupérer les 5 meilleures prédictions

print(f"📊 Classement pour l'image : {image_path}")
for i in range(5):
    print(f"{imagenet_labels[top5_catid[i]]}: {top5_prob[i].item():.4f}")
