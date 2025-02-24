import torch
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
import timm
import numpy as np

# Charger un modèle Transformer (Vision Transformer)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Transformer l'image pour l'adapter au modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Fonction pour extraire les descripteurs d'une image
def extract_features(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Ajouter une dimension batch
    with torch.no_grad():
        features = model.forward_features(image)  # Extraction des caractéristiques
        # Moyenner les features pour obtenir un vecteur unique
        features = torch.mean(features, dim=1)  # Moyenne sur la dimension des patches
    return features

# Créer une base de caractéristiques "normales"
normal_images = ['../../assets/5.jpg', '../../assets/6.jpg', '../../assets/7.jpg']
normal_features = [extract_features(model, img) for img in normal_images]

# Calculer la moyenne des caractéristiques normales
mean_feature = torch.mean(torch.stack(normal_features), dim=0)

# Détecter une anomalie en comparant avec les caractéristiques normales
def detect_anomaly(model, image_path, normal_feature):
    test_feature = extract_features(model, image_path)
    # Calculer la similarité cosinus entre les vecteurs de caractéristiques
    similarity = cosine_similarity(test_feature, normal_feature.unsqueeze(0))
    return similarity.mean().item()  # Retourner la moyenne des similarités

# Charger une nouvelle image à tester
new_image = '../../assets/8.jpg'
score = detect_anomaly(model, new_image, mean_feature)

print(f"Score de similarité : {score:.4f}")

# Si le score est trop bas, c'est une anomalie !
if score < 0.8:
    print("⚠️ Anomalie détectée !")
else:
    print("✅ Image normale.")
