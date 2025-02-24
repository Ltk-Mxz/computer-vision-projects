import torch
import os
from PIL import Image
from torchvision import transforms
import timm
from torch.nn.functional import cosine_similarity

# Fonction pour extraire les caractéristiques d'une image
def extract_features(image_path, model, transform):
    image = Image.open(image_path)
    img_tensor = transform(image).unsqueeze(0)  # Appliquer les transformations et ajouter une dimension batch
    with torch.no_grad():
        features = model.forward_features(img_tensor)  # Extraire les caractéristiques
        # Aplatir les caractéristiques en un vecteur 1D
        features = features.mean(dim=1)  # Moyenne sur la dimension spatiale
    return features

# Charger le modèle pré-entraîné ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Transformation pour adapter l'image au modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dossier contenant les images à comparer
image_folder = '../../assets'

# Extraire les caractéristiques de chaque image dans le dossier
image_features = {}
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    features = extract_features(image_path, model, transform)
    image_features[image_name] = features

# Image de requête
query_image_path = '../../assets/1.jfif'
query_features = extract_features(query_image_path, model, transform)

# Comparer la requête avec les autres images
similarities = {}
for image_name, features in image_features.items():
    # Calculer la similarité cosinus entre les vecteurs aplatis
    similarity = cosine_similarity(query_features.flatten(), features.flatten(), dim=0)
    similarities[image_name] = similarity.item()

# Trier les résultats par similarité décroissante
sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# Afficher les résultats
print(f"Les images les plus similaires à {query_image_path}:")
for image_name, similarity in sorted_results:
    print(f"{image_name}: Similarité = {similarity:.4f}")
