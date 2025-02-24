# Ce code sert à extraire des informations représentatives d'une image (sous forme de vecteurs de caractéristiques), afin de pouvoir les utiliser dans des applications d'apprentissage automatique. Les caractéristiques extraites peuvent ensuite être utilisées pour des tâches comme :

# - Classification d'images : Identifier à quelle catégorie l'image appartient.
# - Recherche d'images similaires : Trouver des images ayant des caractéristiques similaires.
# - Visualisation et analyse : Analyser ce que le modèle a appris et comment il "comprend" l'image.
# - Le fichier features.pt contient ces caractéristiques sous une forme qui peut être réutilisée dans des modèles futurs ou pour d'autres applications d'IA.

import torch
from PIL import Image
import requests
from torchvision import transforms
import timm

# Charger l'image depuis un fichier local
image_path = '../../assets/10.jpg'
image = Image.open(image_path)

# Transformation : redimensionnement et conversion en tenseur
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adapter la taille au modèle
    transforms.ToTensor(),          # Convertir en tenseur
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation
])

# Appliquer les transformations à l'image
img_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension batch

# le ViT fonctionne ainsi :
# - Divise l’image en patches.
# - Transforme chaque patch en un vecteur d’embedding.
# - Passe ces vecteurs à un modèle Transformer pour apprendre les relations.
# - Utilise un vecteur global pour classer ou effectuer une autre tâche sur l’image.

# Charger un modèle pré-entraîné ViT (Vision Transformer)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()  # Mettre le modèle en mode évaluation

# Extraire les caractéristiques
with torch.no_grad():
    features = model.forward_features(img_tensor)

# Afficher les caractéristiques extraites
print(f"Dimensions des caractéristiques extraites : {features.shape}")

# Sauvegarder les caractéristiques dans un fichier
torch.save(features, 'features.pt')
print("Les caractéristiques ont été sauvegardées dans 'features.pt'")

# [ Output ]
# Dimensions des caractéristiques extraites : torch.Size([1, 197, 768])
# 1 : Cela correspond à la taille du batch. Ici, une seule image dans mon batch, d'où la taille 1.
# 197 : Cela fait référence au nombre de patches dans l'image (les patches sont des morceaux de l'image). Le modèle ViT découpe l'image en 16x16 pixels, et dans ce cas, il a produit 197 patches, y compris un token [CLS] (qui sert à résumer l'image).
# 768 : Il s'agit de la dimension de l'espace d'embedding pour chaque patch. Chaque patch est transformé en un vecteur de taille 768.