# Projets de Vision par Ordinateur

Ce dépôt contient une collection de projets de vision par ordinateur utilisant PyTorch et d'autres bibliothèques de deep learning.

## Structure du Projet

1. **Extract Features**
   - Extraction de caractéristiques d'images avec ViT
   - Visualisation des caractéristiques via PCA

2. **Image Retrieval**
   - Recherche d'images similaires dans une base de données
   - Utilisation de la similarité cosinus

3. **Image Classification**
   - Classification d'images avec ViT
   - Utilisation des 1000 classes d'ImageNet

4. **Anomaly Detection**
   - Détection d'anomalies dans les images
   - Comparaison avec une base de référence

5. **Object Detection**
   - Détection d'objets avec Faster R-CNN
   - Visualisation des boîtes englobantes

6. **Image Segmentation**
   - Segmentation sémantique avec DeepLabV3
   - Visualisation des masques de segmentation

7. **Image Similarity**
   - Comparaison de similarité entre images
   - Calcul de scores de similarité

8. **Object Tracking**
   - Suivi de personnes en temps réel
   - Utilisation de MediaPipe pour la détection de pose

## Prérequis Généraux

- Python 3.7+
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- matplotlib
- timm
- opencv-python
- mediapipe

## Installation

pip install torch torchvision pillow numpy matplotlib timm opencv-python mediapipe

## Structure des Dossiers

src/
├── 1 - Extract Features/
├── 2 - Image Retrival/
├── 3 - Image Classification/
├── 4 - Anomaly Detection/
├── 5 - Object Detection/
├── 6 - Image Segmentation/
├── 7 - Image Similarity/
└── 8 - Object Tracking/

## Utilisation

Chaque dossier contient son propre README avec des instructions spécifiques. Pour commencer, naviguez dans le dossier du projet qui vous intéresse et suivez les instructions du README correspondant.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à proposer une pull request.