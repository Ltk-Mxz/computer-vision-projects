# Extraction de Caractéristiques d'Images

## Description
Ce projet utilise un modèle Vision Transformer (ViT) pour extraire des caractéristiques représentatives d'images. Ces caractéristiques peuvent être utilisées pour diverses applications d'apprentissage automatique.

## Fonctionnalités
- Extraction de caractéristiques à partir d'images
- Visualisation des caractéristiques extraites via PCA
- Sauvegarde des caractéristiques dans un fichier features.pt

## Utilisation
1. Placez votre image dans le dossier assets
2. Exécutez `main.py` pour extraire les caractéristiques
3. Utilisez `visualize_features.py` pour visualiser les caractéristiques extraites

## Dépendances
- torch
- PIL
- timm
- sklearn
- matplotlib 