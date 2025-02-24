import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np

# Classes que DeepLabV3 peut détecter
CLASSES = [
    'arrière-plan', 'avion', 'vélo', 'oiseau', 'bateau', 'bouteille',
    'bus', 'voiture', 'chat', 'chaise', 'vache', 'table',
    'chien', 'cheval', 'moto', 'personne', 'plante en pot',
    'mouton', 'canapé', 'train', 'moniteur/TV'
]

# Charger un modèle de segmentation (DeepLabV3)
model = deeplabv3_resnet50(weights='DEFAULT')
model.eval()

# Charger et transformer l'image
image_path = '../../assets/10.jpg'
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

img_tensor = transform(image).unsqueeze(0)

# Faire la prédiction
with torch.no_grad():
    output = model(img_tensor)['out']

# Convertir la sortie en image
segmentation_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Créer une figure avec 3 sous-plots
plt.figure(figsize=(15, 5))

# Image originale
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Image originale")
plt.axis('off')

# Masque de segmentation
plt.subplot(1, 3, 2)
plt.imshow(segmentation_mask, cmap='jet')
plt.title("Masque de segmentation")
plt.axis('off')

# Liste des classes détectées
plt.subplot(1, 3, 3)
plt.text(0.1, 0.5, "Classes détectées:\n\n" + 
         "\n".join([f"- {CLASSES[i]}" for i in np.unique(segmentation_mask) if i < len(CLASSES)]),
         fontsize=10, verticalalignment='center')
plt.axis('off')

plt.tight_layout()
plt.show()
