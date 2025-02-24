import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Charger un modèle pré-entraîné (Faster R-CNN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Charger et transformer l'image
image_path = '../../assets/1.jfif'
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.ToTensor()  # Convertir en tenseur
])

img_tensor = transform(image).unsqueeze(0)

# Faire la prédiction
with torch.no_grad():
    prediction = model(img_tensor)

# Afficher l'image avec les boîtes englobantes
def afficher_detection(image, prediction, seuil=0.5):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for i in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][i].item()
        if score > seuil:  # Afficher uniquement si la confiance est supérieure au seuil
            box = prediction[0]['boxes'][i].numpy()
            label = prediction[0]['labels'][i].item()

            # Créer une boîte englobante
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'Objet {label} : {score:.2f}', color='red')

    plt.axis('off')
    plt.show()

afficher_detection(image, prediction)
