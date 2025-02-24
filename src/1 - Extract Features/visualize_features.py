# PCA: Principal Component Analysis

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les caractéristiques extraites (features.pt)
features = torch.load('features.pt')

# Appliquer PCA pour réduire à 2D
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features.squeeze().numpy())

# Visualiser les caractéristiques réduites
plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.title("Visualisation des caractéristiques extraites (PCA)")
plt.xlabel("Composant principal 1")
plt.ylabel("Composant principal 2")
plt.show()

# Cette visualisation te permet de mieux comprendre comment le modèle organise les informations extraites de l'image.
# Par exemple, tu pourrais voir des clusters qui correspondent à des objets similaires dans l'image, ou voir des patterns intéressants concernant 
# la façon dont le modèle perçoit les différentes parties de l'image.