import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Générer des données 3D
def generate_3d_data(n_samples=300):
    X, y = make_blobs(n_samples=n_samples, centers=2, 
                     n_features=3, random_state=42)
    return X, y

# Créer la grille pour la surface de décision
def make_meshgrid(x, y, z, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    z_min, z_max = z.min() - 1, z.max() + 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h),
                            np.arange(z_min, z_max, h))
    return xx, yy, zz

def plot_svm_3d():
    # Générer les données
    X, y = generate_3d_data()
    
    # Créer et entraîner le SVM
    clf = svm.SVC(kernel='rbf')
    clf.fit(X, y)
    
    # Créer la figure 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer les points
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, 
                        cmap='coolwarm', marker='o', s=50)
    
    # Créer une grille plus grossière pour la surface de décision
    xx, yy, zz = make_meshgrid(X[:, 0], X[:, 1], X[:, 2], h=0.5)
    
    # Prédire pour chaque point de la grille
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Tracer la surface de décision
    ax.contour3D(xx[:,:,0], yy[:,:,0], Z[:,:,0], 
                 levels=[0], alpha=0.5, cmap='coolwarm')
    
    # Configurer les étiquettes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SVM avec noyau RBF en 3D')
    
    # Ajouter une légende
    plt.colorbar(scatter, label='Classes')
    
    # Ajouter une rotation pour une meilleure visualisation
    ax.view_init(elev=20, azim=45)
    
    return fig

# Créer la visualisation
fig = plot_svm_3d()
plt.show()