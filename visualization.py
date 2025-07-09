import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, classes, k):
    """Affiche la matrice de confusion sous forme de heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de confusion pour k={k}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()

def plot_metrics(k_values, metrics_dict):
    """Affiche les courbes des différentes métriques avec des couleurs distinctes."""
    plt.figure(figsize=(12, 8))
    
    # Définir des couleurs spécifiques pour chaque métrique
    colors = {
        'Précision': 'blue',
        'Rappel': 'yellow',
        'F1-Score': 'green',
        'Accuracy': 'red'
    }
    
    # Tracer les courbes avec des styles améliorés
    for metric_name, values in metrics_dict.items():
        plt.plot(k_values, values, 
                marker='o',
                label=metric_name,
                color=colors[metric_name],
                linewidth=2,
                markersize=8)
        
        # Ajouter les valeurs sur les points
        for k, value in zip(k_values, values):
            plt.annotate(f'{value:.2f}', 
                        (k, value),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
    
    plt.xlabel('Valeur de K', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Évolution des métriques en fonction de K', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
