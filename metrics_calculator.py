import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """Calcule toutes les métriques de performance."""
    # Conversion des prédictions et des vraies valeurs
    y_true_str = y_true.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    y_pred_str = [p.decode('utf-8') if isinstance(p, bytes) else p for p in y_pred]
    
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true_str, y_pred_str)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    
    # Calcul des métriques
    metrics = {
        'confusion_matrix': cm,
        'TP': TP.sum(),
        'TN': TN.sum(),
        'FP': FP.sum(),
        'FN': FN.sum(),
        'precision': precision_score(y_true_str, y_pred_str, average='weighted', zero_division=1),
        'recall': recall_score(y_true_str, y_pred_str, average='weighted', zero_division=1),
        'f1': f1_score(y_true_str, y_pred_str, average='weighted', zero_division=1),
        'accuracy': accuracy_score(y_true_str, y_pred_str)
    }
    
    return metrics
