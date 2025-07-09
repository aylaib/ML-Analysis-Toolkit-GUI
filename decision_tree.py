# Import des bibliothèques nécessaires
import numpy as np          # Pour les calculs numériques
import pandas as pd         # Pour la manipulation des données
import math                # Pour les opérations mathématiques
from typing import Optional, Dict, List  # Pour le typage statique
import pydotplus           # Pour la visualisation de l'arbre
from collections import deque  # Pour le parcours en largeur de l'arbre


class TreeNode:
    """
    Classe représentant un nœud dans l'arbre de décision.
    Chaque nœud contient une décision (feature) ou une prédiction (output).
    """
    def __init__(self, feature: Optional[int], output: any):
        self.feature = feature      # Index de la feature utilisée pour la division (None pour les feuilles)
        self.children = {}          # Dictionnaire des nœuds enfants {valeur_feature: nœud_enfant}
        self.output = output        # Classe prédite par ce nœud
        self.index = -1             # Index unique pour identifier le nœud dans la visualisation
        self.samples = 0            # Nombre d'échantillons qui arrivent à ce nœud
        self.class_counts = {}      # Distribution des classes dans ce nœud {classe: nombre}
        self.split_value = None     # Valeur de séparation pour les features continues
        
    def add_child(self, feature_value: any, node: 'TreeNode'):
        """Ajoute un nœud enfant pour une valeur spécifique de la feature"""
        self.children[feature_value] = node


class DecisionTreeClassifier:
    """
    Implémentation d'un classificateur par arbre de décision.
    Supporte plusieurs critères de division (gain ratio, gini index) et
    différentes options de contrôle de la croissance de l'arbre.
    """
    
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_leaf: int = 1, 
                 min_gain: float = 0.0):
        self.__root = None          # Nœud racine de l'arbre
        self.max_depth = max_depth  # Profondeur maximale de l'arbre
        self.min_samples_leaf = min_samples_leaf  # Nombre minimum d'échantillons dans une feuille
        self.min_gain = min_gain    # Gain minimum requis pour effectuer une division
        self.feature_names = None   # Noms des features
        self.classes_ = None        # Classes uniques dans les données

    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Valide le format des données d'entrée.
        
        Args:
            X: Matrice des features
            y: Vecteur des labels
            
        Returns:
            tuple: (X, y) validés
            
        Raises:
            ValueError: Si les données ne sont pas valides
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Vérification des dimensions
        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D")
        if y.ndim != 1:
            raise ValueError("y doit être un tableau 1D")
        if len(X) != len(y):
            raise ValueError(f"Nombre d'échantillons incohérent: X({len(X)}) ≠ y({len(y)})")
            
        return X, y
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """
        Remplace les valeurs manquantes (NaN) par la moyenne de la colonne.
        
        Args:
            X: Matrice des features
            
        Returns:
            np.ndarray: Matrice sans valeurs manquantes
        """
        X = np.copy(X)  # Copie pour ne pas modifier les données originales
        for j in range(X.shape[1]):  # Pour chaque feature
            mask = np.isnan(X[:, j])  # Trouve les valeurs manquantes
            if mask.any():
                mean_val = np.nanmean(X[:, j])  # Calcule la moyenne sans NaN
                X[mask, j] = mean_val  # Remplace les NaN par la moyenne
        return X
    
    def _count_unique(self, Y: np.ndarray) -> Dict:
        """
        Compte le nombre d'occurrences de chaque classe.
        
        Args:
            Y: Vecteur des labels
            
        Returns:
            Dict: {classe: nombre_occurrences}
        """
        unique, counts = np.unique(Y, return_counts=True)
        return dict(zip(unique, counts))
    
    def _entropy(self, Y: np.ndarray) -> float:
        """
        Calcule l'entropie de Shannon d'un ensemble de labels.
        H(Y) = -Σ p(y) * log2(p(y))
        
        Args:
            Y: Vecteur des labels
            
        Returns:
            float: Valeur de l'entropie
        """
        freq_map = self._count_unique(Y)
        entropy_ = 0
        total = len(Y)
        for count in freq_map.values():
            p = count / total
            entropy_ += (-p) * math.log2(p)
        return entropy_
    
    def _gain_ratio(self, X: np.ndarray, Y: np.ndarray, feature: int) -> float:
        """
        Calcule le ratio de gain pour une feature.
        GainRatio = (H(Y) - H(Y|X)) / H(X)
        
        Args:
            X: Matrice des features
            Y: Vecteur des labels
            feature: Index de la feature à évaluer
            
        Returns:
            float: Ratio de gain
        """
        info_orig = self._entropy(Y)  # Entropie initiale
        info_f = 0  # Information conditionnelle
        split_info = 0  # Information de division
        values = np.unique(X[:, feature])  # Valeurs uniques de la feature
        
        # Pour chaque valeur possible de la feature
        for value in values:
            mask = X[:, feature] == value
            subset_size = np.sum(mask)
            if subset_size == 0:
                continue
                
            p = subset_size / len(Y)
            info_f += p * self._entropy(Y[mask])  # Entropie conditionnelle pondérée
            split_info += -p * math.log2(p)  # Information de division
            
        if split_info == 0:  # Évite division par zéro
            return 0
            
        info_gain = info_orig - info_f  # Gain d'information
        return info_gain / split_info  # Ratio de gain
    
    def _gini_index(self, Y: np.ndarray) -> float:
        """
        Calcule l'indice de Gini.
        Gini = 1 - Σ p(y)²
        
        Args:
            Y: Vecteur des labels
            
        Returns:
            float: Indice de Gini
        """
        freq_map = self._count_unique(Y)
        gini = 1
        total = len(Y)
        for count in freq_map.values():
            p = count / total
            gini -= p ** 2
        return gini
    
    def _gini_gain(self, X: np.ndarray, Y: np.ndarray, feature: int) -> float:
        """
        Calcule la réduction d'impureté de Gini après division.
        
        Args:
            X: Matrice des features
            Y: Vecteur des labels
            feature: Index de la feature à évaluer
            
        Returns:
            float: Gain de Gini
        """
        gini_orig = self._gini_index(Y)  # Gini initial
        gini_split = 0  # Gini après division
        values = np.unique(X[:, feature])
        
        for value in values:
            mask = X[:, feature] == value
            subset_size = np.sum(mask)
            if subset_size == 0:
                continue
                
            p = subset_size / len(Y)
            gini_split += p * self._gini_index(Y[mask])  # Gini pondéré des sous-ensembles
            
        return gini_orig - gini_split  # Réduction de l'impureté

    def _decision_tree(self, X: np.ndarray, Y: np.ndarray, 
                      features: List[int], level: int, 
                      metric: str) -> TreeNode:
        """
        Construit récursivement l'arbre de décision.
        
        Args:
            X: Matrice des features
            Y: Vecteur des labels
            features: Liste des features disponibles
            level: Niveau actuel dans l'arbre
            metric: Métrique pour évaluer les divisions ("gain_ratio" ou "gini_index")
            
        Returns:
            TreeNode: Nœud racine du (sous-)arbre construit
        """
        
        # Calcul des statistiques du nœud actuel
        class_counts = self._count_unique(Y)
        total_samples = len(Y)
        output = max(class_counts.items(), key=lambda x: x[1])[0]  # Classe majoritaire
        
        # Création du nœud
        node = TreeNode(None, output)
        node.samples = total_samples
        node.class_counts = class_counts
        
        # Critères d'arrêt
        if (self.max_depth is not None and level >= self.max_depth) or \
           total_samples <= self.min_samples_leaf or \
           len(features) == 0 or \
           len(class_counts) == 1:
            return node
        
        # Recherche de la meilleure feature pour la division
        best_gain = self.min_gain
        best_feature = None
        best_split_value = None
        
        # Évaluation de chaque feature
        for feature in features:
            if metric == "gain_ratio":
                gain = self._gain_ratio(X, Y, feature)
            else:  # gini_index
                gain = self._gini_gain(X, Y, feature)
                
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split_value = np.median(X[:, feature])
        
        # Si aucune division pertinente n'est trouvée
        if best_feature is None:
            return node
        
        # Configuration du nœud avec la meilleure feature
        node.feature = best_feature
        node.split_value = best_split_value
        
        # Division récursive sur les sous-ensembles
        values = np.unique(X[:, best_feature])
        remaining_features = features.copy()
        remaining_features.remove(best_feature)
        
        # Création des nœuds enfants
        for value in values:
            mask = X[:, best_feature] == value
            if np.sum(mask) >= self.min_samples_leaf:
                child = self._decision_tree(
                    X[mask], Y[mask],
                    remaining_features,
                    level + 1,
                    metric
                )
                node.add_child(value, child)
        
        return node

    def fit(self, X: np.ndarray, y: np.ndarray, 
            metric: str = "gain_ratio", 
            feature_names: Optional[List[str]] = None) -> 'DecisionTreeClassifier':
        """
        Entraîne l'arbre de décision sur les données fournies.
        
        Args:
            X: Matrice des features
            y: Vecteur des labels
            metric: Métrique pour évaluer les divisions
            feature_names: Noms des features (optionnel)
            
        Returns:
            self: Le classificateur entraîné
        """
        X, y = self._validate_data(X, y)
        X = self._handle_missing_values(X)
        
        if metric not in ["gain_ratio", "gini_index"]:
            raise ValueError("metric doit être 'gain_ratio' ou 'gini_index'")
        
        # Configuration des noms des features
        self.feature_names = feature_names or [f"X[{i}]" for i in range(X.shape[1])]
        
        # Encodage des labels si nécessaire
        if y.dtype == object:
            unique_labels = np.unique(y)
            self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.array([self.classes_[label] for label in y])
        else:
            self.classes_ = {i: i for i in np.unique(y)}
        
        # Construction de l'arbre
        features = list(range(X.shape[1]))
        self.__root = self._decision_tree(X, y, features, 0, metric)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour de nouvelles données.
        
        Args:
            X: Matrice des features à prédire
            
        Returns:
            np.ndarray: Vecteur des prédictions
        """
        if self.__root is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
            
        X = np.asarray(X)
        X = self._handle_missing_values(X)
        
        def predict_one(x: np.ndarray, node: TreeNode) -> any:
            """Prédit la classe pour un seul échantillon"""
            if not node.children:  # Si c'est une feuille
                return node.output
            
            value = x[node.feature]
            if value not in node.children:
                return node.output
                
            return predict_one(x, node.children[value])
        
        # Prédiction pour chaque échantillon
        y_pred = np.array([predict_one(x, self.__root) for x in X])
        
        # Conversion des labels numériques en labels originaux si nécessaire
        if self.classes_ is not None:
            inv_classes = {v: k for k, v in self.classes_.items()}
            y_pred = np.array([inv_classes[y] for y in y_pred])
            
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcule la précision du modèle sur un jeu de données.
        
        Args:
            X: Matrice des features
            y: Vecteur des vraies classes
            
        Returns:
            float: Score de précision entre 0 et 1
        """
        return np.mean(self.predict(X) == y)

    def export_tree_pdf(self, filename='decision_tree.pdf'):
        """
        Exporte une visualisation de l'arbre en format PDF.
        
        Args:
            filename: Nom du fichier PDF de sortie
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        import graphviz  # Pour la création du graphe
        
        if self.__root is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")
        
        dot = graphviz.Digraph(comment='Arbre de Décision')
        dot.attr(rankdir='TB')
        
        def add_nodes_edges(node, parent_id=None, edge_label=None):
            if node is None:
                return
                
            # Créer un ID unique pour ce nœud
            node_id = str(id(node))
            
            # Préparation du texte pour les distributions de classes
            class_dist = [f"{k}: {v}" for k, v in node.class_counts.items()]
            samples_text = f"\nSamples: {node.samples}"
            dist_text = "\n".join(class_dist)
            
            # Création du label du nœud
            if not node.children:  # Nœud feuille
                label = f"Classe: {node.output}\n{dist_text}{samples_text}"
                dot.node(node_id, label, shape='box')
            else:
                feature_name = self.feature_names[node.feature]
                threshold_text = f"\nSplit: {node.split_value:.2f}" if node.split_value is not None else ""
                label = f"{feature_name}{threshold_text}\n{dist_text}{samples_text}"
                dot.node(node_id, label, shape='diamond')
                
            # Ajout de l'arête depuis le parent
            if parent_id is not None:
                dot.edge(parent_id, node_id, edge_label)
                
            # Récursion sur les enfants
            for value, child in node.children.items():
                add_nodes_edges(child, node_id, f"{value}")
        
        # Commencer la construction avec le nœud racine
        add_nodes_edges(self.__root)
        
        # Sauvegarder et afficher le graphe
        dot.render(filename, view=True, cleanup=True)