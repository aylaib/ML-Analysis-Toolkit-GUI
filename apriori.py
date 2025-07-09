# apriori.py

import numpy as np
import pandas as pd
from itertools import combinations

class Apriori:
    def __init__(self, min_support=0.3, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        
    def _discretize_data(self, df):
        """Discrétise les données numériques en catégories."""
        discretized_data = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Mapping des noms de colonnes (ajustez selon votre fichier ARFF)
        column_mapping = {
            'preg': 'Pregnancies',
            'plas': 'Glucose',
            'pres': 'BloodPressure',
            'skin': 'SkinThickness',
            'insu': 'Insulin',
            'mass': 'BMI',
            'pedi': 'DiabetesPedigree',
            'age': 'Age',
            'class': 'Class'
        }
        
        # Définition des seuils pour chaque attribut
        thresholds = {
            'preg': [0, 2, 4, 6, 17],
            'plas': [0, 95, 140, 200],
            'pres': [0, 80, 90, 100, 122],
            'skin': [0, 20, 30, 40, 100],
            'insu': [0, 80, 155, 300, 846],
            'mass': [0, 25, 30, 35, 67],
            'pedi': [0, 0.5, 1, 2, 2.42],
            'age': [21, 30, 40, 50, 81]
        }
        
        # Labels pour chaque catégorie
        labels = {
            'preg': ['très_bas', 'bas', 'moyen', 'élevé'],
            'plas': ['bas', 'normal', 'élevé'],
            'pres': ['bas', 'normal', 'élevé', 'très_élevé'],
            'skin': ['très_fin', 'fin', 'moyen', 'épais'],
            'insu': ['très_bas', 'bas', 'normal', 'élevé'],
            'mass': ['normal', 'surpoids', 'obèse', 'très_obèse'],
            'pedi': ['faible', 'moyen', 'élevé', 'très_élevé'],
            'age': ['jeune', 'adulte', 'mûr', 'senior']
        }
        
        # Discrétisation de chaque colonne numérique
        for column in numeric_columns:
            if column in thresholds:
                discretized_data[column] = pd.cut(
                    df[column],
                    bins=thresholds[column],
                    labels=labels[column],
                    include_lowest=True
                )
        
        # Conversion des transactions en format lisible
        transactions = []
        for _, row in discretized_data.iterrows():
            transaction = []
            for column in discretized_data.columns:
                if pd.notna(row[column]):
                    # Utilise le nom original de la colonne
                    feature_name = column_mapping.get(column, column)
                    if column == 'class':
                        transaction.append(str(row[column]))
                    else:
                        transaction.append(f"{feature_name}={row[column]}")
            transactions.append(transaction)
            
        return transactions
    
    def _calculate_support(self, itemset, transactions):
        """Calcule le support d'un itemset."""
        count = sum(1 for transaction in transactions if all(item in transaction for item in itemset))
        return count / len(transactions)
    
    def _find_frequent_itemsets(self, transactions):
        """Trouve tous les itemsets fréquents."""
        # Création des itemsets de taille 1
        items = set(item for transaction in transactions for item in transaction)
        itemsets = [{item} for item in items]
        k = 1
        
        while itemsets:
            # Calcul du support pour les itemsets actuels
            frequent_k = {}
            for itemset in itemsets:
                support = self._calculate_support(itemset, transactions)
                if support >= self.min_support:
                    frequent_k[frozenset(itemset)] = support
            
            # Stockage des itemsets fréquents
            if frequent_k:
                self.frequent_itemsets[k] = frequent_k
                
            # Génération des candidats pour la prochaine itération
            k += 1
            itemsets = self._generate_candidates(frequent_k.keys(), k)
    
    def _generate_candidates(self, prev_itemsets, k):
        """Génère les candidats pour la prochaine itération."""
        candidates = set()
        prev_itemsets = list(prev_itemsets)
        
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                itemset1 = set(prev_itemsets[i])
                itemset2 = set(prev_itemsets[j])
                union = itemset1.union(itemset2)
                if len(union) == k:
                    candidates.add(frozenset(union))
        
        return [set(candidate) for candidate in candidates]
    
    def _generate_rules(self, transactions):
        """Génère les règles d'association."""
        for k, frequent_k in self.frequent_itemsets.items():
            if k < 2:
                continue
                
            for itemset, support in frequent_k.items():
                itemset = set(itemset)
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = set(antecedent)
                        consequent = itemset - antecedent
                        
                        ant_support = self._calculate_support(antecedent, transactions)
                        confidence = support / ant_support
                        
                        if confidence >= self.min_confidence:
                            lift = confidence / self._calculate_support(consequent, transactions)
                            self.rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
    
    def fit(self, df):
        """Entraîne le modèle Apriori sur les données."""
        print("Discrétisation des données...")
        transactions = self._discretize_data(df)
        print(f"Nombre de transactions: {len(transactions)}")
        
        print("Recherche des itemsets fréquents...")
        self._find_frequent_itemsets(transactions)
        print(f"Nombre d'itemsets fréquents trouvés: {sum(len(itemsets) for itemsets in self.frequent_itemsets.values())}")
        
        print("Génération des règles d'association...")
        self._generate_rules(transactions)
        print(f"Nombre de règles générées: {len(self.rules)}")
        
        # Trie les règles par confiance décroissante
        self.rules.sort(key=lambda x: (-x['confidence'], -x['support']))
    
    def get_insights(self):
        """Retourne les insights principaux de l'analyse."""
        diabetes_rules = []
        risk_factors = set()
        
        for rule in self.rules:
            # Cherche les règles liées au diabète
            consequent_str = str(list(rule['consequent'])[0])
            if 'tested_positive' in consequent_str or 'tested_negative' in consequent_str:
                diabetes_rules.append(rule)
                # Collecte les facteurs de risque
                for item in rule['antecedent']:
                    if any(level in item for level in ['élevé', 'très_élevé', 'obèse']):
                        risk_factors.add(item)
        
        return {
            'total_rules': len(self.rules),
            'diabetes_rules': diabetes_rules[:10],  # Top 10 règles liées au diabète
            'risk_factors': list(risk_factors)
        }