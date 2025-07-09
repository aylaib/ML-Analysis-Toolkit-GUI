import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

class DiabetesNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=[16, 8], dropout_rate=0.3):
        """
        Initialize Neural Network with configurable architecture
        
        Args:
        - input_size: Number of input features
        - hidden_layers: List of hidden layer sizes
        - dropout_rate: Regularization dropout rate
        """
        super(DiabetesNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Dynamic hidden layers construction
        for layer_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, layer_size),
                nn.ReLU(),
                nn.BatchNorm1d(layer_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = layer_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.hidden_layers = hidden_layers
    
    def forward(self, x):
        return self.model(x)
    
    def visualize_architecture(self, output_file='neural_network_architecture.png'):
        """
        Visualize neural network architecture using networkx
        
        Args:
        - output_file: Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        G = nx.DiGraph()
        
        # Determine input size dynamically
        input_size = self.model[0].in_features
        
        # Layer tracking
        layer_positions = {}
        
        # Add input layer nodes
        layer_positions[0] = [f'Input {i}' for i in range(input_size)]
        for node in layer_positions[0]:
            G.add_node(node)
        
        # Add hidden layers and connections
        prev_layer_nodes = layer_positions[0]
        for layer_idx, layer_size in enumerate(self.hidden_layers, 1):
            current_layer_nodes = [f'Layer {layer_idx} Node {j}' for j in range(layer_size)]
            layer_positions[layer_idx] = current_layer_nodes
            
            # Add current layer nodes
            for node in current_layer_nodes:
                G.add_node(node)
            
            # Connect previous layer to current layer
            for prev_node in prev_layer_nodes:
                for curr_node in current_layer_nodes:
                    G.add_edge(prev_node, curr_node)
            
            prev_layer_nodes = current_layer_nodes
        
        # Add output layer
        output_layer_nodes = ['Output']
        layer_positions[len(self.hidden_layers)+1] = output_layer_nodes
        G.add_node('Output')
        
        # Connect last hidden layer to output
        for prev_node in prev_layer_nodes:
            G.add_edge(prev_node, 'Output')
        
        # Compute positions
        pos = {}
        max_layer_size = max(len(nodes) for nodes in layer_positions.values())
        
        for layer, nodes in layer_positions.items():
            for i, node in enumerate(nodes):
                # Center the nodes vertically
                vertical_pos = i - (len(nodes) - 1) / 2
                pos[node] = (layer, vertical_pos * (1 / max_layer_size))
        
        # Draw the graph
        plt.title('Neural Network Architecture')
        
        # Draw edges first (in light gray)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True)
        
        # Draw nodes with color gradient
        node_colors = []
        for node in G.nodes():
            if 'Input' in node:
                node_colors.append('lightgreen')
            elif 'Layer' in node:
                node_colors.append('lightblue')
            else:  # Output
                node_colors.append('salmon')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Neural network architecture saved to {output_file}")

class NeuralNetworkTrainer:
    def __init__(self, data_path):
        """
        Initialize trainer with data path and preprocessing
        
        Args:
        - data_path: Path to ARFF file
        """
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self):
        """
        Load ARFF data, preprocess, and split into train/test sets
        """
        from scipy.io import arff
        import pandas as pd
        
        # Load ARFF file
        data, meta = arff.loadarff(self.data_path)
        df = pd.DataFrame(data)
        
        # Convert target to binary
        df['class'] = (df['class'] == b'tested_positive').astype(int)
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        # Standard scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split with stratification
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_model(self, epochs=200, learning_rate=0.001, batch_size=32, hidden_layers=[16, 8]):
        """
        Train neural network with hyperparameter tuning
        
        Args:
        - epochs: Number of training epochs
        - learning_rate: Learning rate for optimizer
        - batch_size: Batch size for training
        - hidden_layers: List of hidden layer sizes
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train.values).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.FloatTensor(self.y_test.values).reshape(-1, 1)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model, loss, and optimizer
        model = DiabetesNeuralNetwork(input_size=self.X_train.shape[1], 
                                      hidden_layers=hidden_layers)
        criterion = nn.BCELoss()  # Binary Cross Entropy
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = (model(X_test_tensor) > 0.5).float()
                val_accuracy = (val_pred == y_test_tensor).float().mean()
                
            train_losses.append(total_loss/len(train_loader))
            val_accuracies.append(val_accuracy.item())
            
            # Optional: Early stopping and logging
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {total_loss/len(train_loader):.4f}, Val Accuracy = {val_accuracy.item():.4f}")
        
        return model, train_losses, val_accuracies
    
    def evaluate_model(self, model):
        """
        Comprehensive model evaluation
        """
        # Predictions
        X_test_tensor = torch.FloatTensor(self.X_test)
        predictions = (model(X_test_tensor).detach().numpy() > 0.5).astype(int)
        true_labels = self.y_test.values
        
        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        print("\n--- Model Performance ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

def run_neural_network_analysis(file_path, hidden_layers=[16, 8], epochs=200):
    """
    Main function to run neural network analysis with customizable parameters
    
    Args:
    - file_path: Path to ARFF file
    - hidden_layers: List of hidden layer sizes
    - epochs: Number of training epochs
    """
    print("\n=== Neural Network Analysis for Diabetes Dataset ===")
    trainer = NeuralNetworkTrainer(file_path)
    
    # Train model
    print("Training Neural Network...")
    model, train_losses, val_accuracies = trainer.train_model(
        hidden_layers=hidden_layers, 
        epochs=epochs
    )
    
    # Evaluate model
    print("\nEvaluating Model Performance...")
    results = trainer.evaluate_model(model)
    
    # Plot training progress
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    
    # Visualize network architecture
    model.visualize_architecture('neural_network_architecture.png')
    
    return results
    