import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import STGCN, STChebNet, STGAT, STSGConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch_geometric
from torch_geometric.data import Data, Dataset

def load_connectivity_data(path):
    matrices = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            matrix = np.loadtxt(os.path.join(path, file))
            matrices.append(torch.FloatTensor(matrix))
    return matrices

def main():
    # Load connectivity matrices
    connectivity_matrices = load_connectivity_data('data/connectivity')
    
    # Prepare data for k-fold cross validation
    X = torch.stack(connectivity_matrices)
    y = torch.zeros(len(connectivity_matrices))  # Replace with actual labels here

    # Define data loader as ConnectomeDataset
    class ConnectomeDataset(Dataset):
        def __init__(self, connectivity_path, timeseries_path):
            super().__init__()
            self.connectivity_matrices = {}
            self.timeseries_matrices = {}
            self.subjects = []
            
            # Load connectivity matrices (edge_index, edge_attr)
            for file in os.listdir(connectivity_path):
                if file.endswith('.txt'):
                    subject_id = file.split('sub')[1].split('.')[0]
                    matrix = np.loadtxt(os.path.join(connectivity_path, file))
                    self.connectivity_matrices[subject_id] = torch.FloatTensor(matrix)
                    self.subjects.append(subject_id)
            
            # Load corresponding timeseries (node features)
            for subject_id in self.subjects:
                ts_file = f'sub{subject_id}.txt'
                if os.path.exists(os.path.join(timeseries_path, ts_file)):
                    matrix = np.loadtxt(os.path.join(timeseries_path, ts_file))
                    self.timeseries_matrices[subject_id] = torch.FloatTensor(matrix)
                else:
                    raise ValueError(f"Missing timeseries data for subject {subject_id}")
            
            self.subjects.sort()
            
        def len(self):
            return len(self.subjects)
            
        def get(self, idx):
            subject_id = self.subjects[idx]
            
            # Convert adjacency matrix to edge_index and edge_attr
            adj_matrix = self.connectivity_matrices[subject_id]
            edge_index = (adj_matrix > 0).nonzero().t()
            edge_attr = adj_matrix[edge_index[0], edge_index[1]].unsqueeze(1)
            
            # Node features from timeseries
            x = self.timeseries_matrices[subject_id]
            
            # Create PyG Data object
            data = Data(x=x, 
                       edge_index=edge_index, 
                       edge_attr=edge_attr)
            
            return data

    # Initialize dataset
    dataset = ConnectomeDataset('data/connectivity', 'data/timeseries')

    # Initialize models
    in_channels = connectivity_matrices[0].shape[1]
    hidden_channels = 64
    out_channels = 2  # binary classification
    
    models = {
        'STGCN': STGCN(in_channels, hidden_channels, out_channels),
        'STChebNet': STChebNet(in_channels, hidden_channels, out_channels),
        'STGAT': STGAT(in_channels, hidden_channels, out_channels),
        'STSGConv': STSGConv(in_channels, hidden_channels, out_channels)
    }

    # Create dictionary to store results
    results = {model_name: [] for model_name in models.keys()}

    # Define early stopping parameters
    patience = 10
    min_delta = 0.001
    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_one_fold(model, train_loader, val_loader, criterion, optimizer):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            train_losses = []
            for batch_x in train_loader:
                optimizer.zero_grad()
                output = model(batch_x, None, None)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x in val_loader:
                    output = model(batch_x, None, None)
                    val_loss = criterion(output, target)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # K-fold cross validation loop
    for model_name, model in models.items():
        print(f'Training {model_name}...')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f'Fold {fold + 1}/{n_splits}')
            
            # Split data
            train_loader = torch.utils.data.DataLoader(dataset[train_idx], batch_size=1, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset[val_idx], batch_size=1, shuffle=False)
            
            # Reset model for each fold
            model = models[model_name].__class__(in_channels, hidden_channels, out_channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            train_one_fold(model, X_train, X_val, criterion, optimizer)
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val, None, None)
                val_pred_probs = F.softmax(val_pred, dim=1)
                val_pred_labels = torch.argmax(val_pred_probs, dim=1)
                
                # Calculate and store metrics
                metrics = calculate_metrics(y_val.numpy(), val_pred_labels.numpy(), val_pred_probs.numpy())
                for metric_name, value in metrics.items():
                    results[model_name][metric_name].append(value)
        
        # Save the model
        torch.save(model.state_dict(), f'./output/model/{model_name}".pth')
    
    results = {model_name: {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    } for model_name in models.keys()}

    def calculate_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob[:, 1])
        }
    
    # Print results
    for model_name, metrics in results.items():
        print(f'Model: {model_name}')
        for metric_name, values in metrics.items():
            print(f'{metric_name}: {np.mean(values)}')
        print()

    # Save results
    np.save('./output/results.npy', results)


if __name__ == '__main__':
    main()