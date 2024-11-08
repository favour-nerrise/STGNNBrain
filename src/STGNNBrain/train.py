import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import STGCN, STChebNet, STGAT, STSGConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch_geometric
from torch_geometric.data import Data, Dataset


seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def load_connectivity_data(path):
    conn_matrices = {}

    for file in os.listdir(path):
        if file.endswith('.txt'):
            subject_id = file.split('sub')[1].split('.')[0]
            matrix = np.loadtxt(os.path.join(path, file), dtype='str', delimiter=',')[1:, 1:]
            matrix = matrix.astype(np.float32)
            conn_matrices[subject_id] = torch.FloatTensor(matrix)

    return conn_matrices

# Load corresponding timeseries (node features)
def load_timeseries_data(subjects, path):
    timeseries_matrices = {}
    
    for subject_id in subjects:
        ts_file = f'sub{subject_id}.txt'
        print(path + ts_file)
        if os.path.exists(os.path.join(path, ts_file)):
            matrix = np.loadtxt(os.path.join(path, ts_file), dtype='str', delimiter=',')[1:, 1:]
            matrix = matrix.astype(np.float32)
            timeseries_matrices[subject_id] = torch.FloatTensor(matrix)
        else:
            raise ValueError(f"Missing timeseries data for subject {subject_id}")

def load_labels(path):
    labels = {}
   
    labels_df = pd.read_csv(path)
    for i, row in labels_df.iterrows():
        subject_id = row['subcode'].split('sub')[1]
        label = row['caffeinated']
        labels[subject_id] = label
    
    return labels

def main():
    # Load connectivity matrices and subject IDs
    connectivity_matrices = load_connectivity_data('../../data/connectivity_aal116/')
    print(f'Number of samples: {len(connectivity_matrices)}')
    print(f'Number of ROIs: {connectivity_matrices[list(connectivity_matrices)[0]].shape[0]}')
    print(connectivity_matrices.keys())

    # Load timeseries matrices
    timeseries_matrices = load_timeseries_data(connectivity_matrices.keys(), '../../data/timeseries_aal116')
    print(f'Number of timeseries matrices: {len(timeseries_matrices)}')

    # Load labels
    labels = load_labels('../../data/labels.csv')
    print(f'Class distribution: {np.unique(labels, return_counts=True)}')
    print('-' * 50)
    
    # Prepare data for k-fold cross validation
    # X = torch.stack(connectivity_matrices)
    # y = torch.tensor(labels)   

    # Define data loader as ConnectomeDataset
    class ConnectomeDataset(Dataset):
        def __init__(self, connectivity_matrices, timeseries_matrices, labels):
            super().__init__()
            self.connectivity_matrices = connectivity_matrices
            self.timeseries_matrices = timeseries_matrices
            self.subjects = connectivity_matrices.keys()
            self.labels = labels
            
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
            y = self.labels[subject_id]
            
            # Create PyG Data object
            data = Data(x=x, 
                       edge_index=edge_index, 
                       edge_attr=edge_attr,
                       y=y[idx],
                       s_id=torch.LongTensor([subject_id])
                       )
            
            return data

    # Initialize dataset
    dataset = ConnectomeDataset(connectivity_matrices, timeseries_matrices, labels)

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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def train(model, train_loader, criterion, optimizer, num_epochs=100):

        avg_train_loss = 0
 
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
        
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {round(np.mean(train_losses), 3)}')
    
    def evaluate(model, val_loader, criterion):
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Validation phase
        model.eval()
        val_losses = []

        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                output = model(batch)
                val_loss = criterion(output, batch.y)
                val_pred_probs = F.softmax(output, dim=1)
                val_pred_labels = torch.argmax(val_pred_probs, dim=1)
                
                # Calculate and store metrics
                metrics = calculate_metrics(batch.y.cpu().numpy(), val_pred_labels.numpy(), val_pred_probs.numpy())

                val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
        
                # Early stopping check
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping...')
                        break
        
        return avg_val_loss, metrics
    
    def calculate_metrics(y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_prob[:, 1])
        }

    # K-fold cross validation loop
    for model_name, model in models.items():
        print(f'Training {model_name}...')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, [data.y for data in dataset])):
            print(f'Fold {fold + 1}/{n_splits}')
            
            # Split data
            train_loader = torch.utils.data.DataLoader(dataset[train_idx], batch_size=1, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset[val_idx], batch_size=1, shuffle=False)
            
            # Reset model for each fold
            model = models[model_name].__class__(in_channels, hidden_channels, out_channels)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Train model
            train_loss = train(model, train_loader, val_loader, criterion, optimizer)
           
            # Evaluate model
            val_loss, metrics = evaluate(model, val_loader, criterion)
            for metric_name, value in metrics.items():
                results[model_name][metric_name].append(value)

            print(f'Training Loss: {round(train_loss, 3)} | Validation Loss: {round(val_loss, 3)}')
            for metric_name, values in metrics.items():
                print(f'{metric_name}: {np.mean(values)}')
        
        # Save the model
        torch.save(model.state_dict(), f'./output/model/{model_name}".pth')
    
    results = {model_name: {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    } for model_name in models.keys()}
    
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