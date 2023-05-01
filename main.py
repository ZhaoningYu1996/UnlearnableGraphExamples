import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_max

# 1. Install PyTorch Geometric
# pip install torch-geometric

# 2. Import required packages
# Already imported above

# 3. Load the MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='AIDS')
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# 4. Define a graph neural network model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, dataset.num_classes)
        self.linear2 = torch.nn.Linear(dataset.num_features, dataset.num_features)
        self.linear3 = torch.nn.Linear(2*dataset.num_features, 1)

    def forward(self, data, train=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output_noise = 1000
        
        
        # # Add node feature noise
        # if train:
        #     inter_noise = self.linear2(x)
        #     noise = (inter_noise - inter_noise.mean(dim=0))/inter_noise.std(dim=0)
        #     x = x + noise
        #     output_noise = torch.sum(noise, dim=1)
        
        # make edge pertubation
        if train:
            # print(f"number of edges: {edge_index.size(1)}")
            edge_batch = []
            for i in range(edge_index.size(1)):
                edge_batch.append(batch[edge_index[0,i]])
            
            edge_batch = torch.tensor(edge_batch).to(device)
            # print(f"number of batch: {edge_batch.size(0)}")
            edge_rep = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
            # print(f"check: {edge_rep.size()}")
            edge_predict = self.linear3(edge_rep)
            edge_softmax = softmax(edge_predict, edge_batch)
            # print(f"-------------> {edge_softmax.size()}")
            
            # print(edge_softmax)
            max_values, max_indices = scatter_max(edge_softmax, edge_batch, dim=0)
            global_max_indices = max_indices.squeeze()
            selected_indices = global_max_indices
            
            # Create a mask for the complement of global_max_indices
            total_nodes = edge_index.size(1)
            # print(f"totol nodes: {total_nodes}")
            mask = torch.ones(total_nodes, dtype=torch.bool)
            mask[selected_indices] = False
            edge_index = edge_index[:, mask]
            
            
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.linear(x), dim=1), output_noise

# 5. Split the dataset into train and test sets
dataset = dataset.shuffle()
train_dataset = dataset[:1700]
test_dataset = dataset[1700:]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 6. Train the model
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = GCN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()

def train():
    model.train()
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, noise = model(data, train=False)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        
        loss = criterion(out, data.y)
        target = torch.zeros_like(data.x[:,0])

        # loss += 0.5*mse_loss(noise, target)
        loss.backward()
        optimizer.step()
    return correct / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, noise = model(data, train=False)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

for epoch in range(1, 201):
    train_acc = train()
    # train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

# 7. Evaluate the model
# Training and testing accuracies are printed during training
