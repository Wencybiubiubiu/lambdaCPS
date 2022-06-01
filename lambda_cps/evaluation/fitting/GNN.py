from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dgl.nn import GraphConv


embedding_size = 128

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.mean(x,dim=0)
        #x = F.linear(x, weight=weights)
        x = F.log_softmax(x, dim=0).unsqueeze(0)
        return x


dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

model.train()

#weights = torch.randn(dataset.num_classes, embedding_size)
for epoch in range(1000):

    for i in range(len(dataset)):

        print(epoch,i)
        data = dataset[i].to(device)
        
        optimizer.zero_grad()
        out = model(data)
        #loss = F.cross_entropy(out.float(), label_one_hot[data.y.item()].float())
        #print(out, data.y)
        loss = F.nll_loss(out,data.y)
        loss.backward()
        optimizer.step()


model.eval()
correct = 0

for i in range(len(dataset)):
        
    data = dataset[i]
        
    pred = model(data).argmax(dim=1)
    print(pred,data.y)
    if(pred == data.y):
        correct += 1
    
acc = int(correct) / int(len(dataset))
print(f'Accuracy: {acc:.4f}')