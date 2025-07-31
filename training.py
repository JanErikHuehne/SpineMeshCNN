import torch 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torch.utils.data import Dataset
import os 

from model.mesh_net import MeshConvNet, NTXentLoss
class MeshPairsDataset(Dataset):
    def __init__(self, mesh_dir):
        self.mesh_paths = sorted([os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir)])
        remove_elements = []
        for i in range(len(self.mesh_paths)):
            if not 'stl' in self.mesh_paths[i]:
               remove_elements.append(self.mesh_paths[i])
        for elem in remove_elements:
            self.mesh_paths.remove(elem)

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        return self.mesh_paths[idx]  # The model loads and processes meshes internally


def train_meshconvnet(mesh_dir, epochs=100, batch_size=2, lr=1e-3, device='cpu'):
    
    model = MeshConvNet(k=[32,64,64,128,256], in_channels=5, pool_ratio=0.7, skips=5)
    dataset = MeshPairsDataset(mesh_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = model.to(device)
    loss_fn = NTXentLoss()
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0 
        for batch_paths in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch_files = list(batch_paths)
            x, _ = model(batch_files, mode='train')
            loss = loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
        
        

train_meshconvnet(mesh_dir="/Users/jan/Documents/Microns/CLEAN CODE/clean_spines")
