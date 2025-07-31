
import torch.nn as nn 
import torch.nn.functional as F
import torch
from model.layers.mesh_pool2 import MyMeshPool 
from model.layers.mesh_conv import MeshConv
from model.layers.mesh import MeshLoader

class PruneNet(nn.Module):
    def __init__(self, k: list[int], in_channels=3, skips=5, pool_ratio=0.9):
           
            super().__init__()
            self.k = k
            self.blocks = len(k)
            self.mesh_loader = MeshLoader()
            for i,_ in enumerate(k):
                pool = MyMeshPool(target_edge_ratio=pool_ratio)
                setattr(self, f'pool{i + 1}', pool)
    
    
    def forward(self, batch_files, mode='train'):
        """
        Parameters:
            x: Tensor – input edge features, shape (B, C, N, 1) or (B, C, N)
            meshes: List[Mesh] – list of mesh structures per batch element

        Returns:
            final_features: Tensor – last feature tensor (B, C_out, N_out)
            final_meshes: List[Mesh] – last mesh state per batch
        """
        
        x, meshes = self.mesh_loader.forward(batch_files, mode=mode)
        x.requires_grad_(True)
        if x.dim() == 4:
            x = x.squeeze(-1)  # (B, C, N)

        for i in range(self.blocks):
        
            pool_block = getattr(self, f'pool{i + 1}')
            x, meshes = pool_block(x, meshes)  # MyMeshPool expects (B, C, N, 1)
            print(f'Pruning Returning with {list(x.shape)}')

        return x, meshes         
                
                
class MeshConvNet(nn.Module):
    def __init__(self, k: list[int], in_channels=3, skips=5, pool_ratio=0.9):
            """
            Parameters:
                k: list of int – output channels per stage (e.g. [20, 40, 80])
                in_channels: int – number of input channels (default: 3)
                skips: int – number of residual skips per block
                pool_ratio: float – edge pruning ratio in pooling
            """
            super().__init__()
            self.k = k
            self.blocks = len(k)
            self.mesh_loader = MeshLoader()
            for i, out_channels in enumerate(k):
                block = MeshResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skips=skips
                )
                pool = MyMeshPool(target_edge_ratio=pool_ratio)

                setattr(self, f'res{i + 1}', block)
                setattr(self, f'pool{i + 1}', pool)

                in_channels = out_channels  # for next block
                setattr(self, f'ff1', nn.Linear(k[-1],128))
                setattr(self, f'ff2', nn.Linear(128,64))
                setattr(self, f'ff3', nn.Linear(64,32))
    def forward(self, batch_files, mode='train'):
        """
        Parameters:
            x: Tensor – input edge features, shape (B, C, N, 1) or (B, C, N)
            meshes: List[Mesh] – list of mesh structures per batch element

        Returns:
            final_features: Tensor – last feature tensor (B, C_out, N_out)
            final_meshes: List[Mesh] – last mesh state per batch
        """
                
        x, meshes = self.mesh_loader.forward(batch_files, mode=mode)
        if x.dim() == 4:
            x = x.squeeze(-1)  # (B, C, N)

        for i in range(self.blocks):
          
            res_block = getattr(self, f'res{i + 1}')
            
            pool_block = getattr(self, f'pool{i + 1}')
            print(f'MeshResBlock Called with {list(x.shape)}')
            x, meshes = res_block(x, meshes)   
            #print(f'MeshResBlock Returning with {list(x.shape)}')
            #for i,mesh in enumerate(meshes):
                #print(i, len(mesh.edges))
            x, meshes = pool_block(x, meshes, i)  # MyMeshPool expects (B, C, N, 1)
            #print(f'Pruning Returning with {list(x.shape)}')
        x = torch.mean(x, dim=2)
        #print(f'Global Average {list(x.shape)}')
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)
        x = F.relu(x)
        x = self.ff3(x)
        print(f'Final Output {list(x.shape)}')
        return x, meshes
    
    
    
        
class MeshResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MeshResBlock, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.skips = skips 
        
        self.conv0 = MeshConv(in_channels=in_channels, out_channels=out_channels)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1),  nn.InstanceNorm2d(self.out_channels, affine=True))
            setattr(self, 'conv{}'.format(i + 1), MeshConv(in_channels=out_channels, out_channels=out_channels))
        
        
    def forward(self, x, mesh_list):
        
        x, mesh_list = self.conv0(x, mesh_list)

        x1 = x
    
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x, mesh_list = getattr(self, 'conv{}'.format(i + 1))(x, mesh_list)
           
        x += x1
        x = F.relu(x)
        updated_meshes = []
        for new_values, m in zip(x, mesh_list):
            new_values = new_values.squeeze(-1).permute(1,0)
            m.x = new_values
            updated_meshes.append(m)
        return x, updated_meshes 
    
    def __call__(self, x, mesh):
        return self.forward(x,mesh)
        


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z):
        z = F.normalize(z, dim=1)
        
        sim = torch.matmul(z, z.T) / self.temperature
        n = z.size(0)
        
        mask = torch.eye(n, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -float("inf"))
        pos_idx = torch.arange(n, device=z.device)
        targets = pos_idx ^ 1  # fast way to flip LSB
        loss = F.cross_entropy(sim, targets)
        return loss
    
    
        