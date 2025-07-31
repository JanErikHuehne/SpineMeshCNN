import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.mesh import *

class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, in_channels, out_channels, k=5, bias=True):
        super(MeshConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), bias=bias)
        self.k = k

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        # apparently this tensor concatentates all the edge features from all meshes in the batch
        # x is (batch_size, 5, n_edges), representing a 5-dimensional-feature vector per edge
        
        x = x.squeeze(-1) # (batch_size, 5, n_edges)
        # Building the neighboorhood index tensor G, which encodes the 4-ring neighboorhood for every edge in the batch
        # pad_gemm returns (1, xsz, 5) as tensor shape
        # meshes seems to be a list of meshes so get a total of (m, xsz, 5) as the shape of G 
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        x = self.conv(G)
        upd_mesh =self.update_meshes(x, mesh)
        return x, upd_mesh
    def update_meshes(self, x, meshes):
        updated_meshes = []
        for new_values, m in zip(x, meshes):
            new_values = new_values.squeeze(-1).permute(1,0)
            m.x = new_values
            updated_meshes.append(m)
            
        return updated_meshes
        
    def flatten_gemm_inds(self, Gi):
        # G has the shape of (m, xsz, 5)
        # b = m 
        # ne = xsz
        # nn = 5
        (b, ne, nn) = Gi.shape
        # ne = xsz + 1 (to account for padded dim?)
        ne += 1
        # Is this constructing global offset of each sample in the batch?
        #   xx = (torch.arange(b * ne, device=Gi.device).float() this gives a tensor of [0, 1 , ... , (b * ne) -1]
        #   floor(xx / ne)  yield [0,0,0 ..., 1,1,1, ..., b-1, b-1, ...] where each entry is repeated ne times 
        #   this means that we get an array where for each of the entries of G (m, xsz) we have the representing batch 
        #   number of the sample
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        # the offset if batch number * samples in batch, which is computed here
        add_fac = batch_n * ne
        # the next two lines are just there to extend this to the third dimension (every index in the dataset)
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        # Here the offset is added to Gi, note that we only use add_fac[:, 1:, :] since 
        # ne has been incremented by 1 to account for the adding of the zero padding, but Gi of course 
        # does not have this in it 
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        # this is of shape (m, xsz, 5), m is the batch size, xsz is the padded number of edges, 5 are the neighbors including itself 
        # X shape is (B,5,E_padded)
        Gishape = Gi.shape # (B, E_padded, 5)
        # pad the first row of  every sample in batch with zeros
        # this generates a tensor of zeros with shape (n_edges, 5,1)
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2) # → (B, 5, E_padded + 1)
        # We shift the indices of every Gi by 1 to account for the added 0 padding in the values 
        # I assume that every invalid padding argument should point to -1 to point to the zero vector?
        Gi = Gi + 1 #shift

        # first flatten indices
        # So basically this gives Gi the indices of each neighboors and itself given the total list (flattend) of edges of all meshes, 
        # under the assumption that each mesh is padded by a zero representation
       
        Gi_flat = self.flatten_gemm_inds(Gi) #  → (B, E, 5)
        Gi_flat = Gi_flat.view(-1).long()  #  → (B*E*5,)
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous() # (B, E_padded+1, 5)
        # This will put x into a shape of (mesh*edges, features)
        # This matches how we flattened the Gi indices 
        x = x.view(odim[0] * odim[2], odim[1]) # → (B * E_padded+1, 5)
        # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor

        f = torch.index_select(x, dim=0, index=Gi_flat)
        
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1) # → (B,E_padded, neighboors, features)
        f = f.permute(0, 3, 1, 2)  # → (B,features, edges, neighboors)
        # now f has a format like (batch, channels_in, width, kernel)
        # B: batch 
        # F: number of input features per edge (channels_in)
        # E: number of edges (width)
        # Neighboors: Height of imaginable image 

        # apply the symmetric functions for an equivariant conv
        # 4 featues to make ambigious of order of the edge order 
        
        x_1 = f[:, :, :, 1] + f[:, :, :, 3] # Starting at 1 because 0 is the edge itself a + c
        x_2 = f[:, :, :, 2] + f[:, :, :, 4] #  b + d
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3]) # |a-c|
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4]) # |b-d|
        # we get the same tensor back again 
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        # Returns (B,E_padded, neighboors, features)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        
        # xsz desired number of rows to pad to (typically total number of edges in the largest mesh in the batch)
        # m: a single mesh object
        # m.gemm_edges is (n_edges, 4) the 4 one-right neighbors for each edge 
        # this just converts this to a torch tensor
        padded_gemm = m.gemm_edges.to(device).float()
        padded_gemm = padded_gemm.requires_grad_()
        # torch.arange(m.edge_count) : (n_edges,) [0,1,..., n_edges-1]
        # unsqueeze results in (n_edges,1)
        # torch cat of (n_edges,1) and (n_edges,4) into (n_edges,5)
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        # pads along dimension 0 (rows)
        # after padding (xsz, 5)
        
        """
        Should we not padd with -1 values to point towards the zero vector in the end?
        """
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant",-1)
        # We unsqueeze into (1,xsz,5)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm



"""
class DummyMesh:
    def __init__(self, gemm_edges):
        self.gemm_edges = gemm_edges
        self.edges_count = len(gemm_edges)
        

conv_layer = MeshConv(in_channels=5, out_channels=8, k=5)


# Create dummy batch of 2 meshes, each with 3 edges and 4 neighbors per edge

f = ['/Users/jan/Documents/Microns/CLEAN CODE/spines/864691136106958809_2.stl', '/Users/jan/Documents/Microns/CLEAN CODE/spines/864691136195891788_0.stl']    
mesh_batch = feature_extractor(files=f)
features = []
max_edges = 0

# First pass: find the max number of nodes (edges)
for mesh in mesh_batch:
    if mesh.x.shape[0] > max_edges:
        max_edges = mesh.x.shape[0]
    features.append(mesh.x)

# Second pass: pad each tensor to match max_edges
padded_features = []
for feat in features:
    pad_size = max_edges - feat.shape[0]
    if pad_size > 0:
        # Pad rows with zeros (at the bottom)
        padding = torch.zeros(pad_size, feat.shape[1], dtype=feat.dtype, device=feat.device)
        feat_padded = torch.cat([feat, padding], dim=0)
    else:
        feat_padded = feat
    padded_features.append(feat_padded)

# Optionally stack into a single tensor if dimensions now match
edge_features = torch.stack(padded_features).permute((0,2,1))  # Shape: [batch_size, max_edges, feature_dim]
# Run forward pass
out, batch_meshes = conv_layer(edge_features, mesh_batch)

# Output shape should be: (6, 8, 1, 1)
print("Output shape:", out.shape)
"""