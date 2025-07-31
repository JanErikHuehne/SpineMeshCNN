import torch
import torch.nn as nn
from threading import Thread
#from model.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
from model.layers.mesh import *
import logging
class MyMeshPool(nn.Module):
    def __init__(self, target_edge_ratio=0.05):
        super().__init__()
        self.target_edge_ratio = target_edge_ratio
        self.logger = logging.getLogger('PruneLogger')
        handler = logging.FileHandler('prune_debug.log', mode='w')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
    def forward(self, edge_features, meshes, i):
        edge_features = edge_features.squeeze(-1).clone()
        B, F, N = edge_features.shape
        updated_meshes = []
        updated_features = []
        for mesh in meshes:
            print(len(mesh.edges))
        for b in range(B):
         
            features = edge_features[b, :, :]  # (F, N)
            print("Features shape ", features.shape)
            mesh = meshes[b]
            mesh.x = features.T.contiguous() 
            heap = self.build_heap(features, mesh.edges_count)
            valid_edge_mask = torch.any(mesh.x != 0, dim=1)  # shape: (n_edges,)
            num_valid_edges = valid_edge_mask.sum().item()
            mesh.edge_count = num_valid_edges
            self.pruned_edges = torch.ones(len(features), dtype=torch.bool)

            while mesh.edges_count > (num_valid_edges * self.target_edge_ratio):
                try:
                    _, edge_idx = heappop(heap)
                    mesh = self.prune_edge(edge_idx, mesh)
                    self.check_consistency(mesh)
                except IndexError:
                    break
                    
              
           
            updated_meshes.append(self.clean_mesh(mesh))
            try:
               
                updated_features.append(features[:, self.pruned_edges])
            except Exception:

                print(len(mesh.edges),features.shape, self.pruned_edges.shape)
                raise Exception()
        self.export_mesh_to_stl(*self.reconstruct_faces(mesh), f'{i}.stl')
        return self.pad_features_to_max_edges(updated_features), updated_meshes
    
    def pad_features_to_max_edges(self, feature_list, pad_value=0.0):
        """
        Pads each (F, E_i) feature tensor in feature_list to (F, E_max),
        where E_max is the max number of edges across all batch elements.

        Parameters:
            feature_list: List[Tensor] – each of shape (F, E_i)
            pad_value: float – value to pad with

        Returns:
            padded_features: Tensor of shape (B, F, E_max)
        """
        F = feature_list[0].shape[0]
        max_edges = max(feat.shape[1] for feat in feature_list)
        padded = []

        for feat in feature_list:
            E = feat.shape[1]
            if E < max_edges:
                pad_size = max_edges - E
                padding = feat.new_full((F, pad_size), pad_value)
                feat = torch.cat([feat, padding], dim=1)
            padded.append(feat)

        return torch.stack(padded, dim=0)  # shape: (B, F, E_max)
    
    def get_remaining_neighbors(self,gemm_row, exclude_set):
        """
        Returns the two values in gemm_row that are not in exclude_set
        """
        return [int(e.item()) for e in gemm_row if int(e.item()) not in exclude_set]
    
    def boundary_check(self, e0, mesh):
        for edge in mesh.gemm_edges[e0]:
            if edge == -1 or edge == -1.0 or -1 in mesh.gemm_edges[int(edge)] or -1.0 in mesh.gemm_edges[int(edge)]: 
                return False 
        return True
    
    
    def same_side_check(self, e0, mesh):
        changed = False
        neighbors = mesh.gemm_edges[e0].clone()
        n1, n2, n3, n4 = [int(n.item()) for n in neighbors]
        for (e1, e2, e3, e4) in [[n1,n2, n3, n4], [n3,n4, n1, n2]]:
            
            e1_neigh = mesh.gemm_edges[e1]
            e1_neigh =self.get_remaining_neighbors(e1_neigh, [e2, e0])
            e2_neigh = mesh.gemm_edges[e2]
            e2_neigh =self.get_remaining_neighbors(e2_neigh, [e1, e0])
            shared_items = self.__get_shared_items(e1_neigh, e2_neigh)
            self.logger.debug(((e1, e2, e3, e4), e1_neigh, e2_neigh))
            if len(shared_items) != 0:
                    # This means that we have detected an anotheredge that is shared by n0 and n1 
                    assert len(shared_items) == 2
                    # this is the shared edge
                    e7 = e1_neigh[shared_items[0]] #e7
                    # These are the neighbour  edges of n0 and n1 thare are not shared
                    # not that shared_items contains either 0 or 1 as entires, so we simply 
                    # select the other entry here 
                    try:
                        e5 =  e1_neigh[1 - shared_items[0]] #e5
                        e6 = e2_neigh[1 - shared_items[1]] # e6
                    except IndexError:
                        raise ValueError(f"{e0} {e1} {e2} {mesh.gemm_edges[e1]} {mesh.gemm_edges[e2]}")
                    
                    self.logger.debug(f"Same-Side Pruning Routine")
                    up_e0 = mesh.gemm_edges[e0].clone()
                    up_e1 = mesh.gemm_edges[e1].clone()
                    up_e2 = mesh.gemm_edges[e2].clone()
                    up_e3 = mesh.gemm_edges[e3].clone()
                    up_e4 = mesh.gemm_edges[e4].clone()
                    up_e5 = mesh.gemm_edges[e5].clone()
                    up_e6 = mesh.gemm_edges[e6].clone()
                    up_e7 = mesh.gemm_edges[e7].clone()
                    self.logger.debug(f"Called E0({e0}) {up_e0}")
                    self.logger.debug(f"Called E1({e1}) {up_e1}")
                    self.logger.debug(f"Called E2({e2}) {up_e2}")
                    self.logger.debug(f"Called E3({e3}) {up_e3}")
                    self.logger.debug(f"Called E4({e4}) {up_e4}")
                    self.logger.debug(f"Called E5({e5}) {up_e5}")
                    self.logger.debug(f"Called E6({e6}) {up_e6}")
                    self.logger.debug(f"Called E7({e7}) {up_e7} \n")
                    # we are subst e1 with e5
                    mesh.gemm_edges[int(e0)][mesh.gemm_edges[int(e0)] == float(e1)] = float(e5)
                    # we are subst e1 with e0
                    mesh.gemm_edges[int(e5)][mesh.gemm_edges[int(e5)] == float(e1)] = float(e0)
                    # we are subst e2 with e6
                    mesh.gemm_edges[int(e0)][mesh.gemm_edges[int(e0)] == float(e2)] = float(e6)
                    # we are subst e2 with e0
                    mesh.gemm_edges[int(e6)][mesh.gemm_edges[int(e6)] == float(e2)] = float(e0)
                
                    # we are subst e7 with e5
                    mesh.gemm_edges[int(e6)][mesh.gemm_edges[int(e6)] == float(e7)] = float(e5)
                    # we are subst e7 with e6
                    mesh.gemm_edges[int(e5)][mesh.gemm_edges[int(e5)] == float(e7)] = float(e6)
                    
                    mesh.gemm_edges[int(e7)] = -1.0
                    mesh.gemm_edges[int(e1)] =  -1.0
                    mesh.gemm_edges[int(e2)] =  -1.0
                    
                    self.pruned_edges[e7] = False
                    self.pruned_edges[e1] = False
                    self.pruned_edges[e2] = False
                    # Removing the edges
                    mesh.edges[e7] = -1.0 
                    mesh.edges[e1] = -1.0 
                    mesh.edges[e2] = -1.0 
                    # Finally creating the new edge features     
                    mesh.x[e0] = (mesh.x[e0] + mesh.x[e1] + mesh.x[e2]) / 3.0
                    mesh.x[e5] = (mesh.x[e5] + mesh.x[e7] + mesh.x[e1]) / 3.0
                    mesh.x[e6] = (mesh.x[e6] + mesh.x[e7] + mesh.x[e2]) / 3.0
                    # We the rest of the features to 0, this will not be used 
                    mesh.edges_count -= 3
                    mesh.x[e1] = 0
                    mesh.x[e2] = 0
                    mesh.x[e7] = 0
                    
                    
                    up_e0 = mesh.gemm_edges[e0].clone()
                    up_e1 = mesh.gemm_edges[e1].clone()
                    up_e2 = mesh.gemm_edges[e2].clone()
                    up_e3 = mesh.gemm_edges[e3].clone()
                    up_e4 = mesh.gemm_edges[e4].clone()
                    up_e5 = mesh.gemm_edges[e5].clone()
                    up_e6 = mesh.gemm_edges[e6].clone()
                    up_e7 = mesh.gemm_edges[e7].clone()
                    self.logger.debug(f"Updated E0({e0}) {up_e0}")
                    self.logger.debug(f"Updated E1({e1}) {up_e1}")
                    self.logger.debug(f"Updated E2({e2}) {up_e2}")
                    self.logger.debug(f"Updated E3({e3}) {up_e3}")
                    self.logger.debug(f"Updated E4({e4}) {up_e4}")
                    self.logger.debug(f"Updated E5({e5}) {up_e5}")
                    self.logger.debug(f"Updated E6({e6}) {up_e6}")
                    self.logger.debug(f"Updated E7({e7}) {up_e7} ")
                    changed = True
                    self.logger.debug(f'Returning {changed} \n')
                    return mesh, changed
        return mesh, changed
        
    def cross_side_check(self, e0, mesh):
        neighbors = mesh.gemm_edges[e0].clone()
        n1, n2, n3, n4 = [int(n.item()) for n in neighbors]
        # Now we are checking for cross triangular cases
        for (e1, e2,e3,e4) in[[n1,n3,n2,n4], [n2,n4,n1,n3],  [n1,n4,n2,n3], [n2,n3,n1,n4]]:
            e1_neigh = mesh.gemm_edges[e1]
            e1_neigh =self.get_remaining_neighbors(e1_neigh, [e0,e2,e3,e4])
            e2_neigh = mesh.gemm_edges[e2]
            e2_neigh =self.get_remaining_neighbors(e2_neigh,  [e0,e1,e3,e4])
            shared_items = self.__get_shared_items(e1_neigh, e2_neigh)
            if len(shared_items) != 0 and e1 not in mesh.gemm_edges[e2]:
                    return mesh, True
                    '''
                    # Make sure we are labeled the vertices in the correct order
                    if not e1 in mesh.gemm_edges[e3]:
                        e3, e4 = e4, e3 
                    #print(f'Cross Cleaning side {e0} {i}')
                    assert len(shared_items) == 2
                    # this is the shared edge
                    e7 = e1_neigh[shared_items[0]] #e7
                    
                    # e0 and e7 will be fused so we need to update the vertex of e0 
                    (v1_e0, v2_e0) = mesh.edges[e0].tolist()
                    (v1_e7, v2_e7)  = mesh.edges[e7].tolist()
                    
                    if v1_e0 == v1_e7:
                        mesh.edges[e0, 0] = v2_e0
                        mesh.edges[e0, 1] = v2_e7
                    elif v2_e0 == v2_e7:
                    
                        mesh.edges[e0, 0] = v1_e0
                        mesh.edges[e0, 1] = v1_e7
                    elif v1_e0 == v2_e7:
                    
                        mesh.edges[e0, 0] = v2_e0
                        mesh.edges[e0, 1] = v1_e7
                    elif v2_e0 == v1_e7:
                        mesh.edges[e0, 0] = v1_e0
                        mesh.edges[e0, 1] = v2_e7
                    else:
                        raise ValueError(f'Cross-trinagular cleaning run into an issue {(v1_e0, v2_e0)} ,  {(v1_e7, v2_e7)}')
                    
                    
                    e5, e6 = set(mesh.gemm_edges[e7].tolist()) - set([e1]) - set([e2])
                    up_e0 = mesh.gemm_edges[e0].clone()
                    up_e1 = mesh.gemm_edges[e1].clone()
                    up_e2 = mesh.gemm_edges[e2].clone()
                    up_e3 = mesh.gemm_edges[e3].clone()
                    up_e4 = mesh.gemm_edges[e4].clone()
                    up_e5 = mesh.gemm_edges[e5].clone()
                    up_e6 = mesh.gemm_edges[e6].clone()
                    up_e7 = mesh.gemm_edges[e7].clone()
                    self.logger.debug(f"Cross-Checked Pruning Routine")
                    self.logger.debug(f"Called E0({e0}) {up_e0}")
                    self.logger.debug(f"Called E1({e1}) {up_e1}")
                    self.logger.debug(f"Called E2({e2}) {up_e2}")
                    self.logger.debug(f"Called E3({e3}) {up_e3}")
                    self.logger.debug(f"Called E4({e4}) {up_e4}")
                    self.logger.debug(f"Called E5({e5}) {up_e5}")
                    self.logger.debug(f"Called E6({e6}) {up_e6}")
                    self.logger.debug(f"Called E7({e7}) {up_e7}\n")
                    if not e1 in mesh.gemm_edges[int(e5)]:
                        e5, e6 = e6,e5
                    # we are subst e2 with e6
                    mesh.gemm_edges[e4][mesh.gemm_edges[e4] == e2] = e6
                    mesh.gemm_edges[e0][mesh.gemm_edges[e0] == e2] = e6
                    # we are subst e2 with e4
                    mesh.gemm_edges[int(e6)][mesh.gemm_edges[int(e6)] == e2] = e4
                    # we are subst e1 with e5
                    mesh.gemm_edges[e3][mesh.gemm_edges[e3] == e1] = e5
                    mesh.gemm_edges[e0][mesh.gemm_edges[e0] == e1] = e5
                    # we are subst e1 with e3
                    mesh.gemm_edges[int(e5)][mesh.gemm_edges[int(e5)] == e1] = e3
                    # we are subst e7 with e0 
                    mesh.gemm_edges[int(e6)][mesh.gemm_edges[int(e6)] == e7] = e0
                    mesh.gemm_edges[int(e5)][mesh.gemm_edges[int(e5)] == e7] = e0
                    mesh.edges[e7] = -1.0
                    mesh.edges[e1] = -1.0
                    mesh.edges[e2] = -1.0
                    mesh.gemm_edges[e7] = -1.0
                    mesh.gemm_edges[e2] = -1.0
                    mesh.gemm_edges[e1] = -1.0
                    # Finally creating the new edge features     
                    mesh.x[e0] = (mesh.x[e0] + mesh.x[e7]) / 2.0
                    mesh.x[e4] = (mesh.x[e4] + mesh.x[e2]) / 2.0
                    mesh.x[int(e6)] = (mesh.x[int(e6)] + mesh.x[e2]) / 2.0
                    mesh.x[int(e5)] = (mesh.x[int(e5)] + mesh.x[e1]) / 2.0
                    mesh.x[e3] = (mesh.x[e3] + mesh.x[e1]) / 2.0
                    # We the rest of the features to 0, this will not be used 
                    mesh.edges_count -= 3
                    mesh.x[e1] = 0
                    mesh.x[e2] = 0
                    mesh.x[e7] = 0
                    
                    self.pruned_edges[e7] = False
                    self.pruned_edges[e1] = False
                    self.pruned_edges[e2] = False
                    
                    up_e0 = mesh.gemm_edges[e0].clone()
                    up_e1 = mesh.gemm_edges[e1].clone()
                    up_e2 = mesh.gemm_edges[e2].clone()
                    up_e3 = mesh.gemm_edges[e3].clone()
                    up_e4 = mesh.gemm_edges[e4].clone()
                    up_e5 = mesh.gemm_edges[e5].clone()
                    up_e6 = mesh.gemm_edges[e6].clone()
                    up_e7 = mesh.gemm_edges[e7].clone()
                
                    self.logger.debug(f"Updated E0({e0}) {up_e0}")
                    self.logger.debug(f"Updated E1({e1}) {up_e1}")
                    self.logger.debug(f"Updated E2({e2}) {up_e2}")
                    self.logger.debug(f"Updated E3({e3}) {up_e3}")
                    self.logger.debug(f"Updated E4({e4}) {up_e4}")
                    self.logger.debug(f"Updated E5({e5}) {up_e5}")
                    self.logger.debug(f"Updated E6({e6}) {up_e6}")
                    self.logger.debug(f"Updated E7({e7}) {up_e7}\n")
                    '''
                    changed = True
                    return mesh, changed
        return mesh, False
    
    def __get_shared_items(self, list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items
    
    def export_mesh_to_stl(self, vertices, faces, path):
        import trimesh
        m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.repair.fix_winding(m)
        m.export(path)
        
    def build_heap(self,features, edge_count):
        """
        features: (F, N)
        Returns: list of (priority, edge_index)
        """
        priorities = torch.sum(features * features, dim=0)  # (N,)
        entries = [(priorities[i].item(), i) for i in range(edge_count)]
        heapify(entries)
        return entries
    
    def clean_mesh(self, mesh):
        """
        Cleans the mesh by removing unused edges and vertices,
        and relabels the remaining ones to ensure continuous indexing.
        """

        # STEP 1: Identify used edges from gemm_edges
        valid_edges = []
        for edge_idx,edge in enumerate(mesh.edges):
            if -1.0 not in edge.tolist():
                valid_edges.append(edge_idx)
    
        
        
        # Mapping: old edge idx -> new edge idx
        edge_mapping = {old: new for new, old in enumerate(valid_edges)}
        
        # STEP 2: Keep only used edges
        new_edges = mesh.edges[valid_edges]
       
        # STEP 3: Identify used vertices from remaining edges
        used_vertex_indices = torch.unique(new_edges)
        used_vertex_indices = used_vertex_indices[used_vertex_indices != -1] 
        used_vertex_indices = used_vertex_indices.to(dtype=torch.long)

        # Mapping: old vertex idx -> new vertex idx
        vertex_mapping = {old.item(): new for new, old in enumerate(used_vertex_indices)}
      
        # STEP 4: Remap edge vertex indices
        valid_gemm_edges =  mesh.gemm_edges[valid_edges]
        remapped_edges = torch.stack([
            torch.tensor([vertex_mapping[int(v)] for v in edge], dtype=torch.long)
            for edge in new_edges
        ])

        # STEP 5: Remap gemm_edges
        remapped_gemm_edges = []
       

        for v_edge, row in zip(valid_edges, valid_gemm_edges):
            new_row = []
            for e in row:
                e = e.item()
                if e not in edge_mapping:
                    if e != -1.0:
                        print('---- edge', v_edge, ' invalid ', e, ' row ',row)
                new_row.append(edge_mapping[e] if e in edge_mapping else -1)
            remapped_gemm_edges.append(new_row)
        remapped_gemm_edges = torch.tensor(remapped_gemm_edges, dtype=torch.long)

        # STEP 6: Remap vertices
     
        new_features = mesh.x[valid_edges]

        # Update mesh
        mesh.edges = remapped_edges
        mesh.vertices = mesh.vertices[used_vertex_indices]
        mesh.gemm_edges = remapped_gemm_edges
        self.logger.debug(f'Cleaned mesh maximum gemm_edges { mesh.gemm_edges.max()}')
        mesh.x = new_features
        mesh.edges_count = len(mesh.gemm_edges)
        return mesh
    
   
    def __is_one_ring_valid(self, mesh, edge_id):
        v0, v1 = mesh.edges[edge_id]
        # Find edges incident to v0 and v1
        incident_to_v0 = mesh.edges[(mesh.edges == v0).any(dim=1)]
        incident_to_v1 = mesh.edges[(mesh.edges == v1).any(dim=1)]

        # Flatten and make sets of all vertices in these incident edges
        v_a = set(incident_to_v0.reshape(-1).tolist())
        v_b = set(incident_to_v1.reshape(-1).tolist())

        # Remove the vertices of the edge itself
        shared = (v_a & v_b) - {v0.item(), v1.item()}

        return len(shared) == 2

    def prune_edge(self, e0, mesh):
        """
        We prune e0
        """

        self.logger.debug(f"\n=== PRUNING EDGE {e0} ===")
       
        self.logger.debug(f"Initial gemm_edges: {mesh.gemm_edges[e0]}")


        if self.boundary_check(e0, mesh) and self.__is_one_ring_valid(mesh, e0):
            
            self.logger.debug("Before pruning check routine")
            e1,e2,e3,e4 = [int(i) for i in mesh.gemm_edges[e0].clone()]
            inital_e0 = mesh.gemm_edges[e0].clone()
            inital_e1 = mesh.gemm_edges[e1].clone()
            inital_e2 = mesh.gemm_edges[e2].clone()
            inital_e3 = mesh.gemm_edges[e3].clone()
            inital_e4 = mesh.gemm_edges[e4].clone()
            self.logger.debug(f"Inital E0({e0}) {inital_e0}")
            self.logger.debug(f"Inital E1({e1}) {inital_e1}")
            self.logger.debug(f"Inital E2({e2}) {inital_e2}")
            self.logger.debug(f"Inital E3({e3}) {inital_e3}")
            self.logger.debug(f"Inital E4({e4}) {inital_e4}")
            

            changed = True
            iteration = 1
            while changed:     
                self.logger.debug(f'Checking viability {iteration}')           
                mesh, changed = self.same_side_check(e0, mesh)
                mesh, invalid = self.cross_side_check(e0, mesh)
                if invalid:
                    self.logger.debug('Skipping invalid')
                    return mesh 
                iteration += 1
             
                
            e1,e2,e3,e4 = [int(i) for i in mesh.gemm_edges[e0].clone()]
            self.logger.debug('Pruning check completed')
            if not self.boundary_check(e0, mesh):
                return mesh
            pruning_check_e0 = mesh.gemm_edges[e0].clone()
            pruning_check_e1 = mesh.gemm_edges[e1].clone()
            pruning_check_e2 = mesh.gemm_edges[e2].clone()
            pruning_check_e3 = mesh.gemm_edges[e3].clone()
            pruning_check_e4 = mesh.gemm_edges[e4].clone()

            ######
            # Vertex handling
            ######
            v0, v1 = mesh.edges[e0].tolist()

            new_v = (mesh.vertices[int(v0)] + mesh.vertices[int(v1)]) / 2
            mesh.vertices[int(v0)] = new_v
            mesh.vertices[int(v1)] = -1.0

    

            mesh.edges[mesh.edges == v1] = v0

            ######
            # Edge handling
            ######
            def update_neighbors(e0, keep, remove, mesh):
                keep_neigh = self.get_remaining_neighbors(mesh.gemm_edges[keep], [e0, remove])
                remove_neigh = self.get_remaining_neighbors(mesh.gemm_edges[remove], [e0, keep])
                keep_neigh.extend(remove_neigh)

                if len(keep_neigh) != 4:
                    self.logger.debug(f"WARNING: Bad neighbor count for edge {keep} after update.")
                    self.logger.debug(f"e0 gemm: {mesh.gemm_edges[e0]}")
                    self.logger.debug(f"keep gemm: {mesh.gemm_edges[keep]}")
                    self.logger.debug(f"remove gemm: {mesh.gemm_edges[remove]}")

                return torch.LongTensor(keep_neigh)

            neighbors = mesh.gemm_edges[e0].clone()
            self.logger.debug(f"Neighbors of {e0}: {neighbors}")

            def reorder(e1, e2, e3, e4, e0, mesh):
                e1_neigh = self.get_remaining_neighbors(mesh.gemm_edges[e1], [e0, e2])
                e2_neigh = self.get_remaining_neighbors(mesh.gemm_edges[e2], [e0, e1])
                e3_neigh = self.get_remaining_neighbors(mesh.gemm_edges[e3], [e0, e4])
                e4_neigh = self.get_remaining_neighbors(mesh.gemm_edges[e4], [e0, e3])

                if len(self.__get_shared_items(e2_neigh, e4_neigh)) != 0:
                    e1, e2 = e2, e1
                    e3, e4 = e4, e3
                elif len(self.__get_shared_items(e2_neigh, e3_neigh)) != 0:
                    e1, e2 = e2, e1
                elif len(self.__get_shared_items(e1_neigh, e4_neigh)) != 0:
                    e3, e4 = e4, e3
                return e1, e2, e3, e4

            e1, e2, e3, e4 = reorder(*[int(n.item()) for n in neighbors], e0, mesh)
            self.logger.debug(f"Reordered neighbors: {e1}, {e2}, {e3}, {e4}")

            try:
                mesh.gemm_edges[e1] = update_neighbors(e0, e1, e2, mesh)
                mesh.gemm_edges[e3] = update_neighbors(e0, e3, e4, mesh)
            except Exception as ex:
                self.logger.debug(f"Exception updating neighbors: {ex}")
                self.logger.debug(f"Trying to prune {e0}")
                self.logger.debug(f"e0 neighbors: {e1}, {e2}, {e3}, {e4}")

            mesh.gemm_edges[e2] = -1.0
            mesh.gemm_edges[e4] = -1.0
            mesh.gemm_edges[e0] = -1.0

       

            mesh.edges[e2] = -1.0
            mesh.edges[e4] = -1.0
            mesh.edges[e0] = -1.0

    
            self.pruned_edges[e2] = False
            self.pruned_edges[e0] = False
            self.pruned_edges[e4] = False

            mesh.gemm_edges[mesh.gemm_edges == float(e2)] = float(e1)
            mesh.gemm_edges[mesh.gemm_edges == float(e4)] = float(e3)

          
            self.logger.debug(f"Pruning Check: {e0} {pruning_check_e0} Final {mesh.gemm_edges[e0]}")
            self.logger.debug(f"Pruning Check: {e1} {pruning_check_e1} Final {mesh.gemm_edges[e1]}")
            self.logger.debug(f"Pruning Check: {e2} {pruning_check_e2} Final {mesh.gemm_edges[e2]}")
            self.logger.debug(f"Pruning Check: {e3} {pruning_check_e3} Final {mesh.gemm_edges[e3]}")
            self.logger.debug(f"Pruning Check: {e4} {pruning_check_e4} Final {mesh.gemm_edges[e4]}")
            ######
            # Future handling
            ######
            mesh.x[e1] = (mesh.x[e0] + mesh.x[e1] + mesh.x[e2]) / 3.0
            mesh.x[e3] = (mesh.x[e0] + mesh.x[e2] + mesh.x[e3]) / 3.0

           

            mesh.x[e0] = 0
            mesh.x[e2] = 0
            mesh.x[e4] = 0

    

            mesh.edges_count -= 3
            self.logger.debug(f"New edges count: {mesh.edges_count}")
            e1_neigh = mesh.gemm_edges[e1]
            e3_neigh = mesh.gemm_edges[e3]
            neigh_tensors = [e1_neigh]
            for n1_n in e1_neigh:
                if n1_n.item() != -1.0:
                    assert np.where(mesh.gemm_edges[int(n1_n)] == float(e1), 1, 0).sum() == 1, (n1_n, e1)
           
           
        
    
            for n1_n in e3_neigh:
                if n1_n.item() != -1.0:
                    assert np.where(mesh.gemm_edges[int(n1_n)] == float(e3), 1, 0).sum() == 1, (n1_n, e1)
                    
            for edge in torch.unique(mesh.gemm_edges):
                if edge != -1.0:
                    for neigh in mesh.gemm_edges[edge]:
                        if neigh != -1.0:
                            if edge not in mesh.gemm_edges[neigh]:
                                self.logger.error(f"Edge {edge} {neigh}  Neighbour {neigh} {mesh.gemm_edges[neigh]}" )
                            
                                raise ValueError('Invalid Neighbour Configuration found')
          
          
    

        self.logger.debug(f"=== FINISHED PRUNING EDGE {e0} ===\n")
      
        return mesh
    
    
    def check_consistency(self, mesh):
        _, counts = torch.unique(mesh.gemm_edges, return_counts=True)
        for i,c in zip(_,counts): 
            if i != -1.0 and c not in [2,4]:
                raise ValueError(f'Found inconsistent neighbour configuration != [2,4] {i} with count {c}')
        
    def reconstruct_faces(self,mesh):
        valid_faces = []
        face__keys = []
        edge_pairs = []
        for edge_idx, [p1, p2, p3, p4] in enumerate(mesh.gemm_edges.cpu().tolist()):
            
            if -1.0 not in [p1,p2]:
                ed1 = sorted([edge_idx, p1])
                ed2 = sorted([edge_idx, p2])
                edge_pairs.append(f"{ed1[0]}_{ed1[1]}")
                edge_pairs.append(f"{ed2[0]}_{ed2[1]}")
                cand_fa = sorted([int(edge_idx), int(p1), int(p2)])
                if f"{cand_fa[0]}_{cand_fa[1]}_{cand_fa[2]}" not in face__keys:
                    face__keys.append(f"{cand_fa[0]}_{cand_fa[1]}_{cand_fa[2]}")
                    valid_faces.append(cand_fa)
            if not -1.0 in [p3,p4]:
                ed1 = sorted([edge_idx, p3])
                ed2 = sorted([edge_idx, p4])
                edge_pairs.append(f"{ed1[0]}_{ed1[1]}")
                edge_pairs.append(f"{ed2[0]}_{ed2[1]}")
                cand_fa = sorted([int(edge_idx), int(p3), int(p4)])
                if f"{cand_fa[0]}_{cand_fa[1]}_{cand_fa[2]}" not in face__keys:
                    face__keys.append(f"{cand_fa[0]}_{cand_fa[1]}_{cand_fa[2]}")
                    valid_faces.append(cand_fa)
                    
        face_vertices = []
        for v_face in valid_faces:
            vertices = []
            e0, e1, e2 = v_face 
            vertices.extend(mesh.edges[e0].tolist())
            vertices.extend(mesh.edges[e1].tolist())
            vertices.extend(mesh.edges[e2].tolist())
            vertices = sorted(list(set(vertices)))
            face_vertices.append(vertices)
        
        def check_export(face_vertices):
            unique_keys = []
            retrived_edges = []
            for (v1,v2,v3) in face_vertices:
                
                for v11, v12 in itertools.combinations((v1,v2,v3), 2):
                    edge = sorted([v11, v12])
        
                 
                    if not f"{edge[0]}_{edge[1]}" in unique_keys:
                        unique_keys.append(f"{edge[0]}_{edge[1]}")
                        retrived_edges.append(edge)
        check_export(face_vertices) 
        return mesh.vertices, np.array(face_vertices)
    


        
        
        
        
        
        
        
        
        
        
        
            
        
        