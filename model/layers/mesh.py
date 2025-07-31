import trimesh
import itertools 
import numpy as np 
import torch
import torch.nn as nn 
from pathlib import Path 
from collections import OrderedDict,defaultdict
"""
So I get the vertices and faces from the mesh (stl file)
Steps to do: 
1. I need to extract all the edges in the mesh 
2. For each edge i need to get all its neighborring edges
3. Edges need to be put into the categories valid / invalid depending if they have 2 or 4 neighboring edges
4. For each edge i need to compute the features 
    4.1 Dihedral angle 
    4.2 inner angle 
    4.3 Edge length ratio
    4.4 How do deal with the invalid edges, setting values to 0 i suppose?

"""

class Mesh(): 
    def __init__(self, gemm_edges, features, edges, vertices):
        super(Mesh, self).__init__()
        self.gemm_edges = gemm_edges
        self.edges_count = len(gemm_edges)
        self.x = features
        self.edges = edges
        self.vertices = vertices

class MeshLoader(nn.Module):
    def __init__(self):
        super(MeshLoader, self).__init__()
        
    def forward(self, mesh_files, mode='train'):
        return self.feature_extractor(files=mesh_files)
    
    
    def feature_extractor(self, dir_name=None, files=None, mode='train'):
        ######
        # Inner Helper Functions
        ######
        def __get_edges(faces):
                edges = []
                for face in faces:
                    edges.extend(itertools.combinations(face, 2))
                return  set(edges)
    
        def __get_ordered_edge_neighbors(edges, faces):
            """
            For each edge, return 4 neighbors ordered as:
            [f1, f2] from face A, [f3, f4] from face B
            """
            # Map from face to its three edges
            face_edge_mapping = defaultdict(list)
            # Map from edge to list of faces it belongs to
            edge_face_mapping = defaultdict(list)

            for edge_id, (v1, v2) in edges.items():
                for face_id, verts in faces.items():
                    if v1 in verts and v2 in verts:
                        face_edge_mapping[face_id].append(edge_id)
                        edge_face_mapping[edge_id].append(face_id)

            edge_neighbor_mapping = {}

            for edge_id in edges:
                neighbor_edges = []

                face_ids = edge_face_mapping[edge_id]
                for face_id in face_ids:
                    face_edges = face_edge_mapping[face_id]
                    # Get the other 2 edges in this face
                    others = [eid for eid in face_edges if eid != edge_id]
                    if len(others) != 2:
                        raise ValueError(f"Face {face_id} does not have 3 edges. Something went wrong.")
                    neighbor_edges.extend(others)

                # Sort order: [f1, f2, f3, f4] = neighbors from face A then face B
                if len(neighbor_edges) < 4:
                    # Mesh boundaries will yield <4 neighbors; pad with -1
                    neighbor_edges += [-1] * (4 - len(neighbor_edges))

                edge_neighbor_mapping[edge_id] = neighbor_edges[:4]

            return edge_neighbor_mapping
    
        def __get_edge_neighboor(edges, faces):
            """
            This function is used to extract the neightboor id edges for each edge in the dataset.
            It returns key (edge ids) value (list of edge neighboor) pairs. 
            """
            edge_face_mapping ={}
            face_edge_mapping = {}
            for edge_id, (e1,e2) in edges.items():
                attached_faces =[]
                for face_id, f in faces.items():
                    if e1 in f and e2 in f:
                        attached_faces.append(face_id)
                        if face_id not in face_edge_mapping:
                            face_edge_mapping[face_id] = []
                        face_edge_mapping[face_id].append(edge_id)
                        
                edge_face_mapping[edge_id] = attached_faces
            edge_neighboor_mapping = {}
            
            for edge, face_ids in edge_face_mapping.items():
                neighboors = []
                for face_id in face_ids:
                    for neigh in face_edge_mapping[face_id]:
                        if neigh != edge:
                            neighboors.append(neigh)
                edge_neighboor_mapping[edge] = neighboors
            return edge_neighboor_mapping 
    
        
        def __unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    return vector / np.linalg.norm(vector)
                except RuntimeWarning:
                    return np.zeros(vector.shape)

        def __angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    >>> angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    >>> angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            """
            v1_u = __unit_vector(v1)
            v2_u = __unit_vector(v2)
            
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        
        def __get_inner_angle(ed1, ed2, edges, vertices):
            vert1 = edges[ed1]
            vert2 = edges[ed2]
            
            
            a = vertices[vert1[0]] - vertices[vert1[1]]
            b = vertices[vert2[0]] - vertices[vert2[1]]
            return __angle_between(a,b)
        
        
        def __get_dihedral_angle(face1, face2, edges, vertices):
            
            norm_vectors = []
            for (ed1, ed2) in [face1, face2]:
                vert1 = edges[ed1]
                vert2 = edges[ed2]
                a = vertices[vert1[0]] - vertices[vert1[1]]
                b = vertices[vert2[0]] - vertices[vert2[1]]
                norm_vectors.append(np.cross(a,b))
            return __angle_between(*norm_vectors)
    
        def __edge_length(v1, v2, vertices):
            return np.linalg.norm(vertices[v1] - vertices[v2])

        def __get_edge_length_ratio(edge_id, face_edge_ids, edges, vertices):
            center_edge = edges[edge_id]
            e0 = __edge_length(*center_edge, vertices)

            # get the two other edge lengths in the triangle
            e1 = __edge_length(*edges[face_edge_ids[0]], vertices)
            e2 = __edge_length(*edges[face_edge_ids[1]], vertices)

            return max(e1, e2) / e0 if e0 != 0 else 0.0

        
        def __mesh_alternations(mesh):
            
            def __grow_connected_faces(mesh, start_face_idx, n_faces_target):
                face_adjacency = mesh.face_adjacency 
                adjacncy_dict = {i : set() for i in range(len(mesh.faces))}
                for f1, f2 in face_adjacency:
                    adjacncy_dict[f1].add(f2)
                    adjacncy_dict[f2].add(f1)
                selected_faces = set([start_face_idx])
                frontier = set([start_face_idx])
                while len(selected_faces) < n_faces_target and frontier:
                    current = frontier.pop()
                    neighbors = adjacncy_dict[current] - selected_faces
                    for neighbor in neighbors:
                        if len(selected_faces) < n_faces_target:
                            selected_faces.add(neighbor)
                            frontier.add(neighbor)
                        else:
                            break
                return list(selected_faces)
            
            
            def random_shift(mesh):
                import random 
                m = mesh.copy()
                n = random.randint(5, 15)
                start = random.randint(0, len(mesh.faces) - 1)
                grown_faces = __grow_connected_faces(mesh, start, n)
                face_vertices = m.faces[grown_faces].flatten()
                unique_vertices = np.unique(face_vertices)
                shift = np.random.uniform( -0.05, 0.05, size=3)
                m.vertices[unique_vertices] += shift
                return m
            
            
            for i in range(30):
                mesh = random_shift(mesh)
            
            m = mesh.copy()

            noise = np.clip(np.random.normal(0, 0.005, m.vertices.shape), -0.01, 0.01)
            m.vertices += noise
            return m 
        
        
        def __feature_extractor(edge_neighboor_mapping, edges, vertices):
            """
            This function extracts the futures of every edge. These are:
                1. Inner angles of associated faces
                2. Digedral angle between the two faces 
                3. Length ratio of the triangle
            If the edge is a boundary edge, these values will be except for the one inner angle be set to 0.0 
            
            """
            edge_features = {}
            # Going over every single edge
            for edge, neighboors in edge_neighboor_mapping.items():
                neighboors = np.array(list(neighboors))
                # Case: Two Faces attached to this edge
                if len(neighboors) == 4:
                    face_ids = [neighboors[:2], neighboors[2:]]
                    face_ids.sort(key=lambda x: tuple(sorted(x)))  # Sort faces lexicographically
                    face_1, face_2 = face_ids
        
                    inner_1 = __get_inner_angle(face_1[0], face_1[1], edges, vertices)
                    inner_2 = __get_inner_angle(face_2[0], face_2[1], edges, vertices)
                    inner_1, inner_2 = sorted([inner_1, inner_2])
                    dihedral = __get_dihedral_angle(face_1, face_2, edges, vertices)
                    edge_ratio_1 = __get_edge_length_ratio(edge, face_1, edges, vertices)
                    edge_ratio_2 = __get_edge_length_ratio(edge, face_2, edges, vertices)
                    edge_ratio_1, edge_ratio_2 = sorted([edge_ratio_1, edge_ratio_2])
                    
                    edge_features[edge] = [inner_1, inner_2, dihedral, edge_ratio_1, edge_ratio_2]
                # Case: Boundary edge only attached to one face 
                elif len(neighboors) == 2:
                    inner_1 = __get_inner_angle(neighboors[0], neighboors[1], edges, vertices)
                    edge_ratio_1 = __get_edge_length_ratio(edge, neighboors, edges, vertices)
                    edge_features[edge] = [inner_1, 0.0, 0.0, edge_ratio_1, 0.0]
        
            return edge_features
        
        if not files:
            path = Path(dir_name)
            files = list(path.iterdir())  
        # We iterate over all the files 
        meshes = []
        for file in files: 
            
                mesh_obj = trimesh.load(file)  
                        
                mesh_objs = [mesh_obj]
                if mode =='train':
                    mesh_objs.append(__mesh_alternations(mesh_objs[0]))
                    mesh_objs[0].export('original.stl')
                    mesh_objs[1].export('alternated.stl')
                for mesh_obj in mesh_objs:
                    vertices = mesh_obj.vertices 
                    vertex_id = {i: v for i,v in enumerate(vertices)}
                    faces = mesh_obj.faces 
                    # We need to assign a unique id to each face for processing
                    faces_id = {i: f for (i,f) in enumerate(faces)}
                    edges = mesh_obj.edges_unique
                
                    # Again edges are assigned a unique id 
                    edges_id = {i: e for (i,e) in enumerate(edges)}
                    #edge_neighboors = get_edge_neighboor(edges_id, faces_id)
                    edge_neighboors = __get_edge_neighboor(edges_id, faces_id)
                    n_tensor = []
                    for i,edge_neighboor in edge_neighboors.items():
                        #print(i, len(edge_neighboor))
                        if len(edge_neighboor) == 4:
                            n_tensor.append(list(edge_neighboor))
                        # Boundary edge
                        elif len(edge_neighboor) == 2: 
                            n_tensor.append([*edge_neighboor, -1, -1])
                        else:
                            print('Error ', edge_neighboor)
                    n_tensor = np.vstack(n_tensor)

                    features = __feature_extractor(edge_neighboors, edges_id, vertex_id)

                    features = np.vstack(list(features.values()))
                    mesh = Mesh(gemm_edges=torch.from_numpy(n_tensor.copy()), features=torch.from_numpy(features.copy()), edges=torch.from_numpy(edges.copy()), vertices=vertices)
                    meshes.append(mesh)
                    
        # First pass: find the max number of nodes (edges)
        all_features = []
        max_edges = 0
        for mesh in meshes:
            if mesh.x.shape[0] > max_edges:
                max_edges = mesh.x.shape[0]
            all_features.append(mesh.x)

        # Second pass: pad each tensor to match max_edges
        padded_features = []
        for feat in all_features:
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
        edge_features = edge_features.type(torch.LongTensor)
        

        return edge_features, meshes
    
 
            
            
            
    
            
    
       
''' def read_meshes(files):
        for file in files: 
            mesh = trimesh.load(file)  
            is_manifold = mesh.is_winding_consistent  # True if oriented and 2-manifold

            # Check for boundary edges
            boundary_edges = mesh.facets_boundary
            has_boundary = len(boundary_edges) > 0
         

            print(f"Manifold: {is_manifold}")
            print(f"Boundary: {has_boundary}")

            vertices = mesh.vertices 
'''
#feature_extractor("/Users/jan/Documents/Microns/CLEAN CODE/spines")