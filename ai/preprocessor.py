import numpy as np
import torch
import trimesh

class GeometryProcessor:
    @staticmethod
    def compute_features(mesh: trimesh.Trimesh):
        centroid = mesh.vertices.mean(axis=0)
        vertices = mesh.vertices - centroid
        max_dist = np.max(np.linalg.norm(vertices, axis=1)) + 1e-6
        vertices /= max_dist
        
        normals = mesh.vertex_normals
        
        try:
            curvature = np.abs(mesh.vertex_defects).reshape(-1, 1)
        except:
            curvature = np.zeros((len(mesh.vertices), 1))
            
        curvature = np.nan_to_num(curvature)
        
        features = np.hstack([vertices, normals, curvature])
        return torch.tensor(features, dtype=torch.float32)