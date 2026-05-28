import trimesh

class FileHandler:
    @staticmethod
    def load_mesh(path):
        mesh = trimesh.load(path)
        return mesh.to_geometry() if hasattr(mesh, 'to_geometry') else mesh

    @staticmethod
    def save_mesh(mesh, path):
        mesh.export(path)