import pymongo
import gridfs
from datetime import datetime
import os

class DBManager:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            self.db = self.client["retopo_studio"]
            self.fs = gridfs.GridFS(self.db)
            self.client.server_info() 
            print("БД: Успешно подключено. Схема активирована.")
        except Exception as e:
            self.db = None
            print(f"БД: Ошибка подключения: {e}")

    def create_project(self, name, description=""):
        if self.db is None: return None
        
        project = {
            "name": name,
            "description": description,
            "created_at": datetime.now()
        }
        return self.db.projects.insert_one(project).inserted_id

    def save_source_mesh(self, project_id, mesh, filename):
        if self.db is None or project_id is None: return None
        
        file_data = mesh.export(file_type='glb')
        gridfs_id = self.fs.put(file_data, filename=filename)
        
        ext = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
        
        source_data = {
            "project_id": project_id,
            "filename": filename,
            "poly_count": len(mesh.faces),
            "format": ext,
            "gridfs_id": gridfs_id
        }
        return self.db.source_meshes.insert_one(source_data).inserted_id

    def register_ai_model(self, version_name, weights_path):
        if self.db is None: return None
        
        existing = self.db.ai_models.find_one({"version": version_name})
        if existing:
            return existing["_id"]
            
        weights_id = None
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                weights_id = self.fs.put(f.read(), filename=f"{version_name}.pth")
                
        ai_model_data = {
            "version": version_name,
            "weights_id": weights_id
        }
        return self.db.ai_models.insert_one(ai_model_data).inserted_id

    def save_processing_result(self, project_id, model_id, mesh, target_poly_count, rmse, exec_time):
        if self.db is None or project_id is None: return None
        
        file_data = mesh.export(file_type='glb')
        output_mesh_id = self.fs.put(file_data, filename="result.glb")
        
        result_data = {
            "project_id": project_id,
            "model_id": model_id,
            "target_poly_count": int(target_poly_count),
            "final_poly_count": len(mesh.faces),
            "rmse_error": float(rmse),
            "execution_time": float(exec_time),
            "output_mesh_id": output_mesh_id
        }
        return self.db.processing_results.insert_one(result_data).inserted_id