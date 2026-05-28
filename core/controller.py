import time
import os
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import pyvista as pv  
from ai.engine import AIRetopoEngine
from core.file_handler import FileHandler
from data.db_manager import DBManager

class RetopoAppController:
    def __init__(self, view):
        self.view = view
        self.engine = AIRetopoEngine()
        self.db = DBManager()
        self.mesh_original = None
        self.mesh_math = None
        self.mesh_ai = None
        self.project_id = None
        self.ai_model_id = None 

    def open_file(self, path):
        try:
            self.mesh_original = FileHandler.load_mesh(path)
            
            self.mesh_math = None
            self.mesh_ai = None
            
            filename = os.path.basename(path)
            if self.db:

                self.project_id = self.db.create_project(filename, "Обработка через UI")

                self.db.save_source_mesh(self.project_id, self.mesh_original, filename)

                self.ai_model_id = self.db.register_ai_model("v1_base", "weights/model_v1.pth")
            return self.mesh_original
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            raise e

    def run_process(self, target_pct):
        if self.mesh_original is None:
            return None, None, 0, 0
        
        start_time = time.time()
        
        factor = target_pct / 100.0
        reduction = 1.0 - factor
        
        faces_pv = np.hstack(np.pad(self.mesh_original.faces, ((0, 0), (1, 0)), constant_values=3))
        poly = pv.PolyData(self.mesh_original.vertices, faces_pv)

        print(f"Запуск базовой децимации (удаляем {reduction*100:.1f}%)...")
        math_poly = poly.decimate(target_reduction=reduction, volume_preservation=True, scalars=False)
        
        math_faces = math_poly.faces.reshape(-1, 4)[:, 1:]
        self.mesh_math = trimesh.Trimesh(vertices=math_poly.points, faces=math_faces, process=False)

        print("ИИ: Получение карты важности...")
        weights = self.engine.inference(self.mesh_original).flatten()
        
        smoothed_weights = weights.copy()
        try:
            adj = self.mesh_original.edges_sparse
            for _ in range(3):
                sum_w = adj.dot(smoothed_weights)
                counts = np.array(adj.sum(axis=1)).flatten()
                smoothed_weights = sum_w / (counts + 1e-6)
        except:
            pass

        print("ИИ: Умная интеграция весов в геометрию...")
        ai_poly = poly.copy()
        
        ai_poly.point_data["AI_Importance"] = smoothed_weights
        ai_poly.set_active_scalars("AI_Importance")
        
        res_poly = ai_poly.decimate(target_reduction=reduction, volume_preservation=True, scalars=True)
        
        ai_faces = res_poly.faces.reshape(-1, 4)[:, 1:]
        self.mesh_ai = trimesh.Trimesh(vertices=res_poly.points, faces=ai_faces, process=False)

        exec_time = time.time() - start_time
        
        try:
            tree = cKDTree(self.mesh_original.vertices)
            dist, _ = tree.query(self.mesh_ai.vertices)
            rmse = np.sqrt(np.mean(dist**2))
        except:
            rmse = 0.0

        if self.db and self.project_id:
            target_faces = int(len(self.mesh_original.faces) * factor)
            self.db.save_processing_result(
                project_id=self.project_id,
                model_id=self.ai_model_id,
                mesh=self.mesh_ai,
                target_poly_count=target_faces,
                rmse=float(rmse),
                exec_time=exec_time
            )
        
        print(f"Готово! Время: {exec_time:.2f}с, RMSE: {rmse:.6f}")
        return self.mesh_math, self.mesh_ai, rmse, exec_time

    def export_ai_mesh(self, path):
        if self.mesh_ai is None:
            raise ValueError("Нет готовой ИИ-модели.")
        FileHandler.save_mesh(self.mesh_ai, path)
        
    def export_mesh(self, mode, path):
        mesh_to_save = None
        if mode == "ORIGINAL": mesh_to_save = self.mesh_original
        elif mode == "MATH": mesh_to_save = self.mesh_math
        elif mode == "AI": mesh_to_save = self.mesh_ai

        if mesh_to_save is None:
            raise ValueError("Нет данных для экспорта.")
        FileHandler.save_mesh(mesh_to_save, path)