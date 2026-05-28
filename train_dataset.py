import os
import glob
import trimesh
import torch
import numpy as np
import random
import warnings
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

print("--- ЗАПУСК ПОШАГОВОЙ ДИАГНОСТИКИ ОБУЧЕНИЯ GNN ---")

from ai.engine import AIRetopoEngine
from ai.preprocessor import GeometryProcessor

DATASET_PATH = r"D:\AI_Retopo_App\dataset\ModelNet40" 
WEIGHTS_PATH = "weights/model_v1.pth"
MAX_VERTICES = 15000 

def generate_target_importance(mesh):
    start = time.time()
    try:
        curvature = np.abs(mesh.vertex_defects)
        curvature = np.nan_to_num(curvature)
        
        c_min, c_max = curvature.min(), curvature.max()
        if c_max > c_min:
            curvature = (curvature - c_min) / (c_max - c_min)
        else:
            curvature = np.zeros_like(curvature)
        
        calc_time = time.time() - start
        return curvature.reshape(-1, 1), calc_time
    except Exception as e:
        return np.zeros((len(mesh.vertices), 1)), 0

def train():
    if os.path.exists(WEIGHTS_PATH):
        print(f"ВНИМАНИЕ: Обнаружены старые веса '{WEIGHTS_PATH}'. Удаляем для новой архитектуры...")
        try:
            os.remove(WEIGHTS_PATH)
        except Exception as e:
            print(f"Не удалось удалить файл весов: {e}")

    engine = AIRetopoEngine()
    
    search_pattern = os.path.join(DATASET_PATH, "**", "train", "*.off")
    files = glob.glob(search_pattern, recursive=True)
    random.shuffle(files)
    
    print(f"Всего файлов найдено: {len(files)}. Начинаем обучение...")

    for epoch in range(3):
        print(f"\n=============================")
        print(f">>> СТАРТ ЭПОХИ {epoch+1}")
        print(f"=============================")

        for i, file_path in enumerate(files):
            fname = os.path.basename(file_path)
            
            print(f"[{i}/{len(files)}] {fname} | Загрузка...", end=" ", flush=True)
            try:
                mesh = trimesh.load(file_path, process=False)
                
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                    if isinstance(mesh, (list, tuple, np.ndarray)):
                        mesh = mesh[0]
                
                v_count = len(mesh.vertices)
                e_count = len(mesh.edges) if hasattr(mesh, 'edges') else 0
                
                if v_count == 0 or v_count > MAX_VERTICES or e_count == 0:
                    print(f"ПРОПУСК (вершин: {v_count}, ребер: {e_count}).")
                    continue
                
                print(f"({v_count} v, {e_count} e) | Кривизна...", end=" ", flush=True)
                target, c_time = generate_target_importance(mesh)
                if target is None:
                    print("ОШИБКА МАТЕМАТИКИ.")
                    continue
                
                print(f"[{c_time:.2f}s] | GPU...", end=" ", flush=True)
                loss = engine.train_step(mesh, target)
                
                print(f"OK | Loss: {loss:.6f}")

                if i > 0 and i % 50 == 0:
                    engine.save_weights() 

            except KeyboardInterrupt:
                print("\nОстановка пользователем. Прогресс сохранен (если пройдено >50 шагов).")
                return
            except Exception as e:
                print(f"ОШИБКА: {e}")
                continue

if __name__ == "__main__":
    train()