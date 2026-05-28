import os
import torch
import torch.optim as optim
import torch.nn as nn
from .model import GNNModel
from .preprocessor import GeometryProcessor

class AIRetopoEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = GNNModel(input_dim=7).to(self.device)
        
        self.weights_path = "weights/model_v1.pth"
        if os.path.exists(self.weights_path):
            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=True))
                print(f"ИИ: Веса успешно загружены из {self.weights_path}")
            except Exception as e:
                print(f"ИИ: Не удалось загрузить веса: {e}")
        else:
            print("ИИ: Файл весов не найден, используется случайная инициализация.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.model_id = "v1_base"

    def inference(self, mesh):
        self.model.eval()
        with torch.no_grad():
            x = GeometryProcessor.compute_features(mesh).to(self.device)
            edges = mesh.edges 
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
            weights = self.model(x, edge_index)
            return weights.cpu().detach().numpy()

    def train_step(self, mesh, target_importance):
        self.model.train()
        
        x = GeometryProcessor.compute_features(mesh).to(self.device)
        
        edges = mesh.edges
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        
        pred = self.model(x, edge_index)
        
        y = torch.tensor(target_importance, dtype=torch.float32).to(self.device).view_as(pred)
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"ИИ: Веса принудительно сохранены в {self.weights_path}")