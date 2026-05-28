import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from .viewport import GLViewport
from core.controller import RetopoAppController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Retopo Studio Pro")
        self.resize(1300, 800)
        self.controller = RetopoAppController(self)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        self.sidebar = self._setup_sidebar()
        layout.addWidget(self.sidebar)
        
        v_area = QVBoxLayout()
        self.viewport = GLViewport()
        v_area.addWidget(self.viewport)
        
        self.mode_panel = self._setup_mode_panel()
        v_area.addWidget(self.mode_panel)
        layout.addLayout(v_area, stretch=1)

    def _setup_sidebar(self):
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setStyleSheet("background-color: #2c3e50; color: white;")
        layout = QVBoxLayout(frame)
        
        self.btn_open = QPushButton("OPEN MODEL")
        self.btn_open.setStyleSheet("background-color: #34495e; padding: 10px;")
        self.btn_open.clicked.connect(self.load_file)
        layout.addWidget(self.btn_open)

        self.btn_export = QPushButton("EXPORT AI MODEL")
        self.btn_export.setStyleSheet("background-color: #27ae60; padding: 10px; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_file)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)
        
        layout.addSpacing(20)
        
        self.lbl_info = QLabel("Faces: 0\nEdges: 0\nVerts: 0")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #f1c40f;")
        layout.addWidget(self.lbl_info)

        layout.addSpacing(20)
        self.lbl_slider = QLabel("Reduction %: 50%")
        layout.addWidget(self.lbl_slider)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(lambda v: self.lbl_slider.setText(f"Reduction %: {v}%"))
        layout.addWidget(self.slider)
        
        layout.addStretch()
        
        self.btn_run = QPushButton("RUN PROCESS")
        self.btn_run.setStyleSheet("background-color: #e74c3c; padding: 15px; font-weight: bold;")
        self.btn_run.clicked.connect(self.process)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)
        
        return frame

    def _setup_mode_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background-color: #34495e;")
        layout = QHBoxLayout(panel)
        
        self.btn_orig = QPushButton("ORIGINAL")
        self.btn_math = QPushButton("MATH RESULT")
        self.btn_ai = QPushButton("AI RESULT")
        
        self.group = QButtonGroup(self)
        for b in[self.btn_orig, self.btn_math, self.btn_ai]: 
            b.setCheckable(True)
            b.setStyleSheet("padding: 10px; color: white;")
            self.group.addButton(b)
            layout.addWidget(b)
            
        self.btn_orig.setChecked(True)
        
        self.btn_math.setEnabled(False)
        self.btn_ai.setEnabled(False)
        
        self.btn_orig.clicked.connect(lambda: self.display_mesh_and_info(self.controller.mesh_original))
        self.btn_math.clicked.connect(lambda: self.display_mesh_and_info(self.controller.mesh_math, "#3498db", True, self.controller.mesh_original))
        self.btn_ai.clicked.connect(lambda: self.display_mesh_and_info(self.controller.mesh_ai, "#f1c40f", True, self.controller.mesh_original))
        return panel

    def display_mesh_and_info(self, mesh, color="#ecf0f1", is_heatmap=False, reference=None):
        if mesh is None: return
        
        self.viewport.set_mesh(mesh, color, is_heatmap, reference)
        
        faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        verts = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0

        edges = len(mesh.edges_unique) if hasattr(mesh, 'edges_unique') else 0
        
        self.lbl_info.setText(f"Faces: {faces}\nEdges: {edges}\nVerts: {verts}")

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", "3D Files (*.obj *.ply *.stl *.glb)")
        if path:
            mesh = self.controller.open_file(path)
            self.btn_orig.setChecked(True)
            
            self.display_mesh_and_info(mesh)
            
            self.btn_run.setEnabled(True)
            
            self.btn_export.setEnabled(False)
            self.btn_math.setEnabled(False)
            self.btn_ai.setEnabled(False)

    def process(self):
        val = self.slider.value()
        self.statusBar().showMessage("Идет обработка... Пожалуйста, подождите.")
        self.btn_run.setEnabled(False)
        QApplication.processEvents()
        
        try:
            m_math, m_ai, rmse, t = self.controller.run_process(val)
            self.btn_ai.setChecked(True)
            
            self.display_mesh_and_info(m_ai, "#f1c40f", True, self.controller.mesh_original)
            
            self.btn_export.setEnabled(True)
            self.btn_math.setEnabled(True)
            self.btn_ai.setEnabled(True)
            
            QMessageBox.information(self, "Готово", f"Обработка завершена!\nВремя: {t:.2f} сек.\nRMSE: {rmse:.6f}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка обработки", str(e))
        finally:
            self.btn_run.setEnabled(True)
            self.statusBar().clearMessage()

    def export_file(self):
        if self.controller.mesh_ai is None:
            QMessageBox.warning(self, "Внимание", "Сначала запустите обработку (RUN PROCESS)!")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Экспорт AI-Ретопологии", 
            "AI_retopo_result.obj", 
            "OBJ Files (*.obj);;STL Files (*.stl);;PLY Files (*.ply);;GLB Files (*.glb)"
        )
        
        if path:
            try:
                self.controller.export_ai_mesh(path)
                QMessageBox.information(self, "Успех", f"ИИ-модель успешно сохранена!\nФайл: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", f"Не удалось сохранить файл: {e}")