import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from pyvistaqt import QtInteractor

class GLViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)
        self.plotter.set_background("#2c3e50")

    def set_mesh(self, mesh, color="#ecf0f1", is_heatmap=False, reference=None):
        self.plotter.clear()
    
        faces = np.hstack(np.pad(mesh.faces, ((0, 0), (1, 0)), constant_values=3))
        poly = pv.PolyData(mesh.vertices, faces)
        
        if is_heatmap and reference:
            from scipy.spatial import cKDTree
            tree = cKDTree(reference.vertices)
            dist, _ = tree.query(mesh.vertices)
            poly.point_data["Error"] = dist
            self.plotter.add_mesh(poly, scalars="Error", cmap="jet", show_edges=True, edge_color="#2c3e50")
            self.plotter.add_scalar_bar(title="Distance Error")
        else:
            self.plotter.add_mesh(poly, 
                                 color=color, 
                                 show_edges=True, 
                                 edge_color="#2c3e50", 
                                 line_width=1,
                                 smooth_shading=True)
        self.plotter.reset_camera()
        self.plotter.render()