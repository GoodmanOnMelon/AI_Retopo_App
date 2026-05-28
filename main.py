import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())