# src/dlab/diagnostics/ui/scan_window.py
from __future__ import annotations
import os, datetime, time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QComboBox, QTabWidget, QDoubleSpinBox,
    QGroupBox, QMessageBox, QSpinBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
from dlab.diagnostics.ui.scans.m2_measurement_tab import M2Tab

def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


# ---------- worker ----------
import logging
logger = logging.getLogger("dlab.ui.ScanWindow")


# ---------- main window ----------
class ScanWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scan")
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        self.tabs.addTab(M2Tab(), "M2 measurement")

        # Future: add more tabs easily
        # self.tabs.addTab(SomeOtherScanTab(), "Whatever")
        
    def closeEvent(self, event):
        try:
            self.closed.emit()
        finally:
            super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = ScanWindow()
    w.show()
    sys.exit(app.exec_())
