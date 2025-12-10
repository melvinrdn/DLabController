# src/dlab/diagnostics/ui/scan_window.py
from __future__ import annotations
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
)

from dlab.diagnostics.ui.scans.m2_measurement_tab import M2Tab
from dlab.diagnostics.ui.scans.grid_scan_tab import GridScanTab
from dlab.diagnostics.ui.scans.grating_compressor_scan_tab import GCScanTab
from dlab.diagnostics.ui.scans.temporal_overlap_scan_tab import TOverlapTab
from dlab.diagnostics.ui.scans.two_color_scan_tab import TwoColorScanTab

import logging
logger = logging.getLogger("dlab.ui.ScanWindow")


class ScanWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scan")
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        self.tabs.addTab(GridScanTab(), "Grid scan")
        self.tabs.addTab(GCScanTab(), "Grating compressor scan")
        self.tabs.addTab(M2Tab(), "M2 Scan")
        self.tabs.addTab(TOverlapTab(), "Temporal overlap scan")
        self.tabs.addTab(TwoColorScanTab(), "Two-Color Scan")
        
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
