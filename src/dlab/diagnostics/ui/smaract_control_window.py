from __future__ import annotations

import sys
from typing import Dict, Tuple

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt, QDateTime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QToolButton, QPushButton, QMessageBox, QTextEdit
)

from dlab.hardware.wrappers.smaract_controller import SmarActMCS


class MoveWorker(QObject):
    finished = pyqtSignal(int)
    error = pyqtSignal(int, str)
    started = pyqtSignal(int)

    def __init__(self, mcs: SmarActMCS, channel: int, steps: int, amp: int, freq: int):
        super().__init__()
        self._mcs = mcs
        self._channel = channel
        self._steps = steps
        self._amp = amp
        self._freq = freq

    def run(self):
        ch = self._channel
        self.started.emit(ch)
        try:
            self._mcs.step_move(ch, self._steps, self._amp, self._freq)
        except Exception as e:
            import traceback
            self.error.emit(ch, traceback.format_exc())
        else:
            self.finished.emit(ch)


class SmarActControlWindow(QWidget):
    def __init__(self, dll_path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("SmarAct Control")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(860, 420)

        self._mcs = SmarActMCS(dll_path)
        self._threads: Dict[int, Tuple[QThread, MoveWorker]] = {}
        self._channel_widgets: Dict[int, Dict[str, object]] = {}

        root = QHBoxLayout(self)

        left = QVBoxLayout()
        header = QHBoxLayout()
        self.lbl_conn = QLabel("Disconnected")
        self.btn_open = QPushButton("Open")
        self.btn_close = QPushButton("Close")
        self.btn_clearlog = QPushButton("Clear Log")
        header.addWidget(self.lbl_conn)
        header.addStretch(1)
        header.addWidget(self.btn_open)
        header.addWidget(self.btn_close)
        header.addWidget(self.btn_clearlog)
        left.addLayout(header)

        self.grid = QGridLayout()
        left.addLayout(self.grid)

        footer = QHBoxLayout()
        self.lbl_hint = QLabel("Set Steps/Amp/Freq, puis ◀ / ▶ pour ±Steps.")
        footer.addWidget(self.lbl_hint)
        left.addLayout(footer)

        root.addLayout(left, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 3)

        self.btn_open.clicked.connect(self._on_open)
        self.btn_close.clicked.connect(self._on_close)
        self.btn_clearlog.clicked.connect(self._on_clear_log)

        self._build_channels_ui(3, enabled=False)

    def _ts(self) -> str:
        return QDateTime.currentDateTime().toString("HH:mm:ss.zzz")

    def _append_log(self, text: str):
        self.log.append(f"[{self._ts()}] {text}")

    def _build_channels_ui(self, max_channels: int, enabled: bool):
        for i in range(max_channels):
            gb = QGroupBox(f"Channel {i}")
            lay = QGridLayout(gb)

            lay.addWidget(QLabel("Steps"), 0, 0)
            sb_steps = QSpinBox()
            sb_steps.setRange(-1_000_000, 1_000_000)
            sb_steps.setValue(1000)
            sb_steps.setSingleStep(100)
            lay.addWidget(sb_steps, 0, 1)

            lay.addWidget(QLabel("Amplitude"), 1, 0)
            sb_amp = QSpinBox()
            sb_amp.setRange(1, 4095)
            sb_amp.setValue(4095)
            lay.addWidget(sb_amp, 1, 1)

            lay.addWidget(QLabel("Frequency [Hz]"), 2, 0)
            sb_freq = QSpinBox()
            sb_freq.setRange(1, 10000)
            sb_freq.setValue(2000)
            lay.addWidget(sb_freq, 2, 1)

            btn_left = QToolButton()
            btn_left.setArrowType(Qt.LeftArrow)
            btn_right = QToolButton()
            btn_right.setArrowType(Qt.RightArrow)
            lay.addWidget(btn_left, 0, 2)
            lay.addWidget(btn_right, 0, 3)

            self.grid.addWidget(gb, i // 1, i % 1)

            widgets = {
                "group": gb,
                "steps": sb_steps,
                "amp": sb_amp,
                "freq": sb_freq,
                "left": btn_left,
                "right": btn_right,
            }
            self._channel_widgets[i] = widgets

            btn_left.clicked.connect(lambda _=False, ch=i: self._start_move(ch, negative=True))
            btn_right.clicked.connect(lambda _=False, ch=i: self._start_move(ch, negative=False))

            gb.setEnabled(enabled)

    def _on_open(self):
        try:
            self._mcs.open()
            n = min(3, self._mcs.get_num_channels())
        except Exception as e:
            QMessageBox.critical(self, "SmarAct", f"Open failed:\n{e}")
            self.lbl_conn.setText("Disconnected")
            for i in range(3):
                self._channel_widgets[i]["group"].setEnabled(False)
            return

        self.lbl_conn.setText(f"Connected • {n} channel(s)")
        self._append_log("Connected")
        for i in range(3):
            self._channel_widgets[i]["group"].setEnabled(i < n)

    def _on_close(self):
        if any(ch in self._threads for ch in list(self._threads.keys())):
            QMessageBox.warning(self, "SmarAct", "Wait for moves to finish before closing.")
            return
        try:
            self._mcs.close()
        except Exception as e:
            QMessageBox.warning(self, "SmarAct", f"Close error:\n{e}")
        self.lbl_conn.setText("Disconnected")
        self._append_log("Disconnected")
        for i in range(3):
            self._channel_widgets[i]["group"].setEnabled(False)

    def _on_clear_log(self):
        self.log.clear()

    def _lock_channel(self, ch: int, locked: bool):
        w = self._channel_widgets[ch]
        w["steps"].setEnabled(not locked)
        w["amp"].setEnabled(not locked)
        w["freq"].setEnabled(not locked)
        w["left"].setEnabled(not locked)
        w["right"].setEnabled(not locked)

    def _start_move(self, ch: int, negative: bool):
        if ch in self._threads:
            return
        w = self._channel_widgets[ch]
        if not w["group"].isEnabled():
            return
        steps = int(w["steps"].value())
        amp = int(w["amp"].value())
        freq = int(w["freq"].value())
        if steps == 0:
            return
        steps = -abs(steps) if negative else abs(steps)
        self._append_log(f"Ch {ch}: start move steps={steps}, amp={amp}, freq={freq}")

        worker = MoveWorker(self._mcs, ch, steps, amp, freq)
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.started.connect(lambda _ch=ch: self._on_started(_ch))
        worker.finished.connect(lambda _ch=ch: self._on_finished(_ch, steps, amp, freq))
        worker.error.connect(lambda _ch=ch, tb="": self._on_error(_ch, tb))

        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda _w=worker: _w.deleteLater())

        self._threads[ch] = (thread, worker)
        thread.start()

    def _on_started(self, ch: int):
        self._lock_channel(ch, True)

    def _on_finished(self, ch: int, steps: int, amp: int, freq: int):
        self._append_log(f"Ch {ch}: finished steps={steps}, amp={amp}, freq={freq}")
        self._lock_channel(ch, False)
        thr, _ = self._threads.pop(ch, (None, None))
        if thr is not None:
            thr.quit()
            thr.wait()

    def _on_error(self, ch: int, tb: str):
        self._append_log(f"Ch {ch}: ERROR\n{tb}")
        self._lock_channel(ch, False)
        QMessageBox.critical(self, f"SmarAct Ch {ch}", tb)
        thr, _ = self._threads.pop(ch, (None, None))
        if thr is not None:
            thr.quit()
            thr.wait()

    def closeEvent(self, e):
        if self._threads:
            QMessageBox.warning(self, "SmarAct", "A move is still running.")
            e.ignore()
            return
        try:
            self._mcs.close()
        except Exception:
            pass
        super().closeEvent(e)


def main(dll_path: str):
    app = QApplication(sys.argv)
    w = SmarActControlWindow(dll_path)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(
        r"C:\Users\atto\Documents\DLabController\src\dlab\hardware\drivers\smaract_driver\MCSControl.dll"
    ))
