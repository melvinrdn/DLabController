import datetime

from PyQt5.QtWidgets import QTextEdit


class LogPanel(QTextEdit):
    """A read-only text panel for timestamped log messages."""

    def __init__(self, time_fmt: str = "%H:%M:%S", max_lines: int = 500):
        super().__init__()
        self.setReadOnly(True)
        self._time_fmt = time_fmt
        self._max_lines = max_lines

    def log(self, message: str):
        """Append a timestamped message."""
        timestamp = datetime.datetime.now().strftime(self._time_fmt)
        self.append(f"[{timestamp}] {message}")
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Remove old lines if exceeding max_lines."""
        if self.document().blockCount() > self._max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 50)
            cursor.removeSelectedText()