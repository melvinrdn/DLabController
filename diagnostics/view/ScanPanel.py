# scan_panel.py
import os
import time
import datetime
import numpy as np
from PIL import Image, PngImagePlugin
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QCheckBox, QLineEdit, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from diagnostics.view.StageControl import power_to_angle

class ScanWorker(QThread):
    progress = pyqtSignal(str)          # For updating the scan panel log.
    image_update = pyqtSignal(str, object)  # (camera, image) for live plot updates.
    finished = pyqtSignal()             # Emitted when scan is complete.

    def __init__(self, andor_live, daheng_live, stage_control, scan_params,
                 use_andor, use_daheng, save_scan, comment, delay, background,
                 wp_group, power_mode, parent=None):
        super().__init__(parent)
        self.andor_live = andor_live
        self.daheng_live = daheng_live
        self.stage_control = stage_control  # Reference to the ThorlabsView
        self.scan_params = scan_params      # Dict: {'start': ..., 'stop': ..., 'points': ...}
        self.use_andor = use_andor
        self.use_daheng = use_daheng
        self.save_scan = save_scan
        self.comment = comment
        self.delay = delay
        self.background = background
        self.wp_group = wp_group            # String: "w", "2w" or "3w"
        self.power_mode = power_mode        # Boolean: if True, use calibration conversion.
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        # Create general scan folder and log file.
        scan_dir = os.path.join("C:/data", date_str, "Scans")
        if not os.path.exists(scan_dir):
            os.makedirs(scan_dir)
        log_file_name = f"Scan_log_{now.strftime('%Y_%m_%d')}.txt"
        general_log_path = os.path.join(scan_dir, log_file_name)

        # Write header lines.
        try:
            with open(general_log_path, "a") as f:
                f.write(f"# {self.comment}\n")
                scan_tab = "Intensity scan"
                cameras_used = []
                if self.use_andor:
                    cameras_used.append("Andor")
                if self.use_daheng:
                    cameras_used.append("Daheng")
                cameras_str = ", ".join(cameras_used)

                mode_str = "Power" if self.power_mode else "Angle"
                # Explicitly write the mode (Power/Angle) in the log header
                f.write(
                    f"# Scan Type: {scan_tab} ({self.wp_group}), Mode: {mode_str}, {cameras_str}; "
                    f"Range: {self.scan_params['start']} to {self.scan_params['stop']}; Points: {self.scan_params['points']}\n"
                )
            self.progress.emit("Scan header written.")
        except Exception as e:
            self.progress.emit(f"Error writing scan header: {e}")

        # Generate scan points (stage positions or power values).
        scan_values = np.linspace(self.scan_params['start'], self.scan_params['stop'], self.scan_params['points'])

        for i, point in enumerate(scan_values, start=1):
            if self._abort:
                self.progress.emit("Scan aborted by user.")
                with open(general_log_path, "a") as f:
                    f.write("# Scan aborted by user.\n")
                break

            self.progress.emit(f"Scan point {i}: value = {point}")
            # Move stage before capturing images.
            if self.stage_control is not None:
                try:
                    if self.wp_group == "w":
                        stage_index = 1
                    elif self.wp_group == "2w":
                        stage_index = 2
                    elif self.wp_group == "3w":
                        stage_index = 3
                    else:
                        stage_index = None

                    if stage_index is not None and len(self.stage_control.stage_rows) > stage_index:
                        stage_row = self.stage_control.stage_rows[stage_index]

                        if stage_row.controller is not None:
                            # Synchronize power mode with stage control GUI
                            power_mode_enabled = stage_row.power_mode_checkbox.isChecked()

                            # Make sure the Scan GUI reflects the power mode
                            if hasattr(self, 'scan_power_mode_checkbox'):
                                self.scan_power_mode_checkbox.setChecked(power_mode_enabled)

                            # Determine target angle based on power mode
                            if power_mode_enabled:
                                # Cap the scan point to the maximum calibrated amplitude
                                max_power = stage_row.amplitude
                                if point > max_power:
                                    self.progress.emit(
                                        f"Scan point {point} exceeds max calibration ({max_power}), capping.")
                                    point = max_power

                                target_angle = power_to_angle(point, stage_row.amplitude, stage_row.offset)
                                self.progress.emit(f"Converting power {point} to angle {target_angle:.2f}.")
                            else:
                                target_angle = point % 360  # Direct angle usage

                            # Update stage GUI "target" field before moving
                            stage_row.target_edit.setText(
                                f"{point:.2f}" if power_mode_enabled else f"{target_angle:.2f}")

                            # Move the stage
                            self.progress.emit(f"Moving stage ({self.wp_group}) to position {target_angle:.2f}°...")
                            stage_row.controller.move_to(target_angle, blocking=True)

                            # After move, update the current position displayed
                            new_pos = stage_row.controller.get_position()
                            stage_row.current_edit.setText(f"{new_pos:.2f}" if new_pos is not None else "N/A")
                            self.progress.emit(f"Stage ({self.wp_group}) moved to position {new_pos:.2f}°.")
                        else:
                            self.progress.emit(f"Stage ({self.wp_group}) is not activated.")
                    else:
                        self.progress.emit("Invalid stage group selection.")
                except Exception as e:
                    self.progress.emit(f"Error moving stage: {e}")
            else:
                self.progress.emit("No stage control available; skipping stage move.")

            # Process Andor camera.
            if self.use_andor:
                if self.andor_live is None or self.andor_live.camera_controller is None:
                    self.progress.emit("AndorLive is not activated. Skipping Andor at this point.")
                else:
                    try:
                        exposure_a = int(self.andor_live.exposure_edit.text())
                        avgs_a = int(self.andor_live.avgs_edit.text())
                        andor_img = self.andor_live.camera_controller.take_image(exposure_a, avgs_a)
                        self.progress.emit("Andor image captured.")
                        self.image_update.emit("Andor", andor_img)
                        if self.save_scan:
                            file_name = self.save_andor_scan_image(andor_img, exposure_a, avgs_a)
                            if file_name:
                                with open(general_log_path, "a") as f:
                                    f.write(f"Andor\t{point}\t{file_name}\n")
                                self.progress.emit(f"Andor scan event logged: point {point}, file {file_name}")
                    except Exception as e:
                        self.progress.emit(f"Error capturing Andor image: {e}")
            # Delay between cameras if both are used.
            if self.use_andor and self.use_daheng:
                self.progress.emit(f"Waiting {self.delay} seconds between camera acquisitions...")
                time.sleep(self.delay)
            # Process Daheng camera.
            if self.use_daheng:
                if self.daheng_live is None or self.daheng_live.camera_controller is None:
                    self.progress.emit("DahengLive is not activated. Skipping Daheng at this point.")
                else:
                    try:
                        exposure_d = int(self.daheng_live.exposure_edit.text())
                        gain = int(self.daheng_live.gain_edit.text())
                        avgs_d = int(self.daheng_live.avgs_edit.text())
                        daheng_img = self.daheng_live.camera_controller.take_image(exposure_d, gain, avgs_d)
                        self.progress.emit("Daheng image captured.")
                        self.image_update.emit("Daheng", daheng_img)
                        if self.save_scan:
                            file_name = self.save_daheng_scan_image(daheng_img, exposure_d, gain, avgs_d)
                            if file_name:
                                with open(general_log_path, "a") as f:
                                    f.write(f"Daheng\t{point}\t{file_name}\n")
                                self.progress.emit(f"Daheng scan event logged: point {point}, file {file_name}")
                    except Exception as e:
                        self.progress.emit(f"Error capturing Daheng image: {e}")
            # (Optional: add a delay between scan points.)
        self.finished.emit()

    def save_andor_scan_image(self, image, exposure, avgs):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        dir_path = os.path.join("C:/data", date_str, "AndorCamera")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        timestamp = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"AndorCamera_MCP_{'Background' if self.background else 'Image'}_{timestamp}.png"
        file_path = os.path.join(dir_path, file_name)
        try:
            # Retrieve MCP voltage from AndorLive.
            mcp_voltage = ""
            if self.andor_live is not None and hasattr(self.andor_live, "mcp_voltage_edit"):
                mcp_voltage = self.andor_live.mcp_voltage_edit.text()
            frame_uint8 = np.uint8(np.clip(image, 0, 255))
            img = Image.fromarray(frame_uint8)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", str(exposure))
            metadata.add_text("Averages", str(avgs))
            metadata.add_text("MCP Voltage", str(mcp_voltage))
            metadata.add_text("Comment", self.comment)
            img.save(file_path, pnginfo=metadata)
            # Also log in Andor-specific log.
            log_file_name = f"AndorCamera_log_{now.strftime('%Y_%m_%d')}.txt"
            andor_log_path = os.path.join(dir_path, log_file_name)
            header = "File Name\tExposure (µs)\tAverages\tMCP Voltage\tComment\n"
            if not os.path.exists(andor_log_path):
                with open(andor_log_path, "w") as f:
                    f.write(header)
            with open(andor_log_path, "a") as f:
                f.write(f"{file_name}\t{exposure}\t{avgs}\t{mcp_voltage}\t{self.comment}\n")
            return file_name
        except Exception as e:
            self.progress.emit(f"Error saving Andor scan image: {e}")
            return None

    def save_daheng_scan_image(self, image, exposure, gain, avgs):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        dir_path = os.path.join("C:/data", date_str, "DahengCamera")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        timestamp = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"DahengCamera_Nozzle_{'Background' if self.background else 'Image'}_{timestamp}.png"
        file_path = os.path.join(dir_path, file_name)
        try:
            frame_uint8 = np.uint8(np.clip(image, 0, 255))
            img = Image.fromarray(frame_uint8)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", str(exposure))
            metadata.add_text("Gain", str(gain))
            metadata.add_text("Comment", self.comment)
            img.save(file_path, pnginfo=metadata)
            # Also log in Daheng-specific log.
            log_file_name = f"DahengCamera_log_{now.strftime('%Y_%m_%d')}.txt"
            daheng_log_path = os.path.join(dir_path, log_file_name)
            header = "File Name\tExposure (µs)\tGain\tAverages\tComment\n"
            if not os.path.exists(daheng_log_path):
                with open(daheng_log_path, "w") as f:
                    f.write(header)
            with open(daheng_log_path, "a") as f:
                f.write(f"{file_name}\t{exposure}\t{gain}\t{avgs}\t{self.comment}\n")
            return file_name
        except Exception as e:
            self.progress.emit(f"Error saving Daheng scan image: {e}")
            return None


class ScanPanel(QMainWindow):
    def __init__(self, andor_live, daheng_live, stage_control):
        """
        :param andor_live: Reference to AndorLive GUI.
        :param daheng_live: Reference to DahengLive GUI.
        :param stage_control: Reference to the ThorlabsView (stage control).
        """
        super().__init__()
        self.setWindowTitle("Scan Panel")
        self.resize(600, 650)
        self.andor_live = andor_live
        self.daheng_live = daheng_live
        self.stage_control = stage_control
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- Camera Selection ---
        cam_layout = QHBoxLayout()
        self.andor_checkbox = QCheckBox("Use Andor")
        self.daheng_checkbox = QCheckBox("Use Daheng")
        cam_layout.addWidget(self.andor_checkbox)
        cam_layout.addWidget(self.daheng_checkbox)
        layout.addLayout(cam_layout)

        # --- Stage (WP Group) Selection ---
        wp_layout = QHBoxLayout()
        wp_label = QLabel("Waveplate Group:")
        self.wp_combo = QComboBox()
        self.wp_combo.addItems(["w", "2w", "3w"])
        wp_layout.addWidget(wp_label)
        wp_layout.addWidget(self.wp_combo)
        layout.addLayout(wp_layout)

        # --- Scan Parameters: Start, Stop, Points, and Delay ---
        scan_param_layout = QHBoxLayout()
        scan_start_label = QLabel("Scan Start:")
        self.scan_start_edit = QLineEdit("0")
        scan_param_layout.addWidget(scan_start_label)
        scan_param_layout.addWidget(self.scan_start_edit)
        scan_stop_label = QLabel("Scan Stop:")
        self.scan_stop_edit = QLineEdit("10")
        scan_param_layout.addWidget(scan_stop_label)
        scan_param_layout.addWidget(self.scan_stop_edit)
        scan_points_label = QLabel("Scan Points:")
        self.scan_points_edit = QLineEdit("5")
        scan_param_layout.addWidget(scan_points_label)
        scan_param_layout.addWidget(self.scan_points_edit)
        delay_label = QLabel("Delay (s):")
        self.delay_edit = QLineEdit("1")
        scan_param_layout.addWidget(delay_label)
        scan_param_layout.addWidget(self.delay_edit)
        layout.addLayout(scan_param_layout)

        # --- Save Scan Checkbox ---
        self.save_checkbox = QCheckBox("Save Scan")
        layout.addWidget(self.save_checkbox)

        # --- Background Checkbox ---
        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        layout.addWidget(self.background_checkbox)

        # --- Comment Entry ---
        comment_layout = QHBoxLayout()
        comment_label = QLabel("Comment:")
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(comment_label)
        comment_layout.addWidget(self.comment_edit)
        layout.addLayout(comment_layout)

        # --- Run and Abort Buttons ---
        button_layout = QHBoxLayout()
        self.scan_button = QPushButton("Run Scan")
        self.scan_button.clicked.connect(self.start_scan)
        button_layout.addWidget(self.scan_button)
        self.abort_button = QPushButton("Abort Scan")
        self.abort_button.clicked.connect(self.abort_scan)
        self.abort_button.setEnabled(False)
        button_layout.addWidget(self.abort_button)
        layout.addLayout(button_layout)

        # --- Local Log Text Box ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")

    def start_scan(self):
        try:
            scan_start = float(self.scan_start_edit.text())
            scan_stop = float(self.scan_stop_edit.text())
            scan_points = int(self.scan_points_edit.text())
            delay = float(self.delay_edit.text())
        except ValueError:
            self.log("Invalid scan parameters.")
            return
        use_andor = self.andor_checkbox.isChecked()
        use_daheng = self.daheng_checkbox.isChecked()
        if not use_andor and not use_daheng:
            self.log("No cameras selected for scan.")
            return
        save_scan = self.save_checkbox.isChecked()
        comment = self.comment_edit.text()
        background = self.background_checkbox.isChecked()
        scan_params = {"start": scan_start, "stop": scan_stop, "points": scan_points}
        wp_group = self.wp_combo.currentText()

        wp_group = self.wp_combo.currentText()
        stage_index = {"w": 1, "2w": 2, "3w": 3}.get(wp_group)
        power_mode = False

        if stage_index is not None and len(self.stage_control.stage_rows) > stage_index:
            stage_row = self.stage_control.stage_rows[stage_index]
            power_mode = stage_row.power_mode_checkbox.isChecked()
        else:
            self.log("Invalid waveplate group or stage not available.")
            return

        # Deactivate live capture if running.
        if self.andor_live is not None and getattr(self.andor_live, "capture_thread", None):
            self._andor_live_was_running = True
            self.andor_live.stop_capture()
            self.log("Andor live capture stopped for scan.")
        else:
            self._andor_live_was_running = False
        if self.daheng_live is not None and getattr(self.daheng_live, "capture_thread", None):
            self._daheng_live_was_running = True
            self.daheng_live.stop_capture()
            self.log("Daheng live capture stopped for scan.")
        else:
            self._daheng_live_was_running = False

        self.scan_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self.worker = ScanWorker(self.andor_live, self.daheng_live, self.stage_control, scan_params,
                                 use_andor, use_daheng, save_scan, comment, delay, background, wp_group, power_mode)
        self.worker.progress.connect(self.log)
        self.worker.image_update.connect(self.update_live_plots)
        self.worker.finished.connect(self.scan_finished)
        self.worker.start()

    def update_live_plots(self, camera, image):
        if camera == "Andor" and self.andor_live is not None:
            self.andor_live.update_image(image)
        elif camera == "Daheng" and self.daheng_live is not None:
            self.daheng_live.update_image(image)

    def abort_scan(self):
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.abort()
            self.log("Abort signal sent.")
        else:
            self.log("No scan in progress to abort.")

    def scan_finished(self):
        self.log("Scan finished.")
        self.scan_button.setEnabled(True)
        self.abort_button.setEnabled(False)
        if self.andor_live is not None and self._andor_live_was_running:
            self.andor_live.start_capture()
            self.log("Andor live capture reactivated.")
        if self.daheng_live is not None and self._daheng_live_was_running:
            self.daheng_live.start_capture()
            self.log("Daheng live capture reactivated.")
