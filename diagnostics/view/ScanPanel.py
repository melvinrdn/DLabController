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

    def __init__(self, andor_live, daheng_focus, daheng_nozzle,
                 stage_control, scan_params, use_andor, use_daheng_focus,
                 use_daheng_nozzle, save_scan, comment,
                 delay, background, wp_group, power_mode, parent=None):
        super().__init__(parent)
        self.andor_live = andor_live
        self.daheng_focus = daheng_focus
        self.daheng_nozzle = daheng_nozzle
        self.stage_control = stage_control  # Reference to the ThorlabsView
        self.scan_params = scan_params      # Dict: {'start': ..., 'stop': ..., 'points': ...}
        self.use_andor = use_andor
        self.use_daheng_focus = use_daheng_focus
        self.use_daheng_nozzle = use_daheng_nozzle
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
        log_file_name = f"Scan_log_{now.strftime('%Y-%m-%d')}.txt"
        general_log_path = os.path.join(scan_dir, log_file_name)

        # Write header lines.
        try:
            with open(general_log_path, "a") as f:
                f.write(f"# {self.comment}\n")
                scan_tab = "Energy scan"
                cameras_used = []
                if self.use_andor:
                    cameras_used.append("Andor")
                if self.use_daheng_focus:
                    cameras_used.append("Daheng Focus")
                if self.use_daheng_nozzle:
                    cameras_used.append("Daheng Nozzle")
                cameras_str = ", ".join(cameras_used)

                mode_str = "Power" if self.power_mode else "Angle"
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
                            power_mode_enabled = stage_row.power_mode_checkbox.isChecked()

                            if hasattr(self, 'scan_power_mode_checkbox'):
                                self.scan_power_mode_checkbox.setChecked(power_mode_enabled)

                            if power_mode_enabled:
                                max_power = stage_row.amplitude
                                if point > max_power:
                                    self.progress.emit(
                                        f"Scan point {point} exceeds max calibration ({max_power}), capping.")
                                    point = max_power

                                target_angle = power_to_angle(point, stage_row.amplitude, stage_row.offset)
                                self.progress.emit(f"Converting power {point} to angle {target_angle:.2f}.")
                            else:
                                target_angle = point % 360

                            stage_row.target_edit.setText(
                                f"{point:.2f}" if power_mode_enabled else f"{target_angle:.2f}")

                            self.progress.emit(f"Moving stage ({self.wp_group}) to position {target_angle:.2f}°...")
                            stage_row.controller.move_to(target_angle, blocking=True)

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
                if self.andor_live is None or self.andor_live.cam is None:
                    self.progress.emit("AndorLive is not activated. Skipping Andor at this point.")
                else:
                    try:
                        exposure_a = int(self.andor_live.exposure_edit.text())
                        avgs_a = int(self.andor_live.avgs_edit.text())
                        andor_img = self.andor_live.cam.take_image(exposure_a, avgs_a)
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
            # Delay between cameras if needed.
            if self.use_andor and (self.use_daheng_focus or self.use_daheng_nozzle):
                self.progress.emit(f"Waiting {self.delay} seconds between camera acquisitions...")
                time.sleep(self.delay)

            # Process Daheng Focus camera.
            if self.use_daheng_focus:
                if self.daheng_focus is None or self.daheng_focus.cam is None:
                    self.progress.emit("Daheng Focus is not activated. Skipping Daheng Focus at this point.")
                else:
                    try:
                        exposure_f = int(self.daheng_focus.exposure_edit.text())
                        gain_f = int(self.daheng_focus.gain_edit.text())
                        avgs_f = int(self.daheng_focus.avgs_edit.text())
                        focus_img = self.daheng_focus.cam.take_image(exposure_f, gain_f, avgs_f)
                        self.progress.emit("Daheng Focus image captured.")
                        self.image_update.emit("Daheng Focus", focus_img)
                        if self.save_scan:
                            file_name = self.save_daheng_scan_image(focus_img, exposure_f, gain_f, avgs_f, camera_type="Focus")
                            if file_name:
                                with open(general_log_path, "a") as f:
                                    f.write(f"DahengFocus\t{point}\t{file_name}\n")
                                self.progress.emit(f"Daheng Focus scan event logged: point {point}, file {file_name}")
                    except Exception as e:
                        self.progress.emit(f"Error capturing Daheng Focus image: {e}")

            # Process Daheng Nozzle camera.
            if self.use_daheng_nozzle:
                if self.daheng_nozzle is None or self.daheng_nozzle.cam is None:
                    self.progress.emit("Daheng Nozzle is not activated. Skipping Daheng Nozzle at this point.")
                else:
                    try:
                        exposure_n = int(self.daheng_nozzle.exposure_edit.text())
                        gain_n = int(self.daheng_nozzle.gain_edit.text())
                        avgs_n = int(self.daheng_nozzle.avgs_edit.text())
                        nozzle_img = self.daheng_nozzle.cam.take_image(exposure_n, gain_n, avgs_n)
                        self.progress.emit("Daheng Nozzle image captured.")
                        self.image_update.emit("Daheng Nozzle", nozzle_img)
                        if self.save_scan:
                            file_name = self.save_daheng_scan_image(nozzle_img, exposure_n, gain_n, avgs_n, camera_type="Nozzle")
                            if file_name:
                                with open(general_log_path, "a") as f:
                                    f.write(f"DahengNozzle\t{point}\t{file_name}\n")
                                self.progress.emit(f"Daheng Nozzle scan event logged: point {point}, file {file_name}")
                    except Exception as e:
                        self.progress.emit(f"Error capturing Daheng Nozzle image: {e}")
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
            mcp_voltage = ""
            if self.andor_live is not None and hasattr(self.andor_live, "mcp_voltage_edit"):
                mcp_voltage = self.andor_live.mcp_voltage_edit.text()           
            frame_uint16 = image.astype(np.uint16)
            img = Image.fromarray(frame_uint16)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", str(exposure))
            metadata.add_text("Averages", str(avgs))
            metadata.add_text("MCP Voltage", str(mcp_voltage))
            metadata.add_text("Comment", self.comment)
            img.save(file_path, format='PNG', pnginfo=metadata)
            
            log_file_name = f"AndorCamera_log_{now.strftime('%Y-%m-%d')}.txt"
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

    def save_daheng_scan_image(self, image, exposure, gain, avgs, camera_type):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        # Create directory based on camera type.
        dir_name = f"DahengCamera_{camera_type}"
        dir_path = os.path.join("C:/data", date_str, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        timestamp = now.strftime("%Y%m%d_%H%M%S%f")
        file_name = f"{dir_name}_{'Background' if self.background else 'Image'}_{timestamp}.png"
        file_path = os.path.join(dir_path, file_name)
        try:
            frame_uint8 = np.uint8(np.clip(image, 0, 255))
            img = Image.fromarray(frame_uint8)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", str(exposure))
            metadata.add_text("Gain", str(gain))
            metadata.add_text("Comment", self.comment)
            img.save(file_path, pnginfo=metadata)
            log_file_name = f"{dir_name}_log_{now.strftime('%Y-%m-%d')}.txt"
            daheng_log_path = os.path.join(dir_path, log_file_name)
            header = "File Name\tExposure (µs)\tGain\tAverages\tComment\n"
            if not os.path.exists(daheng_log_path):
                with open(daheng_log_path, "w") as f:
                    f.write(header)
            with open(daheng_log_path, "a") as f:
                f.write(f"{file_name}\t{exposure}\t{gain}\t{avgs}\t{self.comment}\n")
            return file_name
        except Exception as e:
            self.progress.emit(f"Error saving Daheng {camera_type} scan image: {e}")
            return None


class ScanPanel(QMainWindow):
    def __init__(self, andor_live, daheng_focus, daheng_nozzle, stage_control):
        """
        :param andor_live: Reference to AndorLive GUI.
        :param daheng_focus: Reference to Daheng Focus GUI.
        :param daheng_nozzle: Reference to Daheng Nozzle GUI.
        :param stage_control: Reference to the ThorlabsView (stage control).
        """
        super().__init__()
        self.setWindowTitle("Scan Panel")
        self.resize(600, 650)
        self.andor_live = andor_live
        self.daheng_focus = daheng_focus
        self.daheng_nozzle = daheng_nozzle
        self.stage_control = stage_control
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- Andor Camera Selection ---
        cam_layout = QHBoxLayout()
        self.andor_checkbox = QCheckBox("Use Andor")
        cam_layout.addWidget(self.andor_checkbox)
        layout.addLayout(cam_layout)

        self.andor_checkbox.setEnabled(self.andor_live.cam is not None)
        self.andor_live.activate_button.clicked.connect(
            lambda: self.andor_checkbox.setEnabled(True)
        )

        # --- Daheng Cameras Selection ---
        daheng_cam_layout = QHBoxLayout()
        self.daheng_focus_checkbox = QCheckBox("Use Daheng Focus")
        self.daheng_nozzle_checkbox = QCheckBox("Use Daheng Nozzle")
        daheng_cam_layout.addWidget(self.daheng_focus_checkbox)
        daheng_cam_layout.addWidget(self.daheng_nozzle_checkbox)
        layout.addLayout(daheng_cam_layout)

        self.daheng_focus_checkbox.setEnabled(self.daheng_focus.cam is not None)
        self.daheng_focus.activate_camera_btn.clicked.connect(
            lambda: self.daheng_focus_checkbox.setEnabled(True)
        )

        self.daheng_nozzle_checkbox.setEnabled(self.daheng_nozzle.cam is not None)
        self.daheng_nozzle.activate_camera_btn.clicked.connect(
            lambda: self.daheng_nozzle_checkbox.setEnabled(True)
        )

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
        use_daheng_focus = self.daheng_focus_checkbox.isChecked()
        use_daheng_nozzle = self.daheng_nozzle_checkbox.isChecked()
        if not use_andor and not (use_daheng_focus or use_daheng_nozzle):
            self.log("No cameras selected for scan.")
            return
        save_scan = self.save_checkbox.isChecked()
        comment = self.comment_edit.text()
        background = self.background_checkbox.isChecked()
        scan_params = {"start": scan_start, "stop": scan_stop, "points": scan_points}
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
        if self.daheng_focus is not None and getattr(self.daheng_focus, "capture_thread", None):
            self._daheng_focus_live_was_running = True
            self.daheng_focus.stop_capture()
            self.log("Daheng Focus live capture stopped for scan.")
        else:
            self._daheng_focus_live_was_running = False
        if self.daheng_nozzle is not None and getattr(self.daheng_nozzle, "capture_thread", None):
            self._daheng_nozzle_live_was_running = True
            self.daheng_nozzle.stop_capture()
            self.log("Daheng Nozzle live capture stopped for scan.")
        else:
            self._daheng_nozzle_live_was_running = False

        self.scan_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self.worker = ScanWorker(
            self.andor_live,
            self.daheng_focus,
            self.daheng_nozzle,
            self.stage_control,
            scan_params,
            use_andor,
            use_daheng_focus,
            use_daheng_nozzle,
            save_scan,
            comment,
            delay,
            background,
            wp_group,
            power_mode
        )
        self.worker.progress.connect(self.log)
        self.worker.image_update.connect(self.update_live_plots)
        self.worker.finished.connect(self.scan_finished)
        self.worker.start()

    def update_live_plots(self, camera, image):
        if camera == "Andor" and self.andor_live is not None:
            self.andor_live.update_image(image)
        elif camera == "Daheng Focus" and self.daheng_focus is not None:
            self.daheng_focus.update_image(image)
        elif camera == "Daheng Nozzle" and self.daheng_nozzle is not None:
            self.daheng_nozzle.update_image(image)

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
        if self.daheng_focus is not None and self._daheng_focus_live_was_running:
            self.daheng_focus.start_capture()
            self.log("Daheng Focus live capture reactivated.")
        if self.daheng_nozzle is not None and self._daheng_nozzle_live_was_running:
            self.daheng_nozzle.start_capture()
            self.log("Daheng Nozzle live capture reactivated.")
