# AutoAttCalib - Automatic Attenuator Calibration

This project provides an automatic way to calibrate an optical attenuator. The system uses a motorized stage to position a waveplate at specified angles and captures power measurements with a connected Thorlabs power meter. Real-time plotting of power data provides immediate feedback during the calibration process.

## Project Structure
- `auto_att_calib.py`: Contains the `AutoAttCalib` class, which handles motor and power meter initialization, data collection, and real-time plotting.
- `run_calibration.py`: Script to execute the calibration process. This script imports the `AutoAttCalib` class, sets up necessary parameters, and runs the calibration.

## Features
- **Automated Calibration**: Moves the motorized stage to specified angles and records power levels.
- **Real-Time Monitoring**: Displays live power measurement data.
- **Configurable Settings**: Customize motor and power meter parameters, measurement angles, and save paths.

## Usage
1. **Settings Configuration**:
   - Define your motor and power meter settings in `run_calibration.py`.
     - `motor_id`: Thorlabs motor ID for the waveplate rotation stage.
     - `powermeter_id`: VISA resource ID for the Thorlabs power meter (USB address).
2. **Waveplate Configuration**:
   - Specify the waveplate identifier and path to save calibration data.
     - `waveplate_name`: Identifier for the waveplate being calibrated.
     - `waveplate_path`: Directory path where calibration data will be saved.
3. **Measurement Angles**:
   - Define an array, `angles_to_measure`, with angles (in degrees) at which power measurements will be taken.
4. **Run the Calibration**:
   - Execute `run_calibration.py` to start the calibration process. This will initialize hardware and begin data collection with real-time plotting.

## Dependencies
Ensure the following libraries and drivers are installed:

- **Python Libraries**:
  - `pyvisa`: For VISA instrument communication (e.g., power meters).
  - `thorlabs_apt`: For Thorlabs motorized stage control.
  - `ThorlabsPM100`: For interfacing with Thorlabs power meters.
  - `numpy`: For numerical operations.
  - `matplotlib`: For live plotting.

- **Required Drivers**:
  - **Thorlabs APT**: Necessary for `thorlabs_apt` to communicate with Thorlabs motors.
    - **Note**: After installing, move the `APT.dll` file into the `thorlabs_apt` directory.
  - **Thorlabs Power Meter Driver**: Install this driver to enable communication with Thorlabs power meters (installation may take some time).
    - Both drivers are available on the Thorlabs software page.

---
*Written by Melvin Redon on 2024-11-08*