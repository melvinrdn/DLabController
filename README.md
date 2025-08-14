# DLab Controller

A PyQt5 desktop launcher to drive lab hardware (SLM, Andor & Daheng cameras, Avaspec spectrometer, Thorlabs stages, powermeter) from a single GUI.

## Features

- **SLM Window** (pattern generation: flat, lens, Zernike, grating, vortex, tilt, axicon, etc.)
- **Andor Live** (capture & save frames, metadata)
- **Daheng Live** (multiple camera indices, live view, capture & logging)
- **Avaspec Live** (live spectrum, optional Gaussian fit & pulse estimate, save to .txt)
- **Thorlabs Stage Control** (multiple stages, “power mode” using waveplate calibration)
- **Auto Waveplate Calibration** (sweeps angle vs. power, writes TSV files)
- **Pressure monitor** (Pfeiffer MaxiGauge, optional)
- **Central launcher** (`dlab`) to open each window

---

## Requirements

- **OS**: Windows (recommended; tested setup)
- **Python**: 3.10+
- **Drivers/SDKs** (install only what you use):
  - *Andor* ATMCD SDK (DLLs available on PATH)
  - *Daheng* Galaxy / gxipy
  - *Avantes* Avaspec runtime
  - *Thorlabs* APT (for motor stages)
  - *Thorlabs* PM100 (VISA runtime/NI minimal for powermeter)
- **Python packages** (declared in `pyproject.toml`):
  - Core: `PyQt5`, `PyYAML`, `numpy`, `matplotlib`, `prometheus_client`, `pyserial`
  - SLM & optics: `prysm`
  - Hardware helpers: `pylablib`, `ThorlabsPM100`, `thorlabs_apt`

---

## Install

From the repo root:

```powershell
# one-shot setup (creates venv, installs package)
.\scripts\setup.ps1
```

This will:
- Create/activate `.venv`
- Upgrade `pip/setuptools/wheel`
- Install the project in **editable** mode (`pip install -e .`)

---

## Run

After installation/activation of the venv:

```powershell
dlab
```

This opens the **DLab Controller** window with buttons to launch:
- SLM Window
- Andor Live
- Daheng Live (Nomarski / Nozzle / Focus or any index you pick)
- Avaspec Live
- Stage Control & Auto Waveplate Calibration
- Pressure monitor

---

## Configuration

- Main config file: **`config/config.yaml`**
- Resources: **`ressources/`**
  - **SLM defaults**: `ressources/saved_settings/slm_defaults.yaml`  
    Profiles & backgrounds under:
    - `ressources/saved_settings/SLM_red/`
    - `ressources/saved_settings/SLM_green/`
  - **Waveplate calibration mapping**: `ressources/calibration/wp_default.yaml`  
    (maps WP1… to file paths; used by the WaveplateCalib widget)
  - **Calibration data** (produced by AutoWaveplateCalib):  
    `ressources/calibration/<wp_name>/calib_YYYY_MM_DD.txt`

### Waveplate calibration file format

Tab-separated file with a header and two columns:
```
# Date: 2025-08-13T17:20:10
# MotorID: 83837725
# PowermeterID: USB0::0x1313::0x8078::P0045634::INSTR
# Wavelength_nm: 1030.000
# Stabilization_s: 1.000
Angle_deg\tPower_W
0.000000\t1.234000e-03
5.000000\t1.112000e-03
...
```
These files are read by the `WaveplateCalib` widget (and can update Stage Control “power mode”).

---

## Data & Logs

- Saved data (images/spectra): `C:\data\<YYYY-MM-DD>\...`
- App log: `logs/dlab.log`

---

## Project Layout (short)

```
config/
  config.yaml
ressources/
  aberration_correction/
  calibration/
    wp_default.yaml
    wp1/default_wp1.txt
    ...
  saved_settings/
    slm_defaults.yaml
    SLM_red/...
    SLM_green/...
scripts/
  setup.ps1
src/dlab/
  app.py (entry -> dlab)
  boot.py
  diagnostics/ui/
    slm_window.py
    andor_live_window.py
    daheng_live_window.py
    avaspec_live_window.py
    stage_control_window.py
  hardware/drivers/...
  hardware/wrappers/...
logs/
```

---

## Troubleshooting

- **DLL/driver not found**: ensure vendor SDKs are installed and DLLs are on PATH.
- **Device not detected**: check USB permissions/driver, try vendor tool (Kinesis for Thorlabs), unplug replug the devices and relaunch the application.
- **ImportError**: re-run `pip install -e .` inside the activated venv.
- **Permission issues writing data**: create `C:\data` manually or change `paths.data_root` in `config.yaml`.

---

## License

Research / internal lab use.

---

## Author

This code was mainly written by **Melvin Redon**.  
For questions, please contact: **melvin.redon@outlook.com**.
