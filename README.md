# DLab Controller

A PyQt5 desktop launcher to drive lab hardware from a single GUI.

## Run

At the start of VS Code, from the repo root, in the terminal:
```powershell
# This activate the virtual env and launch the app
.\scripts\run.ps1
```

After the virtual environnement has been activated, the script can be launch using
```powershell
dlab
```

This opens the **DLab Controller** window with buttons to launch:
- SLM Window
- Andor Live
- Daheng Live (Nomarski / Nozzle / Focus or any index you pick)
- Avaspec Live
- Stage Control & Auto Waveplate Calibration
- ...

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

## Requirements

- **Drivers/SDKs**:
  - *Andor* ATMCD SDK (DLLs available on PATH)
  - *Daheng* Galaxy / gxipy
  - *Avantes* Avaspec runtime
  - *Thorlabs* APT (for motor stages)
  - *Thorlabs* PM100 (VISA runtime/NI minimal for powermeter)


## Data & Logs

- Saved data (images/spectra): `C:\data\<YYYY-MM-DD>\...`

## Troubleshooting

- **DLL/driver not found**: ensure vendor SDKs are installed and DLLs are on PATH.
- **Device not detected**: check USB permissions/driver, try vendor tool (Kinesis for Thorlabs), unplug replug the devices and relaunch the application.
- **ImportError**: re-run `pip install -e .` inside the activated venv.
- **Permission issues writing data**: create `C:\data` manually or change `paths.data_root` in `config.yaml`.

## Author

This code was mainly written by **Melvin Redon**.  
For questions, please contact: **melvin.redon@outlook.com**.
