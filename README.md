# DLab Controller

DLab Controller is a PyQt5-based application used to control experimental hardware in the DLab at Lund University.

The project provides a modular graphical interface and a Python backend to interact with laboratory devices (stages, power meters, cameras, etc.).

---

## Installation

From the repository root:

```powershell
# One-shot setup (creates venv and installs the package)
.\scripts\setup.ps1
```

---

## Running the application

### Development

```powershell
.\scripts\run.ps1
```

### Manual launch

Once the virtual environment is activated:

```powershell
dlab
```

or:

```powershell
python -m dlab.app
```

---

## Documentation

The documentation (user guide, developer guide, and API reference) is available online https://melvinrdn.github.io/dlabcontroller/

To build the documentation locally:

```powershell
cd docs
.\make.bat html
```

---

## Project layout

```
src/dlab/
├── ui/          # GUI components
├── hardware/    # Hardware abstraction and wrappers
├── core/        # Shared infrastructure (device manager)
├── utils/       # Utilities
└── app.py
```

---

## Author

This software was written by **Melvin Redon**.
