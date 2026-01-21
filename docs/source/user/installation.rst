Installation
============

Requirements
------------
- Python >= 3.10
- Windows

Installation
------------
From the repository root:

.. code-block:: powershell

   .\scripts\setup.ps1

This script:
- creates a local virtual environment (``.venv``)
- installs the package in editable mode
- installs all required dependencies

Hardware dependencies
---------------------
Some features require vendor-specific drivers and SDKs
(Thorlabs, Zaber, Andor, etc.).
