Architecture
============

Overview
--------
The project follows a ``src/`` layout and is organized into clearly
separated functional layers.

Package structure
-----------------
``src/dlab`` contains the following main subpackages:

- ``ui``  
  Graphical user interface components (windows, tabs, dialogs).

- ``ui.scans``  
  Scan orchestration logic and scan-specific UI elements.

- ``hardware``  
  Hardware abstraction layer.

- ``hardware.wrappers``  
  High-level Python wrappers around vendor SDKs and drivers.

- ``core``  
  Shared infrastructure (device registry).

- ``utils``  
  Generic utility functions.

Execution flow
--------------
1. A user action is triggered in the GUI.
2. The UI layer delegates the action to a scan or controller module.
3. The scan logic calls the appropriate hardware wrapper.
4. The wrapper communicates with the underlying device driver or SDK.
5. Results are returned to the UI for display or storage.

