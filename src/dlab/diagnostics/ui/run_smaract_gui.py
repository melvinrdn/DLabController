# run_smaract_gui.py
import os
import sys
import ctypes

DLL_PATH = os.path.abspath(
    r"C:\Users\atto\Documents\DLabController\src\dlab\hardware\drivers\smaract_driver\MCSControl.dll"
)
DLL_DIR = os.path.dirname(DLL_PATH)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DLL_DIR)
else:
    os.environ["PATH"] = DLL_DIR + os.pathsep + os.environ.get("PATH", "")


LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
hmod = kernel32.LoadLibraryExW(DLL_PATH, None, LOAD_WITH_ALTERED_SEARCH_PATH)
if not hmod:
    err = ctypes.get_last_error()
    raise OSError(f"LoadLibraryExW failed for '{DLL_PATH}' (err={err}). "
                  "Vérifie que la DLL est x64 et que ses dépendances VC++ sont installées.")

from dlab.diagnostics.ui.smaract_control_window import main as smaract_main  

if __name__ == "__main__":
    sys.exit(smaract_main(dll_path=DLL_PATH))
