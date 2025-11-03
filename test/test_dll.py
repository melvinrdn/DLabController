import os, ctypes, sys

DLLS = [
    "MCSControl.dll",     # SmarAct
    "atmcd64d.dll",       # Andor SDK2
    "avaspec.dll",        # Avantes
    "GxIAPI.dll",         # Daheng Galaxy
]

print("Python:", sys.executable)
for dll in DLLS:
    try:
        ctypes.WinDLL(dll)
        print(f"[OK] {dll} loaded")
    except OSError as e:
        print(f"[ERR] {dll}: {e}")
