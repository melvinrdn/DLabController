# D-Lab Controller
A GUI application for controlling the hadrware of the D-Lab.

To run the program, execute "DLabController.py", it will load all the other files as it needs to.

## Structure

"model" contains the scripts used for data calculation.

"view" contains the scripts used for data and GUI displaying.

"drivers" contains all the required drivers to communicate with the hardware : 

- "andor_driver" contains drivers to communicate with the XUV camera.

- "avaspec_driver" contains drivers to communicate with the Aventes Spectrometer.

- "gxipy_driver" contains drivers to communicate with the Daheng camera.

- "santec_driver" contains drivers to communicate with the Santec-SLMs.

- "thorlabs_apt_driver" contains drivers to communicate with the thorlabs stage.

- "vimba_driver" contains drivers to communicate with the MCP camera.

"ressources" contains diverse images and references phase map.

- "background" contains the wavefront correction images for the SLMs provided by their manufacturer.

- "beam_profiles" contains the beam profiles captured via the interface.

- "images_wgs" contains a set of examples images to run the wgs algorithm for phase aberration correction.

- "SLM_Aberration_correction_files" contains the correction phase-file calculated by the weighted Gerchberg-Saxton (WGS) routine.


