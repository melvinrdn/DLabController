from drivers import gxipy_driver as gx
import numpy as np


class DahengCamera(object):
    """
    A class to manage the operation of a Daheng camera using the GX driver.

    Attributes:
        index (int): The index of the camera to be initialized.
        cam (gx.Device): The camera device object.
        imshape (tuple): The shape of the captured image array.

    Methods:
        __init__(index):
            Initializes the camera by connecting to the device, setting default parameters,
            and capturing an initial image to get the shape.
        set_exposure_gain(exposure, gain):
            Sets the camera's exposure time and gain.
        take_image(exposure, gain, avgs):
            Captures an image with specified exposure, gain, and averaging across multiple frames.
        close_daheng():
            Closes the connection to the camera and cleans up resources.
    """
    def __init__(self, index):
        """
        Initializes the DahengCamera object and connects to the camera at the given index.

        Parameters:
            index (int): The index of the camera to connect to.

        Initializes the camera with default settings (exposure time of 10000, gain of 1,
        software-trigger mode). Also captures an initial image to set the image shape.

        If connection fails, sets self.cam and self.imshape to None.
        """
        self.index = index
        device_manager_ = gx.DeviceManager()
        dev_num, dev_info_list = device_manager_.update_device_list()

        try:
            self.cam = device_manager_.open_device_by_index(int(self.index))
            print(f"from DahengCamera: Camera {self.index} connected")
            self.cam.ExposureTime.set(10000)
            self.cam.Gain.set(1)
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self.cam.stream_on()
            self.cam.TriggerSoftware.send_command()
            im = self.cam.data_stream[0].get_image()
            np_im = im.get_numpy_array()
            self.cam.stream_off()
            self.imshape = np.shape(np_im)

        except:
            print(f"from DahengCamera: Impossible to connect the camera {self.index}")
            self.cam = None
            self.imshape = None

    def set_exposure_gain(self, exposure, gain):
        """
        Sets the exposure time and gain for the camera.

        Parameters:
            exposure (float): The desired exposure time.
            gain (float): The desired gain value.
        """
        if self.cam is not None:
            self.cam.ExposureTime.set(exposure)
            self.cam.Gain.set(gain)

    def take_image(self, exposure, gain, avgs):
        """
        Captures an image from the camera with the specified exposure, gain, and averaging.

        Parameters:
            exposure (float): The exposure time to set for the image capture.
            gain (float): The gain value to set for the image capture.
            avgs (int): The number of frames to average.

        Returns:
            np.ndarray: The averaged image as a numpy array, or None if the camera is not connected.
        """
        if self.cam is not None:
            self.set_exposure_gain(exposure, gain)
            self.cam.stream_on()
            res = np.zeros([self.imshape[0], self.imshape[1]])
            for i in range(avgs):
                self.cam.TriggerSoftware.send_command()
                im = self.cam.data_stream[0].get_image()
                np_image = im.get_numpy_array()
                res = res + np_image
            self.cam.stream_off()
            res = res / avgs
            return res
        else:
            return None

    def close_daheng(self):
        """
        Closes the connection to the camera.

        If the camera is connected, it will close the device and release resources.
        If the camera is not connected, it will print a message indicating that
        the camera object is None.
        """
        if self.cam is not None:
            self.cam.close_device()
            print(f"from DahengCamera: Camera {self.index} disconnected")
        else:
            print("from DahengCamera: self.cam is None")


device_manager = gx.DeviceManager()





