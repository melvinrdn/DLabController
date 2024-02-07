from drivers import gxipy_driver as gx
import numpy as np


class DahengCamera(object):
    def __init__(self, index):
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
            print("from DahengCamera: womp womp")
            self.cam = None
            self.imshape = None

    def set_exposure_gain(self, exposure, gain):
        if self.cam is not None:
            self.cam.ExposureTime.set(exposure)
            self.cam.Gain.set(gain)

    def take_image(self, exposure, gain, avgs):
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
        if self.cam is not None:
            self.cam.close_device()
            print(f"from DahengCamera: Camera {self.index} disconnected")
        else:
            print("from DahengCamera: self.cam is None")


device_manager = gx.DeviceManager()





