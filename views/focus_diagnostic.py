
##
from drivers import gxipy_driver as gx
import numpy as np
from matplotlib import pyplot as plt





class DahengCamera(object):
    def __init__(self,index):
        self.index = index
        device_manager_ = gx.DeviceManager()
        device_manager_ = gx.DeviceManager()
        dev_num, dev_info_list = device_manager_.update_device_list()

        try:
            self.cam = device_manager_.open_device_by_index(int(index))
            self.cam.ExposureTime.set(10000)
            self.cam.Gain.set(20)
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self.cam.stream_on()
            self.cam.TriggerSoftware.send_command()
            im = self.cam.data_stream[0].get_image()
            np_im = im.get_numpy_array()
            self.cam.stream_off()
            self.imshape = np.shape(np_im)

        except:
            print("Camera could not get initialized :(")
            self.cam = None
            self.imshape = None

    def set_exposure_gain(self,exposure,gain):
        if self.cam is not None:
            self.cam.ExposureTime.set(exposure)
            self.cam.Gain.set(gain)

    def take_image(self,avgs):
        if self.cam is not None:
            self.cam.stream_on()
            res = np.zeros([self.imshape[0], self.imshape[1]])
            for i in range(avgs):
                self.cam.TriggerSoftware.send_command()
                im = self.cam.data_stream[0].get_image()
                np_image = im.get_numpy_array()
                res = res + np_image
            self.cam.stream_off()
            res = res/avgs
            return res
        else:
            return None

    def close_daheng(self):
        try:
            self.cam.close_device()
        except:
            print("There was nothing to close")
##

#device_manager = gx.DeviceManager()
#dev_num, dev_info_list = device_manager.update_device_list()
##

#device_manager = gx.DeviceManager()
#dev_num, dev_info_list = device_manager.update_device_list()
device_manager = gx.DeviceManager()
camera = DahengCamera(1)

##
#device_manager = gx.DeviceManager()

camera.set_exposure_gain(80000,20)

im = camera.take_image(2)
##
camera.close_daheng()














##
# create a device manager




##
# open the first device
#cam1 = device_manager.open_device_by_index(int(1))

##
#cam1.ExposureTime.set(10000)
#cam1.Gain.set(20)


##

#cam1.TriggerMode.set(gx.GxSwitchEntry.ON)
#cam1.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
##
#cam1.stream_on()
#cam1.TriggerSoftware.send_command()
#im = cam1.data_stream[0].get_image()
#numpy_image = im.get_numpy_array()
#plt.imshow(numpy_image)
#cam1.stream_off()
##
#cam1.close_device()


##
#plt.figure(123)
#plt.imshow(numpy_image)
#plt.show()