### heavily based on: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
### see doc here: https://intelrealsense.github.io/librealsense/python_docs/
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseInterface:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

        rs.align(rs.stream.color)

        # Start streaming
        self.profile = self.pipeline.start(config)
        #

        color_sensor = self.profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        # Set fixed exposure (experiment with value, e.g., 100)
        color_sensor.set_option(rs.option.exposure, 100)

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # scaling factor to use when converting from get_data() units to meters
        self.depth_units = depth_frame.get_units()

        # Convert images to numpy arrays
        bgr_image = np.asanyarray(color_frame.get_data())
        img = self.bgr2rgb(bgr_image)
        img = self.center_crop(img, 480, 480)
        img = cv2.resize(img, (256, 256))
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_units

        return img, depth_image

    def rgb2bgr(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def bgr2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def center_crop(self, image, target_height, target_width):
        h, w = image.shape[:2]
        start_x = (w - target_width) // 2
        start_y = (h - target_height) // 2
        return image[start_y : start_y + target_height, start_x : start_x + target_width]

    def show_frame(self, color_image, depth_image=None):
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        bgr_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        if depth_image is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image / self.depth_units, alpha=0.03), cv2.COLORMAP_JET
            )
            img = np.hstack((bgr_img, depth_colormap))
        else:
            img = bgr_img
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    def close(self):
        # Stop streaming
        self.pipeline.stop()
