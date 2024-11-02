from controller import Camera, Display, Robot
import cv2
import numpy as np
from ultralytics import YOLO

class RobotCamera:
    def __init__(self, robot):
        self.robot = robot

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(10)  # Enable the camera with a time step of 10ms
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()

        # Display
        self.display = self.robot.getDevice("display")

    def save_image(self, cv_image):
        # Save the image to a file
        cv2.imwrite('camera_image.png', cv_image)

        # Display the image using cv2.imshow
        cv2.imshow('Camera Image', cv_image)
        cv2.waitKey(1)

    def run(self):
        while self.robot.step(10) != -1:  # Use a time step of 10ms
            # Read frame from the camera
            camera_image = self.camera.getImage()
            cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
            cv_image_bgr = cv_image[:, :, :3]

            # Perform YOLOv8 inference
            model = YOLO('best.pt')
            results = model(cv_image_bgr, show=False, conf=0.4, save=False)

            self.save_image(cv_image_bgr)

if __name__ == '__main__':
    robot = Robot()
    robot_camera = RobotCamera(robot)
    robot_camera.run()
