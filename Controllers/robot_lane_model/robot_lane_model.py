import os
import cv2
import math
import numpy as np
from vehicle import Driver
from controller import GPS, Camera
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class RobotCar:
    SPEED = 40
    UNKNOWN = 99999.99
    FILTER_SIZE = 3
    TIME_STEP = 30
    CAR_WIDTH = 2.015
    CAR_LENGTH = 5.0
    current_speed = SPEED

    def __init__(self):
        # Welcome message
        self.welcomeMessage()
        # Lane
        self.mode = "SEARCHING"
        # Driver
        self.driver = Driver()
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(self.SPEED)
        self.driver.setDippedBeams(True)  # Headlight tipping

        # LIDAR (SICK LMS 291)
        self.lidar = self.driver.getDevice("Sick LMS 291")
        self.lidar.enable(self.TIME_STEP)
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_range = self.lidar.getMaxRange()
        self.lidar_fov = self.lidar.getFov()
        self.lidar.enablePointCloud()

        # GPS
        self.gps = self.driver.getDevice("gps")
        self.gps.enable(self.TIME_STEP)

        # Camera
        self.camera = self.driver.getDevice("camera")
        self.camera.enable(self.TIME_STEP)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()
        print("camera: width=%d height=%d fov=%g" % (self.camera_width, self.camera_height, self.camera_fov))

        # Data capture variables
        self.X_data = []  # List to store images
        self.y_data = []  # List to store steering angles

        # Display
        self.display = self.driver.getDevice("display")
        self.display.attachCamera(self.camera)  # show camera image
        self.display.setColor(0xFF0000)

    def welcomeMessage(self):
        print("******************************** ************")
        print("* Welcome to a simple robot car program *")
        print("**********************************************************")

    def maFilter(self, new_value):
        return new_value

    def control(self, steering_angle):
        LIMIT_ANGLE = 0.5
        if steering_angle > LIMIT_ANGLE:
            steering_angle = LIMIT_ANGLE
        elif steering_angle < -LIMIT_ANGLE:
            steering_angle = -LIMIT_ANGLE
        self.driver.setSteeringAngle(steering_angle)

    def colorDiff(self, pixel, yellow):
        d, diff = 0, 0
        for i in range(0, 3):
            d = abs(pixel[i] - yellow[i])
            diff += d
        return diff / 3

    def calcSteeringAngle(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        sum_x = 0
        pixel_count = 0

        for y in range(int(1.0 * self.camera_height / 3.0), self.camera_height):
            for x in range(0, int(0.5 * self.camera_width)):
                if mask[y, x] == 255:
                    sum_x += x
                    pixel_count += 1

        if pixel_count == 0:
            return self.UNKNOWN

        y_ave = float(sum_x) / pixel_count

        desired_position = 0.24 * self.camera_width

        steer_angle = (y_ave - desired_position) / self.camera_width * self.camera_fov

        MAX_STEER_ANGLE = 0.3
        steer_angle = max(-MAX_STEER_ANGLE, min(steer_angle, MAX_STEER_ANGLE))

        return steer_angle

    def detect_traffic_light_state(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_mask = red_mask + red_mask2

        red_count = np.sum(red_mask == 255)
        green_count = np.sum(green_mask == 255)

        if red_count > green_count:
            return "RED"
        elif green_count > red_count:
            return "GREEN"
        else:
            return "UNKNOWN"

    def calcObstacleAngleDist(self, lidar_data):
        OBSTACLE_HALF_ANGLE = 20.0
        OBSTACLE_DIST_MAX = 20.0
        OBSTACLE_MARGIN = 0.1

        sumx = 0
        collision_count = 0
        obstacle_dist = 0.0

        for i in range(int(self.lidar_width / 2 - OBSTACLE_HALF_ANGLE),
                       int(self.lidar_width / 2 + OBSTACLE_HALF_ANGLE)):
            dist = lidar_data[i]
            if dist < OBSTACLE_DIST_MAX:
                sumx += i
                collision_count += 1
                obstacle_dist += dist

        if collision_count == 0 or obstacle_dist > collision_count * OBSTACLE_DIST_MAX:
            return self.UNKNOWN, self.UNKNOWN
        else:
            obstacle_angle = (float(sumx) / (collision_count * self.lidar_width) - 0.5) * self.lidar_fov
            obstacle_dist = obstacle_dist / collision_count

            if abs(obstacle_dist * math.sin(obstacle_angle)) < 0.5 * self.CAR_WIDTH + OBSTACLE_MARGIN:
                return obstacle_angle, obstacle_dist
            else:
                return self.UNKNOWN, self.UNKNOWN

    def __on_camera_image(self, message):
        img = message.data
        img = np.frombuffer(img, dtype=np.uint8).reshape((message.height, message.width, 4))
        img = img[160:190, :]

        # Capture data
        self.X_data.append(img)
        steering_angle = self.calcSteeringAngle(img)
        self.y_data.append(steering_angle)

    def run(self):
        stop = False
        step = 0

        while self.driver.step() != -1:
            step += 1
            BASIC_TIME_STEP = 10

            recognition_active = self.camera.hasRecognition()
            segmentation_enabled = self.camera.isRecognitionSegmentationEnabled()

            if recognition_active:
                print("recognition_active")
            else:
                print("Not recognition_active")

            if segmentation_enabled:
                print("Segmentation_active")
            else:
                print("Not Segmentatio_active")

            if stop:
                print("Stop!")
                self.driver.setCruisingSpeed(0)

            if step % int(self.TIME_STEP / BASIC_TIME_STEP) == 0:
                # Lidar
                lidar_data = self.lidar.getRangeImage()

                # GPS
                Values = self.gps.getValues()

                # Camera
                camera_image = self.camera.getImage()
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
                cv_image_bgr = cv_image[:, :, :3]
                state = self.detect_traffic_light_state(cv_image_bgr)
                cv2.waitKey(1)

                # Capture data
                self.X_data.append(cv_image_bgr)
                steering_angle = self.calcSteeringAngle(cv_image_bgr)
                self.y_data.append(steering_angle)

                if state == "RED":
                    self.driver.setCruisingSpeed(0)
                    stop = True
                elif state == "GREEN":
                    self.driver.setCruisingSpeed(self.SPEED)
                    self.control(steering_angle)
                    stop = False
                else:
                    self.driver.setCruisingSpeed(0)
                    stop = True

                # Obstacle avoidance
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)
                if RobotCar.current_speed > 30:
                    STOP_DIST = 40
                else:
                    STOP_DIST = 10

                obstacle_detected = False
                for dist in lidar_data:
                    if dist < STOP_DIST:
                        obstacle_detected = True
                        break

                if obstacle_dist < STOP_DIST:
                    print("%d:Find obstacles(angle=%g, dist=%g)" % (step, obstacle_angle, obstacle_dist))
                    stop = True
                else:
                    steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))

                    if steering_angle != self.UNKNOWN:
                        print("%d: Found the yellow line" % step)
                        self.driver.setCruisingSpeed(self.SPEED)
                        self.control(steering_angle)
                        stop = False
                    else:
                        print("%d: Lost the yellow line" % step)
                        self.driver.setCruisingSpeed(0)
                        self.driver.setBrakeIntensity(0.5)
                        stop = True

        # Save captured data to a file after running the simulation
        np.savez('captured_Training_data.npz', X=np.array(self.X_data), y=np.array(self.y_data))

# Function to load the captured data
def load_captured_data(file_path):
    data = np.load(file_path)
    X_data = data['X']
    y_data = data['y']
    return X_data, y_data

# Load captured data
X_data, y_data = load_captured_data('captured_Training_data.npz')

# Visualize a sample image and its corresponding steering angle
if X_data.shape[0] > 0:
    sample_image = X_data[8]  # Assuming the first element is an image
    sample_steering_angle = y_data[8]
    plt.imshow(sample_image)
    plt.title(f"Sample Image\nSteering Angle: {sample_steering_angle}")
    plt.show()
else:
    print("No images in the dataset.")

# Normalize the images
X_data = X_data / 255.0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Define a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(240, 360, 3)))
model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))  # Output layer for steering angle

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save the model
model.save('lane_following_model.h5')
