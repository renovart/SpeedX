
import cv2
import math
import numpy as np
from keras.models import load_model
from vehicle import Driver

class RobotCar:
    SPEED = 30
    UNKNOWN = 99999.99
    FILTER_SIZE = 3
    TIME_STEP = 30
    CAR_WIDTH = 2.015
    CAR_LENGTH = 5.0
    current_speed = SPEED
 #, model_path='final_lane_following_model'
    def __init__(self):
        self.welcomeMessage()

        # Lane
        self.mode = "SEARCHING"

        # Driver
        self.driver = Driver()
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(self.SPEED)
        self.driver.setDippedBeams(True)  # Headlight tipping

        # Load the pre-trained lane-following model
        self.lane_following_model = torch.hub.load('hustv/yolop','yolop',pretrained=true)

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

        # Display
        self.display = self.driver.getDevice("display")
        self.display.attachCamera(self.camera)  # show camera image
        self.display.setColor(0xFF0000)

        # Enable camera recognition
        self.camera.recognitionEnable(self.TIME_STEP)

        # Enable camera recognition segmentation
        
        self.camera.enableRecognitionSegmentation()



    def welcomeMessage(self):
        print("******************************** ************")
        print("* Welcome to a simple robot car program *")
        print("**********************************************************")

    def maFilter(self, new_value):
        return new_value

    def control(self, steering_angle):
        LIMIT_ANGLE = 0.5  # Steering angle limit [rad]
        if steering_angle > LIMIT_ANGLE:
            steering_angle = LIMIT_ANGLE
        elif steering_angle < -LIMIT_ANGLE:
            steering_angle = -LIMIT_ANGLE
        self.driver.setSteeringAngle(steering_angle)

    def preprocess_image(self, image):
        # Resize the image to the required input size
        input_width, input_height = 360, 240  # Adjust based on your model requirements
        resized_image = cv2.resize(image[:, :, :3], (input_width, input_height))  # Use only the first three channels
        
        # Extract the RGB channels and normalize to [0, 1]
        normalized_image = resized_image / 255.0
        
        # Ensure the image has the expected shape (add batch dimension)
        input_image = np.expand_dims(normalized_image, axis=0)
        
        return input_image

    
    
       
    def calcSteeringAngle(self, image):
        # Use the model to predict the steering angle
        input_image = self.preprocess_image(image)
        
        predicted_steering_angle = self.lane_following_model.predict(input_image)[0, 0]

        MAX_STEER_ANGLE = 0.3
        predicted_steering_angle = max(-MAX_STEER_ANGLE, min(predicted_steering_angle, MAX_STEER_ANGLE))
        return predicted_steering_angle

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
  

    def run(self):
        stop = False
        step = 0

        while self.driver.step() != -1:
            step += 1
            BASIC_TIME_STEP = 10

            recognition_active = self.camera.hasRecognition()
            segmentation_enabled = self.camera.isRecognitionSegmentationEnabled()
            
            

            if recognition_active:
                print("Recognition active")

            if segmentation_enabled:
                print("Segmentation active")

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
                print("Image shape:", cv_image.shape)
                cv_image_bgr = cv_image[:, :, :3]

                state = self.detect_traffic_light_state(cv_image_bgr)

                if state == "RED":
                    self.driver.setCruisingSpeed(0)
                    stop = True
                elif state == "GREEN":
                    self.driver.setCruisingSpeed(self.SPEED)
                    stop = False
                else:
                    self.driver.setCruisingSpeed(0)
                    stop = True

                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)

                if RobotCar.current_speed > 40:
                    STOP_DIST = 100
                else:
                    STOP_DIST = 10

                obstacle_detected = False
                for dist in lidar_data:
                    if dist < STOP_DIST:
                        obstacle_detected = True
                        break

                if obstacle_dist < STOP_DIST:
                    print("%d: Find obstacles(angle=%g, dist=%g)" % (step, obstacle_angle, obstacle_dist))
                    stop = True
                else:
                    # Use the modified method to calculate the steering angle
                    predicted_steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))

                    if predicted_steering_angle != self.UNKNOWN:
                        print("%d: Found the lane" % step)
                        self.driver.setCruisingSpeed(self.SPEED)
                        self.control(predicted_steering_angle)
                        stop = False
                    else:
                        print("%d: Lost the lane" % step)
                        self.driver.setCruisingSpeed(0)
                        self.driver.setBrakeIntensity(0.5)
                        stop = True

def main():
    robot_car = RobotCar()
    robot_car.run()

if __name__ == '__main__':
    main()
