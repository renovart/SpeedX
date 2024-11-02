import cv2
import math
import simpleaudio

import numpy as np
from vehicle import Driver
from controller import GPS,Camera, Node

class RobotCar:
    SPEED = 30
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

        # Display
        self.display = self.driver.getDevice("display")
        self.display.attachCamera(self.camera)  # show camera image
        self.display.setColor(0xFF0000)

        # Enable camera recognition
        self.camera.recognitionEnable(self.TIME_STEP)

        # Enable camera recognition segmentation
        
        self.camera.enableRecognitionSegmentation()
        #Play Sound 
        
        self.obstacle_sound = r'C:\Users\Emmanuel\Desktop\Graduation\vehicles\controllers\robot_recognition\Warning.wav'


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

        sum_x = 0  # X-coordinate sum of yellow pixels
        pixel_count = 0  # Total number of yellow pixels

        for y in range(int(1.0 * self.camera_height / 3.0), self.camera_height):
            for x in range(0, int(0.5 * self.camera_width)):
                if mask[y, x] == 255:  # Check if pixel is yellow
                    sum_x += x  # Sum of x-coordinates of yellow pixels
                    pixel_count += 1  # Sum of the number of yellow pixels

        if pixel_count == 0:
            return self.UNKNOWN

        y_ave = float(sum_x) / pixel_count
        desired_position = 0.24 * self.camera_width
        steer_angle = (y_ave - desired_position) / self.camera_width * self.camera_fov
        MAX_STEER_ANGLE = 0.3
        steer_angle = max(-MAX_STEER_ANGLE, min(steer_angle, MAX_STEER_ANGLE))
        
            # Draw contours around yellow pixels
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
        # Display the image with contours (optional)
        # cv2.imshow("Contours", image)
        cv2.waitKey(1)
    
        return steer_angle
    
  
    
    
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
    # def play_sound(self):
        # print("Playing sound from:", self.obstacle_sound)
        # wave_obj = simpleaudio.WaveObject.from_wave_file(self.obstacle_sound)
        # play_obj = wave_obj.play()
        # play_obj.wait_done()
      

    def run(self):
        stop = False
        step = 0

        while self.driver.step() != -1:
            step += 1
            BASIC_TIME_STEP = 10

            recognition_active = self.camera.hasRecognition()
            segmentation_enabled = self.camera.isRecognitionSegmentationEnabled()

            # if recognition_active:
                # print("Recognition active")

            # if segmentation_enabled:
                # print("Segmentation active")

            if stop:
                print("Stop!")
                # self.play_sound()
                self.driver.setCruisingSpeed(0)
            cv_image_bgr = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)  # Default value   

            if step % int(self.TIME_STEP/ BASIC_TIME_STEP) == 0:
                # Lidar
                lidar_data = self.lidar.getRangeImage()

                # GPS
                Values = self.gps.getValues()
                
                # Camera
                camera_image = self.camera.getImage()
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
                cv_image_bgr = cv_image[:, :, :3]

               
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)
                if  RobotCar.current_speed > 40:
                    STOP_DIST = 140  # Stopping distance [m] when speed is more than 30
                else:
                    STOP_DIST = 15  # Stopping distance [m] when speed is below 30
                
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
                        print("%d: Found the lane" % step)
                        self.driver.setCruisingSpeed(self.SPEED)
                        self.control(steering_angle)
                        stop = False
                    else:
                        print("%d: Lost the lane" % step)
                        self.driver.setCruisingSpeed(0)
                        self.driver.setBrakeIntensity(0.5)
                        stop = True
def  main () :  
    robot_car = RobotCar()
    robot_car.run()
    
if __name__ == '__main__':
    main()
