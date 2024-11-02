import cv2
import math
import numpy as np
from vehicle import Driver
from controller import GPS,Camera, Node

class RobotCar:
    SPEED = 25
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
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # Define color thresholds for yellow and white
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
    
        # Create masks for yellow and white regions
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
        # Combine the masks
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
    
        # Apply Gaussian blur to the combined mask
        blurred_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    
        # Apply Canny edge detector
        edges = cv2.Canny(blurred_mask, 50, 150)
    
        # Define the region of interest (ROI)
        roi_vertices = np.array([[(0, self.camera_height), (0, 2*self.camera_height/3),
                                  (self.camera_width, 2*self.camera_height/3), (self.camera_width, self.camera_height)]],
                                dtype=np.int32)
        masked_edges = self.region_of_interest(edges, roi_vertices)
    
        # Apply Hough line transform to detect lines in the edges
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
        if lines is not None:
            # Filter lines based on length and slope
            filtered_lines = [line[0] for line in lines if self.filter_line(line[0])]
    
            # Average the slopes of detected lines
            slope_sum = 0
            count = 0
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                slope_sum += slope
                count += 1
    
            if count > 0:
                average_slope = slope_sum / count
                # Calculate the steering angle based on the average slope
                steer_angle = math.atan(average_slope)
                MAX_STEER_ANGLE = 0.3
                steer_angle = max(-MAX_STEER_ANGLE, min(steer_angle, MAX_STEER_ANGLE))
    
                # Draw the detected lines on the image (optional)
                for line in filtered_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
                # Display the image with detected lines (optional)
                cv2.imshow("Detected Lines", image)
                cv2.waitKey(1)
    
                return steer_angle
    
        return self.UNKNOWN

    def filter_line(self, line):
        # Filter lines based on length and slope
        x1, y1, x2, y2 = line
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
    
        # Define threshold values for line filtering
        min_length = 30
        min_slope = 0.1
        max_slope = 10.0
    
        return length > min_length and min_slope < abs(slope) < max_slope
    
    def region_of_interest(self, img, vertices):
        # Define a mask and apply it to the image
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img


    
    
    def detect_traffic_light_state(self, image):
    # Extract red and green channels
        red_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
    
        # Define intensity thresholds
        red_threshold = 200
        green_threshold = 100
    
        # Check if the average intensity in the red channel is above the threshold
        if np.mean(red_channel) > red_threshold:
            return "RED"
    
        # Check if the average intensity in the green channel is above the threshold
        elif np.mean(green_channel) > green_threshold:
            return "GREEN"
    
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

            # if recognition_active:
                # print("Recognition active")

            # if segmentation_enabled:
                # print("Segmentation active")

            if stop:
                print("Stop!")
                self.driver.setCruisingSpeed(0)
            cv_image_bgr = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)  # Default value   

            if step % int(self.TIME_STEP/ BASIC_TIME_STEP) == 0:
                # Lidar
                lidar_data = self.lidar.getRangeImage()

                # GPS
                Values = self.gps.getValues()
                state = self.detect_traffic_light_state(cv_image_bgr)
                print("Traffic Light State:", state)  # Add this line to print the state to the console
    
                # Display traffic light state on the simulation window
                self.display.drawText("Traffic Light State: " + state, 10, 10)

                # Camera
                camera_image = self.camera.getImage()
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))
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
                    steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))

                    if steering_angle != self.UNKNOWN:
                        print("%d: Found the lane" % step)
                        self.driver.setCruisingSpeed(self.SPEED)
                        self.control(steering_angle)
                        stop = False
                    else:
                        print("%d: Lost the yellow line" % step)
                        self.driver.setCruisingSpeed(0)
                        self.driver.setBrakeIntensity(0.5)
                        stop = True
def  main () :  
    robot_car = RobotCar()
    robot_car.run()
    
if __name__ == '__main__':
    main()
