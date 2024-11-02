import cv2
import math
import numpy as np
from vehicle import Driver
from controller import GPS, Node

""" Robot car class """ 
class  RobotCar () : 
    SPEED = 20        # Cruising speed 
    UNKNOWN = 99999.99   # Value when yellow line cannot be detected 
    FILTER_SIZE = 3      # Filter for yellow line 
    TIME_STEP = 30     # Sensor sampling time [ ms] 
    CAR_WIDTH = 2.015  # Vehicle width [m] 
    CAR_LENGTH = 5.0    # Vehicle length [m]    
    
        
    """ Constructor """ 
    def  __init__ (self) :   
        # Welcome message
        self.welcomeMessage()    
        #Lane 
        self.mode = "SEARCHING"
        # Driver
        self.driver = Driver()
        self.driver.setSteeringAngle( 0 )
        self.driver.setCruisingSpeed(self.SPEED)
        self.driver.setDippedBeams( True ) # Headlight tipping
 
        
        # LIDAR (SICK LMS 291) 
        self.lidar = self.driver.getDevice( "Sick LMS 291" )
        self.lidar.enable(self.TIME_STEP)
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_range = self.lidar.getMaxRange()
        self.lidar_fov = self.lidar.getFov();    
        self.lidar.enablePointCloud()    
        
        # GPS 
        self.gps = self.driver.getDevice( "gps" )
        self.gps.enable(self.TIME_STEP)

        # Camera 
        self.camera = self.driver.getDevice( "camera" )
        self.camera.enable(self.TIME_STEP)
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()
        print( "camera: width=%d height=%d fov=%g" % \
            (self.camera_width, self.camera_height, self.camera_fov))

        # Display 
        self.display = self.driver.getDevice( "display" )
        self.display.attachCamera(self.camera) # show camera image 
        self.display.setColor( 0xFF0000 )
        
      
    """ Welcome message """     
    def  welcomeMessage (self) : 
        print( "******************************** ************" )
        print( "* Welcome to a simple robot car program *" )
        print( "**********************************************************" )       
                

    """ Let's implement a moving average filter.""" 
    def  maFilter (self, new_value) : 
        """
        Write the moving average filter code here
        """    
        return new_value

    
    """ Steering control: If wheel_angle is positive, turn right, if negative, turn left. """ 
    def  control (self, steering_angle) : 
        LIMIT_ANGLE = 0.5  # Steering angle limit [rad] 
        if steering_angle > LIMIT_ANGLE:
            steering_angle = LIMIT_ANGLE
        elif steering_angle < -LIMIT_ANGLE:
            steering_angle = -LIMIT_ANGLE
        self.driver.setSteeringAngle(steering_angle)
    
    
    """ Calculate the average difference between pixels and yellow """ 
    def  colorDiff (self, pixel, yellow) : 
        d, diff = 0 , 0 
        for i in range ( 0 , 3 ):
            d = abs(pixel[i] - yellow[i])
            diff += d
        return diff/ 3


    """ Calculating the steering angle to follow the yellow line """ 
    def calcSteeringAngle(self, image):
        YELLOW = [95, 187, 203]  # Yellow value (BGR format)
        sum_x = 0  # X-coordinate sum of yellow pixels
        pixel_count = 0  # Total number of yellow pixels
        if self.mode == "SEARCHING":
            start_x = int(0.5 * self.camera_width)  # start from the middle
            end_x = self.camera_width  # end at the right-most edge
        else:
            start_x = 0  # start from the left-most edge
            end_x = int(0.5 * self.camera_width)
        for y in range(int(1.0 * self.camera_height / 3.0), self.camera_height):
            for x in range(0, int(0.5 * self.camera_width)):  # only look at the left half
  
                if self.colorDiff(image[y, x], YELLOW) < 30:  # Yellow range
                    sum_x += x  # Sum of x-coordinates of yellow pixels
                    pixel_count += 1  # Sum of the number of yellow pixels
    
        if pixel_count == 0:
            return self.UNKNOWN
    
        # Calculate the average x-coordinate of yellow pixels
        y_ave = float(sum_x) / pixel_count
    
        # Define the desired position (e.g., 20% from the left edge)
        desired_position = 0.2 * self.camera_width
    
        # Calculate the steering angle based on the desired position
        steer_angle = (y_ave - desired_position) / self.camera_width * self.camera_fov
    
        # Limit the steering angle to stay on the left side of the yellow line
        MAX_STEER_ANGLE = 0.2  # Adjust this value to control the steering sensitivity
        steer_angle = max(-MAX_STEER_ANGLE, min(steer_angle, MAX_STEER_ANGLE))
    
        return steer_angle
                

    """ Returns the direction and distance of the obstacle. If the obstacle cannot be found, returns UNKNOWN.
        Detect obstacles in the rectangular area in front of the robot car    
    """ 
    def  calcObstacleAngleDist (self, lidar_data) : 
        OBSTACLE_HALF_ANGLE = 20.0  # Obstacle detection angle. 20[°] to the left and right of the center. 
        OBSTACLE_DIST_MAX = 20.0  # Maximum obstacle detection distance [m] 
        OBSTACLE_MARGIN = 0.1   # When avoiding obstacles Lateral margin [m]
        
        sumx = 0 
        collision_count = 0 
        obstacle_dist = 0.0 
        for i in range(int(self.lidar_width/ 2 - OBSTACLE_HALF_ANGLE), \
            int(self.lidar_width/ 2 + OBSTACLE_HALF_ANGLE)):
            dist = lidar_data[i]
            if dist < OBSTACLE_DIST_MAX:
                sumx += i
                collision_count += 1 
                obstacle_dist += dist
                
        if collision_count == 0  or obstacle_dist > collision_count * OBSTACLE_DIST_MAX:   # If no obstacle can be detected 
            return self.UNKNOWN, self.UNKNOWN
        else :
            obstacle_angle = (float(sumx) /(collision_count * self.lidar_width) - 0.5 ) * self.lidar_fov
            obstacle_dist = obstacle_dist/collision_count
            
            if abs(obstacle_dist * math.sin(obstacle_angle)) < 0.5 * self.CAR_WIDTH + OBSTACLE_MARGIN:
                 return obstacle_angle, obstacle_dist # else when colliding
            else :
                 return self.UNKNOWN, self.UNKNOWN


    """ Run """ 
    def  run (self) :  
        stop =   False 
        step = 0        
        while self.driver.step() != -1 :
            step += 1  
            BASIC_TIME_STEP = 10  # Simulation update is 10[ms]
            
            if stop == True :
                print( "Stop!" )
                 #self.driver.setBrakeIntensity(1.0) # Apply the brake. 
                self.driver.setCruisingSpeed( 0 )      # Emergency stop
               
            # Set sensor update to TIME_STEP[ms]. In this example 60[ms] 
            if step % int(self.TIME_STEP / BASIC_TIME_STEP) == 0 :
                 # Lidar
                lidar_data = self.lidar.getRangeImage()   
                
                # GPS
               
                Values = self.gps.getValues()     

                # Camera
                camera_image = self.camera.getImage()
                # OpenCV's Python handles images as numpy arrays, so convert them.            
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4 ))

                # Obstacle avoidance
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)
                STOP_DIST = 5  # Stopping distance [m]
                
                if obstacle_dist < STOP_DIST:
                    print( "%d:Find obstacles(angle=%g, dist=%g)" % \
                        (step, obstacle_angle, obstacle_dist))
                    stop = True 
                else :                
                     # Calculate the operation amount (steering angle)
                    steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))

                    if steering_angle != self.UNKNOWN:
                       
                        print("%d: Found the yellow line"% step)
                        self.driver.setCruisingSpeed(self.SPEED)   # Set the driving speed                      
                        self.control(steering_angle)               # Steering control (simplified version of proportional control) 
                        stop = False 
                    else :
                        
                        print("%d: Lost the yellow line"% step)
                        self.driver.setCruisingSpeed( 0 ) # Stop because the yellow line to follow is lost. 
                        # self.driver.setBrakeIntensity(0.5) # Stop because the yellow line to follow is lost. 
                        stop = True

        
""" Main function """  
def  main () :  
    robot_car = RobotCar()
    robot_car.run()  
    
""" Notation for using this script as a module """ 
if __name__ == '__main__' :
    main()