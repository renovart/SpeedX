import math                  # Import math module 
from vehicle import Driver   # Import Driver class from vehcle module from 
from controller import GPS #   Import GPS class from controller module

TIME_STEP = 60               # GPS sampling time 60[ms]  

driver = Driver()              # Generate an instance of the Driver class 
driver.setSteeringAngle( 0.0 )   # Set the steering angle to 0.0[rad] 
driver.setCruisingSpeed( 20 )    # Set the cruising speed to 20[km/h]

# GPS 
gps = driver.getDevice( "gps" ) # Generate an instance of the class from the device name 
gps.enable(TIME_STEP)          # Data can be acquired from GPS at sampling interval TIME_STEP

coordinate = gps.getCoordinateSystem()         # Get coordinate system 
print( "Coordinate system is " ,coordinate)

while driver.step() != -1 :
    angle = 0.03 * math.cos(driver.getTime()) # Calculate steering angle 
    driver.setSteeringAngle(angle)             # Set steering angle
    
    # GPS 
    values = gps.getValues()                   # Get data from GPS 
    print( "%g[s] position:(%g, %g, %g9[m]" % (driver.getTime(),values[ 0 ] , values[ 1 ], values[ 2 ]))

    if driver.getTime() > 5.0 :                 # After 5[s], 
        driver.setCruisingSpeed( 0.0 )           # Stop