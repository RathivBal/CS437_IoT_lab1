import sys
sys.path.append('./picar_4wd')
from constants import * # my constant values
from my_utils import * # my utiity methods
import picar_4wd as fc
from threading import Thread

obstacle_detected = False # for tracking if there is an obstacle at any point

# drive around, if there is an obstacle, pic a random direction, back up then
# turn to face that direction
def drive_around():

    while not get_quit_pressed():
        # print("OBSTACLE detected in drive_around: " + str(obstacle_detected))
        if obstacle_detected:
            fc.stop() #stop the car
            delay(500) # delay for half a second
            direction = randrange(90,270) # pick a new direction
            while obstacle_detected:  #backup until no longer detect obstacle
                backward(5,1)
                turn(direction) # turn in chosen direction
                delay(500) #hold for hal a second

        else:
            forward(1,1)
        return

#continously scan for obstcles and update obstacle status
# @parm threshold - range on ultrasonic sensor at which an obstacle is
# considered detected
def scan_for_obstacles(threshold):
    # set angle 0 by default
    angle = 0
    scanning_right = True # start by scanning to the right
    while not get_quit_pressed():
        # detect an obstacle if object detected at or beyond threshold distance
        distance = fc.get_distance_at(angle)
        obstacle_detected = distance != -2 and distance <=threshold
	# adjust angle, switching driection if hit MIN and MAX angle bounds
        if scanning_right:
            angle += ANGLE_STEP
            if angle == MAX_ANGLE:
                scanning_right = False
	   
            else:
                angle -= ANGLE_STEP
            if angle == MIN_ANGLE:
                scanning_right = True
    return

# drive around while scanning for obstacles, if one is found pick a random
# direction, back up, then turn to that direction and continue
# press 'q' to quit
def run_obstacle_avoidance():
    global obstacle_detected, OBSTACLE_THRESHOLD

    quit_thread = Thread(target=read_keyboard_for_quit, daemon=True)
    scan_thread = Thread(target=scan_for_obstacles, args=(OBSTACLE_THRESHOLD,),daemon=True)
    drive_thread = Thread(target=drive_around, daemon=True)
    
    quit_thread.start()
    scan_thread.start()
    drive_thread.start()

def main():

    run_obstacle_avoidance()

    # keep running until quit_thread presed
    while not get_quit_pressed():
        continue
    sys.exit()

if __name__ == "__main__" :
    try:
        main()
    finally:
         fc.stop()	 
				 
    
	
