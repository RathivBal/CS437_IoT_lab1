import picar_4wd as fc
import sys
import tty
import termios
import asyncio

#power_val = 50
key = 'status'
quit_pressed = False
print("If you want to quit.Please press q")

def delay(time):
    fc.time.sleep(time)
    
def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def Keyborad_control():
    while True:
        global power_val
        key=readkey()
        if key=='6':
            if power_val <=90:
                power_val += 10
                print("power_val:",power_val)
        elif key=='4':
            if power_val >=10:
                power_val -= 10
                print("power_val:",power_val)
        if key=='w':
            fc.forward(power_val)
        elif key=='a':
            fc.turn_left(power_val)
        elif key=='s':
            fc.backward(power_val)
        elif key=='d':
            fc.turn_right(power_val)
        else:
            fc.stop()
        if key=='q':
            print("quit")  
            break  

# neatly print sensor readings
def print_readings(readings):
    print("Readings:")
    for reading in readings:
        if reading >= INF - 1:
            print("INF", end=' ')
        else:
            print("reading", end=' ')
    print()
    return
    
# turns the car any number of degrees in [-inf,inf]
# Note: this needs to be updated beacuse jst sleeping doesn't do items
# The speed at which the car moves heavily depends on the battery charge
# need to instead use wheel turning amount, which means we need a fixed speed
# sensor.UGH
def turn(deg):
    # stop the car and delay for a moment
    fc.stop()
    delay(100)
    
    # normalize deg to value in (0,359)
    deg = deg % 360
    
    # check if we should turn left or right
    turn_right = False
    if deg > 180:
        turn_right = True
        deg = 360 - deg
    # calculate the number of stes to turn (different for left vs right becasue
    # of misaligned wheels  - time should be in ratio of 23:20)
    if turn_right:
        delay_steps = round(deg * (5 + (1/3)))
    else:
        delay_steps = round(deg * 4.8)
    # now we turn left or right based on number of steps to sleep
    if turn_right: # if deg > 180, we turn right
        #print("\nTurning right " +str(deg) + " "degrees", flush=True)
        fc.turn_right(100)
    else: # else , turn left
        #print("\nTurning left " +str(deg) + " "degrees", flush=True)
        fc.turn_left(100)
    delay(delay_steps) # delay
    fc.stop() # stop car
  
# move the car forward
# @parm power - power level to move forward
# @parm steps - number of deciseconds to move for
def forward(power, steps):
    fc.forward(power) # set car to move forward
    delay(steps) # delay 
    fc.stop() # stop car
    
# move the car backward
# @parm power - power level to move forward
# @parm steps - number of deciseconds to move for
def backward(power, steps):
    fc.backward(power) # set car to move backward
    delay(steps) # delay
    fc.stop() # stop car
    return
    
# poll keyboard for q key to quit program (set quit_pressed to True)
def read_keyboard_for_quit():
    
    print ("Press q to quit")
    while not quit_pressed:
        key = read_key()
        if key == 'q':
            print ("Done")
            quit_pressed = True
    return
    
def get_quit_pressed():
    return get_quit_pressed
    
# my adaptaion of professor code to move forward by some multiple of 2.5cm
# @parm steps - how many multiples of 2.5cm the car should move forward
def forward_2_5_cm(steps):
    fc.forward(100)
    delay(int(round(steps * 100 / 3.264)))
    fc.stop()
    
# convert polar coorindates to cartesian coordinates
def polar_to_cartesian(r, theta):
    x_s = r * np.cos(theta)
    y_s = r * np.sin(theta)
    return np.column_stack((x_s, y_s))
    