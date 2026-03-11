import numpy as np
import math

x_des = 274
y_des = 0

Kp = 0.5
Kd = 0.01
# Ki = 0.1
#sum = 0

freq = 10 #Hz

prev_theta_error = 0

def prelims_controller (x_c, y_c, theta_c, prev_theta_error):

    if x_c > 250: # section 3
        v = 0.5
        x_pos = x_des
        y_pos = y_des
        theta = math.atan(y_pos-y_c, x_pos-x_c)
    elif y_c > 7: # section 2
        v = 3 #max
        theta = 0 #go straight
    else: # section 1
        v = 0.5
        theta = math.pi/6
    
    # vel_error = math.sqrt(y_error **2 + x_error **2)
    theta_error = theta - theta_c

    # Kd * (vel_error - prev_vel_error) *freq + Ki * (vel_sum)
    w = Kp * theta_error + Kd * (theta_error - prev_theta_error)*freq
    prev_theta_error = theta_error
    # vel_sum += vel_error/freq
    # theta_sum += theta_error/freq

    
    return v,w, prev_theta_error