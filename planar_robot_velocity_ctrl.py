
import math
import matplotlib.pyplot as plt
import numpy as np

# Set time step
delta_t = 0.02

# P Control
def controller_p_control(theta_des, theta_t):
    K_P = 5.0
    u_t = K_P * theta_error_t

    return u_t

# PI Control
def controller_pi_control(theta_des, theta_t, sum_error_tm1):
    K_P = 4.0
    K_I = 4.0
    sum_error_tm1 += theta_error_t

    # anti-windup: because discrete sum
    saturation = 0.3
    if (sum_error_tm1 > saturation):
        sum_error_tm1 = saturation

    u_t = K_P * theta_error_t + K_I * (sum_error_tm1)

    return u_t, sum_error_tm1

# Simulate
def sim_state_update(theta_des, theta_error_tm1, u_t):
    theta_error_dot_t = theta_dot_des - u_t
    theta_error_t = theta_error_dot_t * delta_t + theta_error_tm1
    theta_t = theta_des - theta_error_t

    return theta_t, theta_error_t



######## MAIN ########
t_max = 12.0
t = 0
theta_des = 1
theta_dot_des = 0.5
theta_t = 0
theta_dot_t = 0
sum_error_t = 0
t_list = []
theta_t_list = []
theta_dot_t_list = []
theta_error_t = theta_des - theta_t
while t < t_max:
    
    # Store variables for plotting
    theta_t_list.append(theta_t)
    theta_dot_t_list.append(theta_dot_t)
    t_list.append(t)
    
    # Set control signal
    u_t, sum_error_t = controller_pi_control(theta_des, theta_t, sum_error_t)
    #u_t = controller_p_control(theta_des, theta_t)

    # Simulate
    theta_t, theta_error_t = sim_state_update(theta_des, theta_error_t, u_t)
    t += delta_t




# Animate the robot
animate_theta = True
if animate_theta:
    for t in range(len(theta_t_list)):
        plt.clf()
        plt.plot([0, math.cos(theta_t_list[t])],[0, math.sin(theta_t_list[t])],'k',linewidth=15)
        plt.plot([0],[0],'wo')
        plt.axis('equal')
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.draw()
        plt.pause(delta_t)
    plt.show()

# Plot the time trend
plt.clf()
plt.plot([0,t_max],[theta_des, theta_des],'c--')
plt.plot(t_list, theta_t_list,'k')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.show()

