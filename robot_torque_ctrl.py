
import math
import matplotlib.pyplot as plt
import numpy as np


animate_theta = False
pd_control = False
pid_control = not pd_control

M = 1
m = 1
b = 1
r = 0.5

delta_t = 0.02
if pid_control:
    g = 9.81
else:
    g = 0

# P Control
def controller_pd_control(theta_des, theta_dot_des, theta_t, theta_dot_t):
    K_P = 15.0
    K_D = 2* math.sqrt(K_P * M) - b # ensures zeta is 1 (=critically damped)
    u_t = K_P * (theta_des-theta_t) + K_D * (theta_dot_des - theta_dot_t)

    return u_t

# PI Control
def controller_pid_control(theta_des, theta_dot_des, theta_ddot_des, theta_t, theta_dot_t, sum_error_tm1):
    K_P = 10.0
    K_D = 1.8
    K_I = 4.0 # good starting point is 0, but beware steady-state error

    theta_error = theta_des - theta_t
    theta_error_dot = theta_dot_des - theta_dot_t
    sum_error_t = sum_error_tm1 + theta_error * delta_t

    u_t = K_P * (theta_error) + K_I * (sum_error_t) + K_D * (theta_error_dot)

    return u_t, sum_error_t


def sim_state_update(theta_tm1, theta_dot_tm1, u_t):
    theta_ddot_t = 1/M*(-b*theta_dot_tm1 -m*g*r*math.cos(theta_tm1) + u_t)
    theta_dot_t = theta_dot_tm1 + theta_ddot_t*delta_t
    theta_t = theta_tm1 + theta_dot_t*delta_t

    return theta_t, theta_dot_t

######## MAIN ########
t_max = 10.0
t = 0
theta_des = 1
theta_dot_des = 0#0.1
theta_ddot_des = 0#0.01
theta_t = 0
theta_dot_t = 0
sum_error_t = 0
t_list = []
theta_t_list = []
theta_dot_t_list = []
while t < t_max:
    if pid_control:
        u_t, sum_error_t = controller_pid_control(theta_des, theta_dot_des, theta_ddot_des, theta_t, theta_dot_t, sum_error_t)
    else:
        u_t = controller_pd_control(theta_des, theta_dot_des, theta_t, theta_dot_t)
    theta_t, theta_dot_t = sim_state_update(theta_t, theta_dot_t, u_t)
    theta_t_list.append(theta_t)
    theta_dot_t_list.append(theta_dot_t)
    t_list.append(t)
    t += delta_t


# Animate the robot
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

