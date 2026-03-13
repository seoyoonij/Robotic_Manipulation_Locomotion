import math
import numpy as np
import scipy

def forward_kinematics(theta1, theta2, theta3):
    def rotation_x(angle):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])

    def rotation_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def translation(x, y, z):
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    T_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta1)
    T_1_2 = rotation_y(-1.57080) @ rotation_z(theta2)
    T_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta3)
    T_3_ee = translation(0.06231, -0.06216, 0.01800)
    T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
    return T_0_ee[:3, 3]


# 2a.
def get_error_leg (theta, target_ee):
    error = target_ee - forward_kinematics(theta[0],theta[1],theta[2])
    return error.dot(error) # return magnitude

def inverse_kinematics_with_optimizer (target_ee):
    initial_guess = np.array([0.0, 0.0, 0.0])
    res = scipy.optimize.minimize(get_error_leg, initial_guess, args = (target_ee))
    return res.x

#2b. 
def get_cost (theta, target_ee):
    error = target_ee - forward_kinematics(theta[0],theta[1],theta[2]) # error vector
    C = np.sum(error ** 2) # sum of error^2
    mean_error = np.mean(np.abs(error)) # mean|error|
    return C, mean_error

def get_gradient (theta, target_ee):
    eps = 1e-5
    grd = np.zeros(3)

    for i in range(3): # get gradient for each axis
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += eps
        theta_minus[i] -= eps
        C_plus, _ = get_cost(theta_plus, target_ee)
        C_minus, _ = get_cost(theta_minus, target_ee)

        grd[i] = (C_plus - C_minus) / (2*eps)

    return grd

def inverse_kinematics_with_gradient (target_ee):
    learning_rate = 0.02
    max_steps = 5000
    tolerance = 1e-3
    step_count = 0

    theta = inverse_kinematics_with_optimizer(target_ee) # initial guess: use optimzer result
    # theta = np.array([0.0, 0.0, 0.0])
    _, mean_error = get_cost(theta, target_ee)
    
    while (step_count < max_steps) and (mean_error > tolerance):
        grd = get_gradient(theta, target_ee)
        theta = theta - learning_rate * grd
        _, mean_error = get_cost(theta, target_ee)
        step_count += 1


    return theta
