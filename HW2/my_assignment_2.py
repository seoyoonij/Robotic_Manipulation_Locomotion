import numpy as np
import math

def cos(theta):
    return math.cos(theta)
def sin(theta):
    return math.sin(theta)

# 2a
## base operation
def rotation_x (theta):
    return np.array(
        [
             [1, 0, 0, 0],
            [0, cos(theta), -sin(theta), 0],
            [0, sin(theta), cos(theta), 0],
            [0, 0, 0, 1]
        ]
    )
def rotation_y (theta):
    return np.array(
        [
            [cos(theta), 0, sin(theta), 0],
            [0, 1, 0, 0],
            [-sin(theta), 0, cos(theta), 0],
            [0, 0, 0, 1]
        ]
    )
def rotation_z (theta):
    return np.array(
        [
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta), cos(theta),0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    )
def translation (x,y,z):
    return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

## approach: T_i,i+1 = T_frame_i @ R_joint_i+1

def get_T01 (theta_1):
    # wrt to Frame 0, offset Frame 1. then joint is theta_1 about z1
    frame_1 = np.identity(4)
    return frame_1 @ rotation_z(theta_1)

def get_T12 (theta_2):
    # wrt to Frame 1, offset Frame 2. then joint is theta_2 about z2
    frame_2 = translation(0,0.3,0) @ rotation_x(math.pi/2) 
    return  frame_2 @ rotation_z(theta_2)

def get_T23 (theta_3):
    # wrt to Frame 2, offset Frame 3. then joint is theta_3 about z3
    frame_3 = translation(0.4, 0, 0)
    return frame_3 @ rotation_z(theta_3)

def get_T34 ():
    # wrt to Frame 3, offset Frame 4.
    frame_4 = translation(0, 0.3, 0) @ rotation_x(-math.pi/2) 
    return frame_4

def get_FK (theta_1, theta_2, theta_3):
    T04 = get_T01(theta_1) @ get_T12(theta_2) @ get_T23(theta_3) @ get_T34()
    return T04

# 2b.
def ee_in_collision (lst_angles, p_point, tolerance):
    # extract EE position: T04's row 012 col 3
    pos_EE = (get_FK(lst_angles[0], lst_angles[1], lst_angles[2]))[:3,3] 
    distance = math.sqrt((pos_EE[0]-p_point[0])**2 + (pos_EE[1]-p_point[1])**2 + (pos_EE[2]-p_point[2])**2)
    if distance < tolerance:
        return True
    return False

# 2c.
def path_in_collision(path, object_list):
    # path: list of ( (theta_1, _2, _3), (), ... )
    # object_list: list of ( ([p_sx, p_sy, p_sz], radius) )
    
    for lst_theta in path:
        for sphere in object_list:
            is_collide = ee_in_collision (lst_theta, sphere[0], sphere[1])
            if is_collide:
                return True
    return False