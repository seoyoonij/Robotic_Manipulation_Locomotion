import numpy as np
import math

# 2a
def rotate2D (theta, p_point): # angle in radians
   
   # input validation
    if not isinstance(p_point, np.ndarray):
        raise TypeError ("p_point must be a numpy array")
    if not p_point.size == 2:
        raise ValueError("p_point must have 2 elements")
    
    Rot = np.array ([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

    # [q_point] = [Rot][p_point]
    q_point = np.matmul(Rot, p_point)

    return q_point

# 2b
def rotate3D (theta, axis_of_rotation, p_point):

     # input validation
    if not isinstance(p_point, np.ndarray):
        raise TypeError ("p_point must be a numpy array")
    if not p_point.size == 3:
        raise ValueError("p_point must have 3 elements")
    if not isinstance(axis_of_rotation, str):
        raise TypeError ("axis_of_rotation must be a string ('x', 'y', or 'z')")

    if axis_of_rotation == 'x':
        Rot = np.array ([    
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]
        ])
    elif axis_of_rotation == 'y':
        Rot = np.array ([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta),0 , math.cos(theta)]
        ])
    elif axis_of_rotation == 'z':
        Rot = np.array ([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta),0],
            [0, 0, 1]
        ])
    else: 
        raise ValueError("Invalid axis of rotation (x,y,z)")

    # [q_point] = [Rot][p_point]
    q_point = np.matmul(Rot, p_point)

    return q_point

# 2c
def rotate3D_many_times(rotation_list, p_point):
    
    # initialize in case rotation_list is empty
    q_point = p_point

    for rotation in rotation_list:
        q_point = rotate3D(rotation[0], rotation[1], q_point)
    return q_point
