##############
# 姓名：于佳琨
# 学号：2100013119
##############
import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, input_joint_positions, input_joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """    
    joint_positions = input_joint_positions
    joint_orientations = input_joint_orientations
    path, _, path1, path2 = meta_data.get_path_from_root_to_end()
    path1 = list(reversed(path1))
    init_offset = [[0, 0, 0]]*len(meta_data.joint_name)
    joint_length = [0]*len(meta_data.joint_name)
    path_length = [0]*len(path)
    local_rotation=[[0,0,0,1]]*len(meta_data.joint_name)
    for i in range(1, len(meta_data.joint_name)):
        p = meta_data.joint_parent[i]
        init_offset[i] = meta_data.joint_initial_position[i] - meta_data.joint_initial_position[p]
        joint_length[i] = init_offset[i].dot(init_offset[i])**0.5
        parent_rotation_T = np.transpose(R.from_quat(joint_orientations[p]).as_matrix())
        global_rotation = R.from_quat(joint_orientations[i]).as_matrix()
        local_rotation[i] = R.from_matrix(parent_rotation_T @ global_rotation).as_quat()
    if len(path2) > 1:
        for i in range(1, len(path2)):
            path_length[i] = joint_length[path2[i-1]]
    for j in range(len(path1)):
        path_length[j+len(path2)] = joint_length[path1[j]]
    # hyperparameters
    max_iteration = 32
    error_threshold = 0.001
    num_joints = len(path)
    # FABRIK
    backward_positions = [np.array([0, 0, 0])]*num_joints
    forward_positions = [np.array([0, 0, 0])]*num_joints
    root_position=np.copy(joint_positions[path[0]])
    for m in range(max_iteration):
        # backward update
        next_position = np.copy(target_pose)
        backward_positions[-1] = np.copy(target_pose)
        for i in range(num_joints-2, -1, -1):
            direction = joint_positions[path[i]]-next_position
            direction = direction/(direction.dot(direction)**0.5)
            next_position += direction*path_length[i+1]
            backward_positions[i] = np.copy(next_position)
        # forward update
        now_position = np.copy(root_position)
        forward_positions[0] = np.copy(root_position)
        for i in range(num_joints-1):
            direction = backward_positions[i+1]-now_position
            direction = direction/(direction.dot(direction)**0.5)
            now_position += direction*path_length[i+1]
            forward_positions[i+1] = np.copy(now_position)
        for i in range(num_joints):
            joint_positions[path[i]] = np.copy(forward_positions[i])
        error = joint_positions[path[-1]]-target_pose
        error = error.dot(error)**0.5
        if error < error_threshold or m > max_iteration:
            break
    # Compute joint rotation by position
    # path2
    start_offset = forward_positions[1]-forward_positions[0]
    joint_orientations[path[0]] = my_rotate(start_offset,init_offset[path[1]])
    if len(path2) > 1:
        for i in range(len(path2)-1):
            new_offset = forward_positions[i]-forward_positions[i+1]
            joint_orientations[path[i+1]] = my_rotate(new_offset, init_offset[path[i]])
    # path1
    for i in range(len(path2), len(path)-1):
        new_offset = forward_positions[i+1]-forward_positions[i]
        joint_orientations[path[i]] = my_rotate(new_offset, init_offset[path[i+1]])
    # forward IK
    for i in range(1, len(meta_data.joint_name)):
        p = meta_data.joint_parent[i]
        if not i in path:
            joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(local_rotation[i])).as_quat()
        joint_positions[i] = joint_positions[p] + R.from_quat(joint_orientations[p]).apply(init_offset[i], inverse=False)
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    end = meta_data.joint_name.index(meta_data.end_joint)
    root = meta_data.joint_name.index(meta_data.root_joint)
    parent = meta_data.joint_parent[end] #lwist
    meta_data.end_joint = meta_data.joint_name[parent]

    target_pose_end = np.copy(joint_positions[end])
    target_pose_root = np.copy(joint_positions[root])
    target_pose_parent = np.copy(joint_positions[parent])

    offset = meta_data.joint_initial_position[end] - meta_data.joint_initial_position[parent]

    target_pose_end[0] = relative_x + target_pose_root[0]
    target_pose_end[2] = relative_z + target_pose_root[2]
    target_pose_end[1] = target_height - np.linalg.norm(offset)

    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose_end)
    joint_positions[parent] = target_pose_end
    joint_positions[end] = target_pose_end + [0,np.linalg.norm(offset),0]
    
    euler_angles = [0, 90, 0]  # XYZ顺序，y轴旋转90度
    rotation = R.from_euler('XYZ', euler_angles, degrees=True)
    quaternion = rotation.as_quat()

    joint_orientations[parent] = quaternion
    joint_orientations[end] = quaternion
    
    return joint_positions, joint_orientations
    
def my_rotate(target, src):
    a = np.copy(target)
    b = np.copy(src)
    a /= a.dot(a)**0.5
    b /= b.dot(b)**0.5
    mid = a+b
    q = [0, 0, 0, 0]
    if mid.dot(mid) == 0:
        q[-1] = 1
    else:
        mid /= mid.dot(mid)**0.5
        q[-1] = mid.dot(b)
        q[:-1] = np.cross(b, mid)
    return q