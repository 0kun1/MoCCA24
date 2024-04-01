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
    joint_orientations = input_joint_orientations
    joint_positions = input_joint_positions
    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets

    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #
    if len(path2) == 1 and path2[0] != 0:
        path2 = []

    # 每个joint的local rotation，用四元数表示
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()


    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    rotation_path = np.empty((len(path),), dtype=object)
    position_path = np.empty((len(path), 3))
    orientation_path = np.empty((len(path),), dtype=object)
    offset_path = np.empty((len(path), 3))

    # 对chain进行初始化
    if len(path2) > 1:
        orientation_path[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        orientation_path[0] = R.from_quat(joint_orientations[path[0]])

    position_path[0] = joint_positions[path[0]]
    rotation_path[0] = orientation_path[0]
    offset_path[0] = np.array([0.,0.,0.])

    for i in range(1, len(path)):
        index = path[i]
        position_path[i] = joint_positions[index]
        if index in path2:
            # essential
            orientation_path[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_path[i] = R.from_quat(joint_rotations[path[i]]).inv()
            offset_path[i] = -joint_offsets[path[i - 1]]
            # essential
        else:
            orientation_path[i] = R.from_quat(joint_orientations[index])
            rotation_path[i] = R.from_quat(joint_rotations[index])
            offset_path[i] = joint_offsets[index]


    # CCD IK
    times = 10
    distance = np.sqrt(np.sum(np.square(position_path[-1] - target_pose)))
    end = False
    while times > 0 and distance > 0.001 and not end:
        times -= 1
        # 先动手
        for i in range(len(path) - 2, -1, -1):
        # 先动腰
        # for i in range(1, len(path) - 1):
            if joint_parent[path[i]] == -1:
                continue
            cur_pos = position_path[i]
            # 计算旋转的轴角表示
            c2t = target_pose - cur_pos
            c2e = position_path[-1] - cur_pos
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            # 由于float的精度问题，cos可能cos(theta)可能大于1.
            cos = min(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)), 1.0)
            theta = np.arccos(cos)
            # 防止quat为0？
            if theta < 0.0001:
                continue
            delta_rotation = R.from_rotvec(theta * axis)
            # 更新当前的local rotation 和子关节的position, orientation
            orientation_path[i] = delta_rotation * orientation_path[i]
            rotation_path[i] = orientation_path[i - 1].inv() * orientation_path[i]
            for j in range(i + 1, len(path)):
                orientation_path[j] = orientation_path[j - 1] * rotation_path[j]
                position_path[j] = np.dot(orientation_path[j - 1].as_matrix(), offset_path[j]) + position_path[j - 1]
            distance = np.sqrt(np.sum(np.square(position_path[-1] - target_pose)))


    # 把计算之后的IK写回joint_rotation
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_path[i]
        if index in path2:
            joint_rotations[index] = rotation_path[i].inv().as_quat()
        else:
            joint_rotations[index] = rotation_path[i].as_quat()

    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_path[0]).as_quat()

    # 如果rootjoint在IK链之中，那么需要更新rootjoint的信息
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_path[root_index].as_quat()
            joint_positions[0] = position_path[root_index]


    # 最后计算一遍FK，得到更新后的position和orientation
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入左手相对于根节点前进方向的xz偏移，以及目标高度，lShoulder到lWrist为可控部分，其余部分与bvh一致
    注意part1中只要求了目标关节到指定位置，在part2中我们还对目标关节的旋转有所要求
    """
    end = meta_data.joint_name.index(meta_data.end_joint)
    root = meta_data.joint_name.index(meta_data.root_joint)
    parent = meta_data.joint_parent[end] #lwist
    # meta_data.end_joint = meta_data.joint_name[parent]

    target_pose_end = np.copy(joint_positions[end])
    target_pose_root = np.copy(joint_positions[root])
    # target_pose_parent = np.copy(joint_positions[parent])
    # offset = meta_data.joint_initial_position[end] - meta_data.joint_initial_position[parent]
    
    target_pose_end[0] = relative_x + target_pose_root[0]
    target_pose_end[2] = relative_z + target_pose_root[2]
    target_pose_end[1] = target_height
    # target_pose_end[1] = target_height - np.linalg.norm(offset)

    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose_end)
    # joint_positions[parent] = target_pose_end
    # joint_positions[end] = target_pose_end + [0,np.linalg.norm(offset),0]
    joint_positions[end] = target_pose_end
    # euler_angles = [0, 90, 0]  # XYZ顺序，y轴旋转90度
    # rotation = R.from_euler('XYZ', euler_angles, degrees=True)
    # quaternion = rotation.as_quat()

    # joint_orientations[parent] = quaternion
    # joint_orientations[end] = quaternion
    
    return joint_positions, joint_orientations
