##############
# 姓名：于佳琨
# 学号：2100013119
##############
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = None
    joint_parent = None
    joint_offset = None

    with open(bvh_file_path,'r') as f:
        lines = f.readlines()
        cnt = 0
        parent = [0]
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                joint_name = [lines[i][lines[i].find("ROOT")+4:].strip()]
                joint_parent = [-1]
                i += 2
                data = [float(x) for x in lines[i]
                        [lines[i].find("OFFSET")+6:].split()]
                joint_offset = [np.array(data).reshape(1,-1)]
                break
        for line in lines[i+2:]:
            if "JOINT" in line:
                joint_name.append(line[line.find("JOINT")+5:].strip())
                joint_parent.append(parent[-1])
                cnt += 1
            elif "OFFSET" in line:
                data = [float(x) for x in line[line.find("OFFSET")+6:].split()]
                joint_offset.append(np.array(data).reshape(1,-1))
            elif "End" in line:
                joint_name.append(joint_name[parent[-1]] + "_end")
                joint_parent.append(parent[-1])
                cnt += 1
            elif "{" in line:
                parent.append(cnt)
            elif "}" in line:
                parent.pop()
        joint_offset=np.concatenate(joint_offset,axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None
    
    num = len(joint_name)
    motion = motion_data[frame_id] # 一行的运动数据    
    joint_positions = [motion[0:3]]
    joint_orientations = [R.from_euler('XYZ', motion[3:6], degrees=True)]
    motion = motion[3:]
    cnt = 1 # frame数据中到第几个rotation
    for i in range(1,num): # 提取position和orientations
        if "end" in joint_name[i]:
            joint_orientations.append(joint_orientations[joint_parent[i]])
            joint_positions.append(joint_positions[joint_parent[i]]+
                                   joint_orientations[joint_parent[i]].apply(joint_offset[i],inverse=False))
        else:
            local_rotation = R.from_euler('XYZ', motion[3*cnt:3*cnt+3], degrees=True) # 转化为欧拉角
            cnt += 1
            joint_orientations.append(joint_orientations[joint_parent[i]]*local_rotation)
            joint_positions.append(joint_positions[joint_parent[i]]+
                                   joint_orientations[joint_parent[i]].apply(joint_offset[i],inverse=False))
    for j in range(len(joint_orientations)):
        joint_orientations[j]=joint_orientations[j].as_quat() # 转化为四元数
    joint_positions = np.concatenate(joint_positions,axis=0).reshape(-1,3)
    joint_orientations = np.concatenate(joint_orientations,axis=0).reshape(-1,4)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    
    A_motion_data = load_motion_data(A_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    # A转到T的旋转矩阵
    A_to_T_orientations = [[0, 0, 0]]*len(T_joint_name) # A_to_T[i]对应的是T[i]
    # 计算A_to_T
    for i in range(len(T_joint_name)):
        if "end" in T_joint_name[i]:
            A_to_T_orientations[i] = A_to_T_orientations[T_joint_parent[i]]           
        else:
            A_offset = A_joint_offset[A_joint_name.index(T_joint_name[i])]
            T_offset = T_joint_offset[i]
            A_to_T,_ = R.align_vectors([T_offset], [A_offset])
            A_to_T = A_to_T.as_euler('XYZ', degrees=True)
            A_to_T_orientations[T_joint_parent[i]] = A_to_T
    # 重定位
    motion_data = np.zeros(np.shape(A_motion_data))
    motion_data[:, 0:6] = A_motion_data[:, 0:6]
    Fr = np.shape(motion_data)[0] # 帧数
    num = len(A_joint_name)
    for i in range(Fr):
        cnt = 1
        for j in range(1, num):
            if "end" not in A_joint_name[j]:
                A_local_rotation = R.from_euler('XYZ', A_motion_data[i][3*cnt+3: 3*cnt+6], degrees=True).as_matrix() # 转化为欧拉角           
                cnt += 1
                parent_index = T_joint_name.index(A_joint_name[A_joint_parent[j]])
                joint_index = T_joint_name.index(A_joint_name[j])
                parent_A_to_T = R.from_euler('XYZ', A_to_T_orientations[parent_index], degrees=True).as_matrix()
                joint_T_to_A = np.transpose(R.from_euler('XYZ', A_to_T_orientations[joint_index], degrees=True).as_matrix())
                T_local_rotation = parent_A_to_T@A_local_rotation@joint_T_to_A
                joint_index_no = joint_index
                for k in range(0, joint_index):
                    if "end" in T_joint_name[k]:
                        joint_index_no -= 1
                motion_data[i][3+3*joint_index_no:6+3 * joint_index_no] = R.from_matrix(T_local_rotation).as_euler('XYZ', degrees=True)
    return motion_data