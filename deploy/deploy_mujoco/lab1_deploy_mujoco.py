import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

# mujoco_from_isaaclab_transform = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]
# mujoco_from_isaaclab_transform = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
# isaaclab_from_mujoco_transform = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
# isaaclab_from_mujoco_transform = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
indices_to_delete = [13,14,19,20,21,26,27,28]

# mujoco_from_isaaclab_transform1 =[0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]
def insert_zeros_at_new_indices(arr, target_indices_after_insertion, value=0, axis=0):
    insertion_indices = []
    current_offset = 0
    for target_idx in target_indices_after_insertion:
        original_idx = target_idx - current_offset
        insertion_indices.append(original_idx)
        current_offset += 1

    for idx in sorted(insertion_indices, reverse=True):
        arr = np.insert(arr, idx, value, axis=axis)

    return arr


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    
    # return (target_q - q) * kp + (target_dq - dq) * kd
    return (target_q - q) * kp*0.5 + (target_dq - dq) * kd

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action_mujoco = np.zeros(default_angles.size, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    np.set_printoptions(suppress=True, precision=6)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        d.qpos[7:] = target_dof_pos  # 例如 default_angles
        d.qvel[6:] = np.zeros_like(d.qvel[6:])  # 速度归零
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            print("tau: ", tau)
            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
  
                # create observation
                # base_lin_vel = d.qvel[:3]  # 基础线速度

                
                
                # # print("base_ang_vel: ",base_ang_vel)
                # quat = d.qpos[3:7]  # 四元数
                # ===========================
                # base_lin_vel_world = d.qvel[:3]  # 世界系基础线速度
                # base_ang_vel_world = d.qvel[3:6]  # 世界系基础角速度
                

                # neg_quat = np.zeros(4, dtype=np.float64)
                # mujoco.mju_negQuat(neg_quat, quat)

                # base_lin_vel = np.zeros(3, dtype=np.float64)
                # base_ang_vel = np.zeros(3, dtype=np.float64)
                # mujoco.mju_rotVecQuat(base_lin_vel, base_lin_vel_world, neg_quat)
                # mujoco.mju_rotVecQuat(base_ang_vel, base_ang_vel_world, neg_quat)

                base_ang_vel = d.qvel[3:6]  # 基础角速度
                quat = d.qpos[3:7]  
                gravity_orientation = get_gravity_orientation(quat)  # 投影重力方向


                velocity_commands = cmd * cmd_scale  # 速度命令

                joint_pos = d.qpos[7:]  # 关节位置
                
                joint_vel = d.qvel[6:]  # 关节速度
          
                # print("=======================")
                # print("joint_pos: ", joint_pos)
                # 归一化和缩放
                joint_pos_with_init = joint_pos - default_angles
                

                # # 组装 observation
                # obs[:3] = base_lin_vel  # 基础线速度
                # obs[3:6] = base_ang_vel  # 基础角速度
                # obs[6:9] = gravity_orientation  # 投影重力方向
                # obs[9:12] = velocity_commands  # 速度命令
                # obs[12:35] = action_mujoco[isaaclab_from_mujoco_transform] # 上一时刻的动作
                # # obs[12:35] = action_mujoco# 上一时刻的动作
                # obs[35:58] = joint_pos[isaaclab_from_mujoco_transform]  # 关节位置
                # obs[58:81] = joint_vel[isaaclab_from_mujoco_transform]  # 关节速度


                
                # obs[9:32] = action_mujoco[isaaclab_from_mujoco_transform] # 上一时刻的动作
               
                joint_pos_isaac = np.delete(joint_pos_with_init, indices_to_delete, axis=0)
                joint_vel_isaac = np.delete(joint_vel, indices_to_delete, axis=0)
                action_isaac= np.delete(action_mujoco, indices_to_delete, axis=0)

                # 组装 observation
                obs[:3] = base_ang_vel  # 基础角速度
                obs[3:6] = gravity_orientation  # 投影重力方向
                obs[6:9] = velocity_commands  # 速度命令
                obs[9:9+1*num_actions] = action_isaac
                obs[9+1*num_actions:9+2*num_actions] = joint_pos_isaac  # 关节位置
                obs[9+2*num_actions:9+3*num_actions] = joint_vel_isaac  # 关节速度
                
                # print("obs: ", obs.shape)  # 上一时刻的动作
                
                # print("Last  obser: ", obs)  # 上一时刻的动作
        
                if counter % ( 100) == 0:
                    # print("obs: ", obs)  # 上一时刻的动作
                    # print("action_scale: ", action_scale)
                    # break
                    pass

                # 转换为张量
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)

                # policy inference
                action_isaaclab = policy(obs_tensor).detach().numpy().squeeze()
                # print("Action: ", action_isaaclab)  # 上一时刻的动作
                # print("action_isaaclab: ", action_isaaclab)
                action_mujoco = insert_zeros_at_new_indices(action_isaaclab, indices_to_delete, value=0, axis=0)
                # print("action_mujoco: ", action_mujoco)
                # print("action_mujoco: ", action_mujoco)
                # transform action_mujoco to target_dof_pos
                # print("action_scale: ", action_scale)
                target_dof_pos = action_mujoco * action_scale + default_angles
                # target_dof_pos[-1]=1
                # print("target_dof_pos: ", target_dof_pos)
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
