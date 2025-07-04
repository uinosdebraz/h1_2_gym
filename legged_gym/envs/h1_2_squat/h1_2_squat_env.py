
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np

class H1_2SquatEnv(LeggedRobot):

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        scales = self.cfg.noise.noise_scales
        lvl = self.cfg.noise.noise_level
        noise_vec[:3] = scales.ang_vel * lvl * self.obs_scales.ang_vel
        noise_vec[3:6] = scales.gravity * lvl
        start = 6
        noise_vec[start:start+self.num_actions] = scales.dof_pos * lvl * self.obs_scales.dof_pos
        noise_vec[start+self.num_actions:start+2*self.num_actions] = scales.dof_vel * lvl * self.obs_scales.dof_vel
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rb)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _sample_root_height_params(self):
        self.rh_base  = 1.00 * torch.ones(self.num_envs, device=self.device)
        self.rh_amp   = 0.15 * torch.ones(self.num_envs, device=self.device)
        self.rh_freq  = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.rh_phase = 2*np.pi * torch.rand(self.num_envs, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._sample_root_height_params()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos   = self.feet_state[:, :, :3]
        self.feet_vel   = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()
        period = 0.8; offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left  = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1),
                                     self.phase_right.unsqueeze(1)], dim=-1)
        return super()._post_physics_step_callback()

    def compute_observations(self):
        sin_p = torch.sin(2*np.pi*self.phase).unsqueeze(1)
        cos_p = torch.cos(2*np.pi*self.phase).unsqueeze(1)

        # pelvis height & target traj
        pelvis_z = self.root_states[:,2].unsqueeze(1)
        t = self.episode_length_buf * self.dt
        target_z = (self.rh_base + self.rh_amp*torch.sin(2*np.pi*self.rh_freq*t + self.rh_phase)).unsqueeze(1)
        height_err = pelvis_z - target_z

        # euler from quat
        qx,qy,qz,qw = (self.base_quat[:,0], self.base_quat[:,1],
                       self.base_quat[:,2], self.base_quat[:,3])
        ysqr = qy*qy
        t0 = 2*(qw*qx + qy*qz); t1 = 1 - 2*(qx*qx + ysqr)
        roll  = torch.atan2(t0, t1).unsqueeze(1)
        t2 = 2*(qw*qy - qz*qx); t2 = torch.clamp(t2, -1, 1)
        pitch = torch.asin(t2).unsqueeze(1)
        t3 = 2*(qw*qz + qx*qy); t4 = 1 - 2*(ysqr + qz*qz)
        yaw   = torch.atan2(t3, t4).unsqueeze(1)

        # feet spacing
        pos_xy   = self.feet_pos[:,:, :2]
        feet_dist = torch.norm(pos_xy[:,0]-pos_xy[:,1], dim=1, keepdim=True)

        self.commands.zero_()

        self.obs_buf = torch.cat([
            height_err, pelvis_z, target_z, feet_dist,
            yaw, roll, pitch,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos)*self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_p, cos_p
        ], dim=-1)

        self.privileged_obs_buf = torch.cat([
            height_err, pelvis_z, target_z, feet_dist,
            yaw, roll, pitch,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos)*self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_p, cos_p
        ], dim=-1)

        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec

    # def _reward_contact(self):
    #     res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(self.feet_num):
    #         is_stance = self.leg_phase[:, i] < 0.55
    #         contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    #         res += ~(contact ^ is_stance)
    #     return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    # def _reward_hip_pos(self):
    #     return torch.sum(torch.square(self.dof_pos[:, [0,2,6,8]]), dim=1)

    def _reward_squat_height(self):
        pelvis_z = self.root_states[:, 2]
        t = self.episode_length_buf * self.dt
        target = self.rh_base + self.rh_amp * torch.sin(2 * np.pi * self.rh_freq * t + self.rh_phase)
        diff = pelvis_z - target
        reward = torch.exp(-50.0 * diff**2)
        if self.episode_length_buf[0] % 200 == 0:
            print(f"[REWARD] squat_height: {reward[0]:.3f}  z={pelvis_z[0]:.3f}  target={target[0]:.3f}")
        return reward

    # def _reward_upright(self):
    #     up_proj = self.base_quat * torch.tensor([0, 0, 1, 0], device=self.device)
    #     return up_proj[:, 2]

    # def _reward_contact_phase(self):
    #     reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(self.feet_num):
    #         phase_i = self.leg_phase[:, i]
    #         is_stance = phase_i < 0.5
    #         contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
    #         reward += (is_stance & contact).float()
    #     return reward

    def _reward_feet_distance(self):
        """
        Reward feet spacing around a target distance to avoid scissoring.
        """
        pos_xy = self.feet_pos[:, :, :2]
        vec = pos_xy[:, 0] - pos_xy[:, 1]
        dist = torch.norm(vec, dim=1)
        target_dist = 0.2
        return torch.exp(-10.0 * (dist - target_dist)**2)

    def _reward_action_smooth(self):
        delta = self.actions - self.last_actions
        return -torch.sum(delta ** 2, dim=1)
    
    def _reward_default_joint_pos(self):
        # 惩罚“腿完全伸直”倾向，鼓励髋膝有轻微弯曲
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll  = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-100.0 * yaw_roll) - 0.01 * torch.norm(joint_diff, dim=1)

    # def _reward_upper_body_pos(self):
    #     # 奖励上半身保持静止，不用上肢晃动来平衡
    #     torso_idx = 12
    #     diff = self.dof_pos - self.default_dof_pos
    #     ub = diff[:, torso_idx:]
    #     ub_err = torch.mean(torch.abs(ub), dim=1)
    #     return torch.exp(-4.0 * ub_err)

    # def _reward_root_vel(self):
    #     # root 在水平面上的速度平方和
    #     return torch.sum(self.base_lin_vel[:, :2]**2, dim=1)

    # def _reward_root_pos(self):
    #     # 强烈惩罚根坐标在 XY 平面上的任何偏移
    #     pos_xy = self.root_states[:, :2]
    #     dist2  = torch.sum(pos_xy**2, dim=1)
    #     return -dist2

    def _reward_orientation(self):
        """
        Reward torso being upright via gravity projection.
        """
        return -self.projected_gravity[:, 2]
    
        
    def _reward_feet_contact(self):
        # 奖励双脚同时着地，避免单腿悬空倾斜。
        # contact_forces[...,2] 是法向力，>1N 表示着地
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # contacts.shape = [envs, feet_num]; both feet 都 True 时返回 1.0
        return contacts.all(dim=1).float()

    def _reward_leg_symmetry(self):
        # 奖励左右膝关节对称弯曲，避免单腿承重。
        # dof_pos 的 index: 3=left_knee, 9=right_knee
        l_knee = self.dof_pos[:, 3]
        r_knee = self.dof_pos[:, 9]
        diff   = torch.abs(l_knee - r_knee)
        return torch.exp(-50.0 * diff)


    def pre_physics_step(self, actions):
        self.last_actions[:] = self.actions.clone()
        super().pre_physics_step(actions)
        self.commands.zero_()
