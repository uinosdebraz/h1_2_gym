
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class H1_2SquatEnv(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,2,6,8]]), dim=1)
    
    # def _reward_squat_height(self):
    #     pelvis_z = self.root_states[:, 2]
    #     t = self.episode_length_buf * self.dt
    #     target = 1.05 + 0.05 * torch.sin(2 * np.pi * 0.5 * t)
        
    #     # Debug print
    #     if self.episode_length_buf[0] % 100 == 0:  # 每100步打印一次
    #         print(f"[squat debug] step={self.episode_length_buf[0]}  z={pelvis_z[0].item():.3f}  target={target[0].item():.3f}")
        
    #     return -torch.abs(pelvis_z - target)

    def _reward_squat_height(self):
        pelvis_z = self.root_states[:, 2]
        t = self.episode_length_buf * self.dt
        target = 1.05 + 0.03 * torch.sin(2 * np.pi * 0.25 * t)
        error = torch.abs(pelvis_z - target)
        reward = 1.0 - torch.clamp(error / 0.1, 0, 1)

        if self.episode_length_buf[0] % 200 == 0:
            print(f"[REWARD] squat_height: {reward[0]:.3f}  z={pelvis_z[0]:.3f}  target={target[0]:.3f}")

        return reward
    
    # def _reward_upright(self):
    #     # Encourage Z-axis of gravity vector to align with world Z (up)
    #     val = torch.clamp(self.projected_gravity[:, 2], 0, 1)
    #     if self.episode_length_buf[0] % 200 == 0:
    #         print(f"[REWARD] upright: {val[0]:.3f}")
    #     return val

    def _reward_upright(self):
        # 机器人身体z轴和世界z轴对齐
        up_proj = self.base_quat * torch.tensor([0, 0, 1, 0], device=self.device)
        return up_proj[:, 2]  # 取 z 分量

    def _reward_contact_phase(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            phase_i = self.leg_phase[:, i]
            is_stance = phase_i < 0.5
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            reward += (is_stance & contact).float()
        return reward
    
    def _reward_action_smooth(self):
        delta = self.actions - self.last_actions
        penalty = torch.sum(delta ** 2, dim=1)
        return penalty


    def pre_physics_step(self, actions):
        self.last_actions[:] = self.actions.clone()
        super().pre_physics_step(actions)
        # # perturb the robot with a random push
        # if self.cfg.domain_rand.push_robots:
        #     if torch.rand(1).item() < self.dt / self.cfg.domain_rand.push_interval_s:
        #         direction = torch.randn((self.num_envs, 2), device=self.device)
        #         direction = direction / torch.norm(direction, dim=1, keepdim=True)
        #         impulse = direction * self.cfg.domain_rand.max_push_vel_xy
        #         self.root_states[:, 7:9] += impulse  # apply velocity push

