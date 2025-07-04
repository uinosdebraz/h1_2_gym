from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1_2SquatFlatConfig(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]
        default_joint_angles = {
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'torso_joint': 0,
            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,
            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
        }

    class env(LeggedRobotCfg.env):
        num_observations       = 51  # height_err, pelvis_z, target_z, feet_dist, yaw/roll/pitch + 6 + 12 + 12 + 1 + 1
        num_privileged_obs     = 54
        num_actions            = 12

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,
        }
        damping = {
            'hip_yaw_joint': 2.5,
            'hip_roll_joint': 2.5,
            'hip_pitch_joint': 2.5,
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,
        }
        action_scale = 0.25
        decimation = 8

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_12dof.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        armature = 1e-3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0

        class scales(LeggedRobotCfg.rewards.scales):
            # —— 原先保留的几项 ——
            dof_acc           = -2.5e-7
            dof_vel           = -1e-3
            feet_swing_height = -20.0

            # —— 核心目标奖励 —— 
            squat_height      = 20.0    # 追踪目标下蹲
            feet_distance     = 0.5     # 保持脚距
            orientation       = 1.0     # 身体竖直度
            default_joint_pos = 0.5     # 避免腿锁死伸直
            feet_contact      = 5.0     # 强烈鼓励双脚同时着地
            leg_symmetry      = 3.0     # 奖励左右膝角一致

            # —— 辅助平滑惩罚 —— 
            action_smooth     = 0.002   # 允许合理的关节摆动

            # —— 其余全禁用 —— 
            ang_vel_xy        = 0.0
            action_rate       = 0.0
            collision         = 0.0
            contact_no_vel    = 0.0
            feet_air_time     = 0.0
            hip_pos           = 0.0
            upright           = 0.0
            contact           = 0.0
            contact_phase     = 0.0
            alive             = 0.0
            tracking_lin_vel  = 0.0
            tracking_ang_vel  = 0.0
            lin_vel_z         = 0.0
            root_vel          = 0.0    # we use root_pos, 不再用 root_vel



class H1_2RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'h1_2_squat'
