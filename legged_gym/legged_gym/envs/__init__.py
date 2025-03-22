
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1.h1_teleop_config import H1TeleopCfg, H1TeleopCfgPPO
# 添加G1的配置
from legged_gym.envs.g1.g1_teleop_config import G1TeleopCfg, G1TeleopCfgPPO

from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1:teleop", LeggedRobot, H1TeleopCfg(), H1TeleopCfgPPO())
# 添加G1的注册
task_registry.register("g1:teleop", LeggedRobot, G1TeleopCfg(), G1TeleopCfgPPO())