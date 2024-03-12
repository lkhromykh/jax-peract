from src.environment.gcenv import GoalConditionedEnv
try:
    from src.environment.rlbench_env import RLBenchEnv
except:
    RLBenchEnv = None
from src.environment.ur5_env import UREnv
