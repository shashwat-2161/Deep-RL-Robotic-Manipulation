import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import os

# 1. Configuration
log_dir = "./sac_reach_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# 2. Create Environment
env = gym.make("PandaReach-v3")

# 3. Define Model with Logging
# tensorboard_log=log_dir enables the graphs
model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

# 4. Train
print("--- Starting Training with Logging ---")
model.learn(total_timesteps=50000, tb_log_name="sac_reach_run")
print("--- Training Complete ---")

# 5. Save
model.save("sac_reach_v1")
print("Model saved as 'sac_reach_v1.zip'")