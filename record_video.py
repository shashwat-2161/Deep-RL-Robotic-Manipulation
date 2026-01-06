import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo

# 1. Setup Environment with Video Recorder
# It will save the video to the "./videos" folder
env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)

# 2. Load Model
# Use your latest model file
model = SAC.load("sac_pick_place_fast_final", env=env)

# 3. Record 5 Episodes
print("--- Recording 5 Episodes ---")
obs, info = env.reset()
for _ in range(250): # 50 steps * 5 episodes
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("--- Video Saved in ./videos folder ---")