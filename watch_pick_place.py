import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC

# 1. Load the Environment with Visuals
env = gym.make("PandaPickAndPlace-v3", render_mode="human")

# 2. Load the Best Model so far
# Note: Check your folder. It might be 'sac_pp_fast_final.zip' or a checkpoint.
# If training is still running, try loading the latest checkpoint from ./models_fast/
try:
    # Try to find the latest checkpoint if 'final' doesn't exist yet
    import glob
    import os
    list_of_files = glob.glob('./models_fast/*.zip') 
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading latest checkpoint: {latest_file}")
    model = SAC.load(latest_file, env=env)
except Exception as e:
    print("Could not auto-load. Trying default name...")
    model = SAC.load("sac_pick_place_fast_final", env=env)

# 3. Watch 10 Episodes
print("--- Watching Agent ---")
obs, info = env.reset()

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Resetting Environment")

env.close()