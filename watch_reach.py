import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import time

# 1. Load the Environment with Visuals
# render_mode="human" pops up the 3D window
env = gym.make("PandaReach-v3", render_mode="human")

# 2. Load your trained model
try:
    model = SAC.load("sac_reach_v1", env=env)
    print("Loaded SAC model.")
except:
    print("Model not found! Train it first.")
    exit()

# 3. Run the simulation loop
obs, info = env.reset()

for i in range(1000):
    # Predict the action (deterministic=True removes randomness for evaluation)
    action, _ = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Slow down slightly so you can see the movement
    time.sleep(0.05)
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Target Reached! Resetting...")

env.close()