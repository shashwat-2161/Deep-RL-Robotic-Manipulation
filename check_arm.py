import gymnasium as gym
import panda_gym
import time

# 1. Create the environment
# 'render_mode="human"' opens the 3D window
env = gym.make("PandaReach-v3", render_mode="human")

observation, info = env.reset()

print("--- 3D Simulation Started ---")
print("Action Space:", env.action_space) 
# Expect: Box(-1.0, 1.0, (3,), float32) -> X, Y, Z velocities

for _ in range(100):
    # 2. Take a random action (Move X, Y, Z randomly)
    action = env.action_space.sample() 
    
    # 3. Step
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Slow down so you can see it
    time.sleep(0.05)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("--- Simulation Closed ---")