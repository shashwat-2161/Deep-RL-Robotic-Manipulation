import gymnasium as gym
import panda_gym
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os

# --- CONFIGURATION ---
NUM_CPU = 8  # Use 8 parallel environments (adjust based on your CPU cores)
TOTAL_TIMESTEPS = 1000000
LOG_DIR = "./sac_pp_fast_tensorboard/"
MODEL_DIR = "./models_fast/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- GPU CHECK ---
device_target = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Training on {device_target.upper()} with {NUM_CPU} CPU Workers ---")

if __name__ == '__main__':
    # 1. Create Vectorized Environment (The Speed Boost)
    # This runs 8 independent simulations at the same time.
    # We use 'SubprocVecEnv' to force them onto separate CPU cores.
    vec_env = make_vec_env(
        "PandaPickAndPlace-v3", 
        n_envs=NUM_CPU, 
        vec_env_cls=SubprocVecEnv
    )

    # Define "Policy Keyword Arguments" to increase network size
    policy_kwargs = dict(net_arch=[400, 300])

    # 2. Define Optimized SAC Model
    model = SAC(
        "MultiInputPolicy", 
        vec_env, 
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,                 # <--- INCREASED: Learn more from failures
            goal_selection_strategy="future",
        ),
        policy_kwargs=policy_kwargs,          # <--- INCREASED: Bigger Brain
        verbose=1, 
        buffer_size=100000,     
        batch_size=2048,        
        learning_rate=1e-3, 
        gamma=0.95,
        tau=0.05,
        train_freq=64,          
        gradient_steps=64,      
        learning_starts=1000,   
        device=device_target,
        tensorboard_log=LOG_DIR
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // NUM_CPU, # Adjust freq for parallel envs
        save_path=MODEL_DIR, 
        name_prefix="sac_pp_fast"
    )

    # 4. Train
    print("--- Starting High-Speed Training ---")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, tb_log_name="sac_pp_fast_run")
    print("--- Training Complete ---")

    model.save("sac_pick_place_fast_final")