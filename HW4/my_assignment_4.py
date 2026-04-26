import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import os

LEARN_MODE = False
DEPLOY_MODE = True

# Create RL training environment
env = gym.make("CartPole-v1", render_mode ="human")

# WandB data logging
config = {
"policy_type": "MlpPolicy",
"total_timesteps": 20000,
"env_name": "CartPole-v1",
}

# Learning rate sweep
learning_rates = [0.00003, 0.0003, 0.003, 0.03, 0.3]

# === LEARN===
if LEARN_MODE:
    for lr in learning_rates:

        # Faster learning environment
        train_env = gym.make("CartPole-v1", render_mode=None)

        run = wandb.init(
        project="assignment_04a",
        config={**config, "learning_rate": lr},
        group="LR_sweep", # group the sweep for plotting
        name=f"run_LR_{lr}",
        sync_tensorboard=True, # auto-upload sb3's tensorboard metrics
        monitor_gym=False, # auto-upload the videos of agents playing the game
        save_code=True, # optional
        )

        # Add PPO model using MLP + Initial parameters
        model = PPO("MlpPolicy", env,
        learning_rate=lr, # 0.00003, 0.0003, 0.003, 0.03, 0.3
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"runs/{run.id}")

        # Add WandB logging
        model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
            ) ,
        )
        # Save model and finish
        model.save(f"ppo_cartpole_model_lr_{lr}")
        run.finish()


# === DEPLOY + download video===
if DEPLOY_MODE:
    video_folder = "videos/"
    os.makedirs(video_folder, exist_ok=True)

    for lr in learning_rates:

        # Create fresh env for each LR
        base_env = gym.make("CartPole-v1", render_mode="rgb_array") # "human" for watching, "rgb_array" for recording
        video_env = DummyVecEnv([lambda: base_env]) 

        video_env = VecVideoRecorder(
            video_env, 
            video_folder,
            record_video_trigger=lambda x: x == 0, # Record starting at step 0
            video_length=500,
            name_prefix=f"model_lr_{lr}"
        )

        try:
            # Load the trained model
            loaded_model = PPO.load(f"ppo_cartpole_model_lr_{lr}")

            # Reset the env and run sim
            obs = video_env.reset()
            for _ in range(500):
                action, _states = loaded_model.predict(obs, deterministic=True) # based on observation, model determines next action
                obs, reward, terminated, info = video_env.step(action) # log what happens with that action
                # env.render() # visualize
                
                # exit sim
                if terminated[0]:
                    obs = video_env.reset()

        finally:
            video_env.close()
            print(f"Video for LR {lr} saved in: {video_folder}")