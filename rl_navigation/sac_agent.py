import os
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from rl_navigation.env import MobileRobotEnv

def main():
    
    env = MobileRobotEnv()
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,  
        buffer_size=50000,     
        batch_size=64,         
        ent_coef='auto',       
        train_freq=1,          
        gradient_steps=1,      
        learning_starts=100,   
        tau=0.005,             
        gamma=0.98,            
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500, 
        save_path=os.path.expanduser("~/ros2_ws/src/rl_navigation/rl_navigation/models/checkpoints/"),
        name_prefix="sac_mobile_robot"
    )

    print("Training model...")
    
    model.learn(total_timesteps=10000, callback=checkpoint_callback)

    model_dir = os.path.expanduser("~/ros2_ws/src/rl_navigation/rl_navigation/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sac_mobile_robot")
    model.save(model_path)

    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()