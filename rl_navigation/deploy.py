import os
import time
import gym
from stable_baselines3 import SAC
from rl_navigation.env import MobileRobotEnv

def main():
    print("Initializing environment...")
    env = MobileRobotEnv()

    model_dir = os.path.expanduser("~/ros2_ws/src/rl_navigation/rl_navigation/models")
    model_path = os.path.join(model_dir, "sac_mobile_robot")

    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)
    print("Model loaded successfully!")

    print("Starting deployment loop...")
    total_reward = 0
    success_count = 0
    episodes = 5  
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nStarting Episode {episode+1}/{episodes}")
        start_time = time.time()
        
        while not done:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            episode_reward += rewards
            
            if step % 10 == 0:  
                print(f"Step: {step}, Action: [{action[0]:.2f}, {action[1]:.2f}], Reward: {rewards:.2f}")
        
        elapsed = time.time() - start_time
        total_reward += episode_reward
        
        
        if 'termination_reason' in info and info['termination_reason'] == 'goal_reached':
            success_count += 1
            print(f"Episode {episode+1} finished successfully! Reward: {episode_reward:.2f}, Time: {elapsed:.1f}s")
        else:
            print(f"Episode {episode+1} failed. Reason: {info.get('termination_reason', 'unknown')}")
            print(f"Reward: {episode_reward:.2f}, Time: {elapsed:.1f}s")
    
    
    print("\n--- Deployment Summary ---")
    print(f"Total episodes: {episodes}")
    print(f"Success rate: {success_count/episodes*100:.1f}%")
    print(f"Average reward: {total_reward/episodes:.2f}")

if __name__ == "__main__":
    main()