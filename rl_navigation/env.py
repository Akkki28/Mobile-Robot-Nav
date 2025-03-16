import gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
import math
import threading

class MobileRobotEnv(gym.Env):
    def __init__(self):
        super(MobileRobotEnv, self).__init__()

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('mobile_robot_env')
        
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.action_space = gym.spaces.Box(
            low=np.array([-0.3, -0.8]),  
            high=np.array([0.3, 0.8]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=np.zeros(18),
            high=np.ones(18) * 10.0,
            dtype=np.float32
        )
        
        self.lidar_data = np.zeros(12)  
        self.robot_pose = np.zeros(3)  
        self.prev_pose = np.zeros(3)
        self.goal = [4.0, 4.0] 
        self.spinning = True
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.start()
        self.max_steps = 200
        self.step_count = 0
        time.sleep(1.0)

    def _spin(self):
        while self.spinning and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.01)

    def lidar_callback(self, msg):
        
        num_points = len(msg.ranges)
        step = num_points // 12  
        ranges = np.array(msg.ranges)
        
        
        ranges = np.nan_to_num(ranges, nan=5.0, posinf=5.0, neginf=0.0)
        
        
        indices = np.arange(0, num_points, step)[:12]
        self.lidar_data = np.clip(ranges[indices], 0.0, 5.0)  

    def odom_callback(self, msg):
        self.prev_pose = self.robot_pose.copy()
        
        
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        
        
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.robot_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def _get_distance_to_goal(self):
        return math.sqrt((self.goal[0] - self.robot_pose[0])**2 + 
                         (self.goal[1] - self.robot_pose[1])**2)

    def _get_heading_to_goal(self):
        return math.atan2(self.goal[1] - self.robot_pose[1], 
                          self.goal[0] - self.robot_pose[0])

    def _get_normalized_heading_error(self):
        
        heading_to_goal = self._get_heading_to_goal()
        heading_error = heading_to_goal - self.robot_pose[2]
        
        
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        
        return heading_error / math.pi

    def step(self, action):
        self.step_count += 1
        
        
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_pub.publish(twist)
        
        
        time.sleep(0.1)
        
        
        dist_to_goal = self._get_distance_to_goal()
        heading_error = self._get_normalized_heading_error()
        
        
        
        obs = np.concatenate([
            self.lidar_data,
            self.robot_pose,
            np.array(self.goal),
            np.array([dist_to_goal])
        ])
        
        
        reward = 0.0
        
        
        prev_dist = math.sqrt((self.goal[0] - self.prev_pose[0])**2 + 
                             (self.goal[1] - self.prev_pose[1])**2)
        distance_improvement = prev_dist - dist_to_goal
        reward += distance_improvement * 20.0  
        
        
        alignment_reward = 1.0 - abs(heading_error)  
        reward += alignment_reward * 1.0
        
        
        if action[0] <= 0:
            reward -= 0.1
        
        
        reward += action[0] * 0.5  
        
        
        done = False
        info = {}
        
        
        min_lidar = np.min(self.lidar_data)
        if min_lidar < 0.25:
            reward = -5.0
            done = True
            info['termination_reason'] = 'collision'
        
        
        elif dist_to_goal < 0.5:
            reward = 50.0
            done = True
            info['termination_reason'] = 'goal_reached'
            
        
        elif self.step_count >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        
        return obs, reward, done, info

    def reset(self):
        
        twist = Twist()
        self.cmd_pub.publish(twist)
        time.sleep(0.5)
        
        
        self.step_count = 0
        
        
        time.sleep(0.5)
        
        
        dist_to_goal = self._get_distance_to_goal()
        obs = np.concatenate([
            self.lidar_data,
            self.robot_pose,
            np.array(self.goal),
            np.array([dist_to_goal])
        ])
        
        return obs

    def close(self):
        self.spinning = False
        if self.spin_thread.is_alive():
            self.spin_thread.join()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()