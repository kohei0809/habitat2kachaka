import sys
import os
import time

import numpy as np
import math

import kachaka_api
from utils import move
sys.path.append(f"/Users/{os.environ['USER']}/Desktop/habitat2kachaka/kachaka-api/python/")

def cal_distance(pre_location, now_location):
    pre_x = pre_location.x
    pre_y = pre_location.y
    pre_theta = pre_location.theta
    now_x = now_location.x
    now_y = now_location.y
    now_theta = now_location.theta
    
    dif_x = now_x - pre_x
    dif_y = now_y - pre_y
    dif_theta = now_theta - pre_theta
    distance = math.sqrt(dif_x*dif_x + dif_y*dif_y)
    dif_theta_deg = math.degrees(dif_theta)
    
    print(f"distance={distance}, theta={dif_theta_deg}")


if __name__ == "__main__":
    ip = "192.168.100.35:26400"
    client = kachaka_api.KachakaApiClient(ip)
    client.update_resolver()
    #print(client.get_locations())
    
    pos_num = 1
    # startに移動する
    client.move_shelf("S01", "start")
    client.set_auto_homing_enabled(False)
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    location = client.get_robot_pose()
    print(location)
    
    client.move_forward(0.25)
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    client.move_forward(0.25)
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    theta_rad = -math.pi/6
    client.rotate_in_place(theta_rad) 
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    client.move_forward(0.25)
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    theta_rad = math.pi/6
    client.rotate_in_place(theta_rad) 
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    client.move_forward(0.25)
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    theta_rad = math.pi/6
    client.rotate_in_place(theta_rad) 
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
    theta_rad = math.pi/6
    client.rotate_in_place(theta_rad) 
    time.sleep(1)
    print(f"########## pose {pos_num} ############")
    pos_num += 1
    pre_location = location
    location = client.get_robot_pose()
    print(location)
    cal_distance(pre_location, location)
    time.sleep(1)
    
