#!/usr/bin/env python3

import rospy
import json
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point

import sys
import termios
import tty

class CommandNode:
    def __init__(self):
        rospy.init_node("command_node", anonymous=True)
        self.pub = rospy.Publisher("/serial/command", String, queue_size=10)
        self.init_imu = rospy.Publisher("/init_imu", Float32, queue_size=10)
        self.init_position = rospy.Publisher("/init_position", Point, queue_size=10)
        self.speed =0
        self.steer_angle = 0

        self.select_mode()

    def select_mode(self): #maybe too many loop?
        while not rospy.is_shutdown():
            mode = input("Select mode: kl or wasd or init: ")
            if mode == "kl":
                self.kl_mode()
                break
            elif mode == "wasd":
                self.wasd_mode()
            elif mode == "init":
                self.init_param()
            elif mode == "q":
                exit(0)
            else:
                print("Invalid mode")

    def init_param(self):
        while not rospy.is_shutdown():
            try:
                mode = input("Select mode: init_yaw or init_position: ")
                if mode == "init_yaw":
                    self.init_imu.publish(Float32(0))
                    break
                elif mode == "init_position":
                    position = Point()
                    position.x = float(input("Enter x position: "))
                    position.y = float(input("Enter y position: "))
                    position.z = 0
                    self.init_position.publish(position)
                    break
                else:
                    print("Invalid mode")
            except rospy.ROSInterruptException:
                break
            except KeyboardInterrupt:
                rospy.loginfo("KL Command Node Stopped.")
                break



    def getch(self):
        fd = sys.stdin.fileno()
        old_settings=termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd,termios.TCSADRAIN, old_settings)
        return ch
    

    def kl_mode(self):
        """사용자로부터 KL 값을 입력받아 JSON 변환 후 ROS 메시지로 Publish"""
        rospy.loginfo("\nPress 'b' to go back to mode select, 'q' to quit")
        while not rospy.is_shutdown():
            try:
                user_input = input("Enter KL mode (0, 15, 30): ").strip()

                # 입력값이 0, 15, 30 중 하나인지 확인
                if user_input in ["0", "15", "30"]:
                    command = {"action": "kl", "mode": int(user_input)}
                    command_str = json.dumps(command)  
                    self.pub.publish(command_str)
                    rospy.loginfo(f"Sent KL command: {command_str}")
                elif user_input == "b":
                    self.select_mode()
                    break
                elif user_input=="q":
                    exit(0)
                else:
                    rospy.logwarn("Invalid input. Please enter 0, 15, or 30.")
            
            
            except rospy.ROSInterruptException:
                break
            except KeyboardInterrupt:
                rospy.loginfo("KL Command Node Stopped.")
                break
    
    def wasd_mode(self):
        rospy.loginfo("\nWASD Mode: press w, a, s, d to control the car")
        rospy.loginfo("Press 'b' to go back to mode select, 'q' to quit")
        while not rospy.is_shutdown():
            try:
                key=self.getch()
                if key == "w":
                    self.speed +=20
                elif key =="s":
                    if self.speed>=20:
                        self.speed -=20
                elif key == "a":
                    self.steer_angle -=20
                elif key == "d":
                    self.steer_angle +=20
                elif key =="b":
                    self.select_mode()
                    break
                elif key == "q":
                    exit(0)
                else:
                    continue

                if (key == "w") or (key == "s"):
                    speed_command = {"action": "speed", "speed":int(self.speed)}
                    speed_str = json.dumps(speed_command)
                    self.pub.publish(speed_str)
                elif (key == "a") or (key == "d"):
                    steer_command = {"action": "steer", "steerAngle":int(self.steer_angle)}
                    steer_str = json.dumps(steer_command)
                    self.pub.publish(steer_str)
            except rospy.ROSInterruptException:
                break
            except KeyboardInterrupt:
                rospy.loginfo("KL Command Node Stopped.")
                break
                


if __name__ == "__main__":
    try:
        CommandNode()
    except rospy.ROSInterruptException:
        pass

