import rospy
import time
import math
import serial
from std_msgs.msg import Float32, Float64
from bfmc.msg import bfmc_imu
import numpy as np


class SerialReadNode:
    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        rospy.init_node("serial_read_node", anonymous=True)
        rospy.loginfo("Serial Read Node Started")

        self.serial_port = serial.Serial(port, baudrate, timeout=0.1)

        self.speed_pub = rospy.Publisher("/sensor/speed",Float32,queue_size=10)
        self.seral_speed_pub = rospy.Publisher("/speed",Float32,queue_size=10)
        self.imu_pub = rospy.Publisher("/BFMC_imu",bfmc_imu,queue_size=10)
        self.Q = 0.3
        self.R = 1.3
        self.P = 2.0
        self.K = 0.0

        self.x_est_last = 0.0
        self.x_est = 0.0

        self.count = 0
        self.speed_avg = 0.0
        self.start = time.time()
    def KalmanF(self,speed):
        self.P = self.P+self.Q
        self.K = self.P / (self.P + self.R)
        self.x_est = self.x_est_last + self.K * (speed - self.x_est_last)
        self.P = (1 - self.K) * self.P
        self.x_est_last = self.x_est
        return self.x_est



    def sendqueue(self,buff):
        try:
            action, value = buff.split(":") # @action:value;;
            action = action[1:]
            value = value[:-2]
            if action == "speedSensor":
                speed = value.split(",")[0]
                # F_speed = self.KalmanF(float(speed))
                # F_speed = (math.pi*6.1)*0.25*float(F_speed)/60 # 0.2s
                # self.speed_pub.publish(F_speed)
                
                F_speed = self.KalmanF(float(speed))
                F_speed = (math.pi*6.5)*0.05*float(F_speed)/60*0.9 # 0.5s 
                rospy.loginfo(f"Received speed: {F_speed}")
                self.speed_pub.publish(F_speed)
            elif action == "speed":
                serial_speed = value.split(",")[0]
                if self.isFloat(serial_speed):
                    #print("seral:",serial_speed)
                    self.seral_speed_pub.publish(float(serial_speed))
            elif action == "imu":
                splittedValue = value.split(";")
                if(len(buff)>20):
                    data = {
                        "roll": splittedValue[0],
                        "pitch": splittedValue[1],
                        "yaw": splittedValue[2],
                        "accelx": splittedValue[3],
                        "accely": splittedValue[4],
                        "accelz": splittedValue[5],
                    }
                imu_msg = bfmc_imu()
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.header.frame_id = "imu_link"
                imu_msg.roll = float(data["roll"])
                imu_msg.pitch = float(data["pitch"])
                imu_msg.yaw = float(data["yaw"])
                imu_msg.accelx = float(data["accelx"])
                imu_msg.accely = float(data["accely"])
                imu_msg.accelz = float(data["accelz"])

                self.imu_pub.publish(imu_msg)

            
            
            

        except Exception as e:
            rospy.logerr(f"Error processing serial data: {e}")

    def run(self):
        buff =""
        isResponse = False
        self.start = time.time()

        while not rospy.is_shutdown():
            read_chr = self.serial_port.read()
            try:
                read_chr = read_chr.decode("ascii")
                if read_chr == "@":
                    isResponse = True
                    buff = ""
                elif read_chr == "\r":
                    isResponse = False
                    if len(buff) != 0:
                        self.sendqueue(buff)
                if isResponse:
                    buff += read_chr
            except Exception as e:
                print(e)

    def isFloat(self, string):
        try: 
            float(string)
        except ValueError:
            return False
        
        return True
            
if __name__ == "__main__":
    node = SerialReadNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass

#===========================================================#

# import rospy
# import time
# import math
# import serial
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from std_msgs.msg import Float32
# from bfmc.msg import bfmc_imu
# import threading

# class SerialReadNode:
#     def __init__(self, port="/dev/ttyACM0", baudrate=115200):
#         rospy.init_node("serial_read_node", anonymous=True)
#         rospy.loginfo("Serial Read Node Started")

#         self.serial_port = serial.Serial(port, baudrate, timeout=0.1)

#         self.speed_pub = rospy.Publisher("/sensor/speed", Float32, queue_size=10)
#         self.seral_speed_pub = rospy.Publisher("/speed", Float32, queue_size=10)

#         self.Q = 0.3
#         self.R = 1.3
#         self.P = 2.0
#         self.K = 0.0

#         self.x_est_last = 0.0
#         self.x_est = 0.0

#         self.speed_data = []
#         self.f_speed_data = []
#         self.time_data = []

#         self.start_time = time.time()
#         self.count = 0
#         self.speed_avg = 0.0
        

#     def KalmanF(self, speed):
#         self.P = self.P + self.Q
#         self.K = self.P / (self.P + self.R)
#         self.x_est = self.x_est_last + self.K * (speed - self.x_est_last)
#         self.P = (1 - self.K) * self.P
#         self.x_est_last = self.x_est
#         return self.x_est

#     def sendqueue(self, buff):
#         try:
#             action, value = buff.split(":")  
#             action = action[1:]
#             value = value[:-2]

#             if action == "speedSensor":
#                 speed = float(value.split(",")[0])
#                 F_speed = self.KalmanF(speed)

                
#                 # speed = (math.pi * 6.1) * 0.1 * speed / 60  

#                 # F_speed = (math.pi * 6.1) * 0.1 * F_speed / 60  
#                 speed = speed/120  

#                 F_speed =F_speed / 120 

#                 current_time = time.time() - self.start_time
#                 self.speed_data.append(speed)
#                 self.f_speed_data.append(F_speed)
#                 self.time_data.append(current_time)
#                 rospy.loginfo(f"speed: {speed}")
#                 rospy.loginfo(f"F_speed: {F_speed}")
#                 # rospy.loginfo(f"Received speed: {F_speed}")
#                 self.speed_pub.publish(F_speed)
                
#                 # average speed
#                 self.count += 1
#                 self.speed_avg += speed
#                 if self.count == 20:
#                     rospy.loginfo(f"###Average speed###: {self.speed_avg / 20}")
#                     self.count = 0
#                     self.speed_avg = 0

                

#             elif action == "speed":
#                 serial_speed = value.split(",")[0]
#                 if self.isFloat(serial_speed):
#                     self.seral_speed_pub.publish(float(serial_speed))

#         except Exception as e:
#             rospy.logerr(f"Error processing serial data: {e}")

#     def run(self):
#         buff = ""
#         isResponse = False

#         while not rospy.is_shutdown():
#             read_chr = self.serial_port.read()
#             try:
#                 read_chr = read_chr.decode("ascii")
#                 if read_chr == "@":
#                     isResponse = True
#                     buff = ""
#                 elif read_chr == "\r":
#                     isResponse = False
#                     if len(buff) != 0:
#                         self.sendqueue(buff)
#                 if isResponse:
#                     buff += read_chr
#             except Exception as e:
#                 print(e)

#     def isFloat(self, string):
#         try:
#             float(string)
#         except ValueError:
#             return False
#         return True


# def update_graph(frame, node, ax, line1, line2):
#     """ 실시간으로 speed와 F_speed를 업데이트하는 함수 """
#     ax.clear()
#     ax.set_title("Speed vs Filtered Speed")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Speed (m/s)")

#     if len(node.time_data) > 50:  
#         node.time_data = node.time_data[-50:]
#         node.speed_data = node.speed_data[-50:]
#         node.f_speed_data = node.f_speed_data[-50:]

#     ax.plot(node.time_data, node.speed_data, label="Raw Speed", color="blue")
#     ax.plot(node.time_data, node.f_speed_data, label="Filtered Speed (Kalman)", color="red")
#     ax.legend()
#     ax.grid()


# if __name__ == "__main__":
#     node = SerialReadNode()
    
#     # 백그라운드에서 SerialReadNode 실행
#     thread = threading.Thread(target=node.run, daemon=True)
#     thread.start()

#     # # 그래프 초기화
#     fig, ax = plt.subplots()
#     ani = animation.FuncAnimation(fig, update_graph, fargs=(node, ax, None, None), interval=500, cache_frame_data=False)

#     # 그래프 창 띄우기
#     plt.show()

