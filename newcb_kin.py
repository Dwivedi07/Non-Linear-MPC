#!/usr/bin/env python
from logging import captureWarnings
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from std_msgs.msg import UInt8
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from io import StringIO
import numpy as np
import sys
import os
import do_mpc
from do_mpc.data import save_results, load_results
from casadi import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import datetime
import math
sys.path.append('../../')





path = [[],[]]
vel = []
x_initial = vertcat([0],[0],[1.5],[0],[0])
a = 1
steer = 0

class Quaternion:
  def __init__(self, x, y, z, w):
    self.x = x
    self.y = y
    self.z = z
    self.w = w

class EulerAngles:
  def __init__(self, roll, pitch, yaw):
    self.roll = roll
    self.pitch = pitch
    self.yaw = yaw

def quaternion_to_yaw(Q):

    x=Q.x
    y=Q.y
    z=Q.z
    w=Q.w
        
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_angle = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_angle = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_angle = math.degrees(math.atan2(t3, t4))
        
    EuA = EulerAngles(roll_angle, pitch_angle, yaw_angle)
    return EuA.yaw


def path_callback(data):
    global path, yaw_angle, car_x, car_y
    
    # Local frame
    # for j in range(len(data.poses)):
    #     path[0].append((data.poses[j].pose.position.x ))
    #     path[1].append((data.poses[j].pose.position.y ))
    
    # path[0] = path[0][::-1]
    # path[1] = path[1][::-1]

    # Global Frame 
    path_1 = [[],[]]
    for j in range(len(data.poses)):
        path_1[0].append((data.poses[j].pose.position.x ))
        path_1[1].append((data.poses[j].pose.position.y ))

    for j in range(len(data.poses)):
        path[0].append((path_1[0][j] - 10)*np.cos(yaw_angle) + (path_1[1][j] - 10)*np.sin(yaw_angle) + car_x)
        path[1].append((-1)*(path_1[0][j] - 10)*np.sin(yaw_angle) + (path_1[1][j] - 10)*np.cos(yaw_angle) + car_y)
    
    # print('These are intialized global path points path callback(x,y)', path[0][0],path[1][0]) 



def vel_callback(vel_arr):
    
    vel = np.zeros(len(vel_arr.data))
    for i in range(len(vel_arr.data)):
        vel[i] = vel_arr.data[i]
    
    


def steer_callback(steer_i):
    steer = steer_i.data / 17
    

def state_callback(data):
    global x_0, yaw_angle, car_x, car_y, a
    pose_arr = []
    twist_arr = []
    a = a*2
    yaw_angle = quaternion_to_yaw(Quaternion(data.pose[11].orientation.x,
                                             data.pose[11].orientation.y,
                                             data.pose[11].orientation.z,
                                             data.pose[11].orientation.w))


    #Global frame
    car_x = data.pose[11].position.x
    car_y = data.pose[11].position.y

    # print('car_x', 'car_y',car_x,car_y)

    pose_arr.append(data.pose[-1].position.x)
    pose_arr.append(data.pose[-1].position.y)

    twist_arr.append(data.twist[-1].linear.x)
    twist_arr.append(data.twist[-1].linear.y)
    twist_arr.append(data.twist[-1].angular.z)

    #Global Frame
    x_0 = vertcat(  [pose_arr[0]],
                    [pose_arr[1]],
                    [float(np.sqrt(twist_arr[0]**2 + twist_arr[1]**2))],
                    [yaw_angle],                 
                    [steer]
                )


    # #Local Frame
    # x_0 = vertcat(  [car_x],
    #                 [car_y],
    #                 [float(np.sqrt(twist_arr[0]**2 + twist_arr[1]**2))],
    #                 [0],       
    #                 [steer]
    #             )




class NMPController():
    def __init__(self, N_ref = 1):

       

        #Model and MPC variables
        self.model_type = 'continuous'  
        self.J = 375            # moment of interia
        self.La = 1.2           # distance of front tires from COM in m
        self.Lb = 1.2           # distance of back tires from COM in m
        self.m = 200            # mass of vehicle in kg
        self.Cy = 0.1           # Tyre stiffness constant
        self.t_s = 0.1          # sample time
        self.N = 10             # Control Horizon
        self.N_ref = 1          # control iterations


    
        #MPC  variables
        self.model = self.mpc_model()
    
        for i in range(10):
            self.control_callback(self.model)

            
        



    def mpc_model(self):

        model = do_mpc.model.Model(self.model_type)

        # States variables of the model
        xc = model.set_variable(var_type='_x',var_name='xc',shape = (1,1))                     # x position
        yc = model.set_variable(var_type='_x',var_name='yc',shape = (1,1))                     # y position
        v = model.set_variable(var_type='_x',var_name='v',shape = (1,1))                       # velocity
        psi = model.set_variable(var_type='_x',var_name='psi',shape = (1,1))                   # yaw angle
        delta = model.set_variable(var_type='_x',var_name='delta',shape = (1,1))               # steering angle
    
        #time varying parameter
        x_set = model.set_variable(var_type='_tvp', var_name='x_set', shape = (1,1))
        y_set = model.set_variable(var_type='_tvp', var_name='y_set', shape = (1,1))
        v_set = model.set_variable(var_type='_tvp', var_name='v_set', shape = (1,1))
        #control inputs
        a = model.set_variable(var_type='_u',var_name='a',shape = (1,1))    # acceleration
        w = model.set_variable(var_type='_u',var_name='w',shape = (1,1))    # steering rate (angular)
    
        equations = vertcat( v * np.sin(psi), 
                           v * np.cos(psi),
                           a ,
                           (v/self.Lb) * np.tan(delta),
                           w
                         )
        
        model.set_rhs('xc',equations[0])
        model.set_rhs('yc',equations[1])
        model.set_rhs('v',equations[2])
        model.set_rhs('psi',equations[3])
        # model.set_rhs('psi_dot',equations[4])
        model.set_rhs('delta',equations[4])


        #model being set up:
        model.setup()
        print("Model has been Set:",model.x.keys())

        return model






    def control_callback(self, model):
    
        global x_0, steer

        def mpc_controller(self, model):
            print('Here i just entered the conroller setup')
            mpc = do_mpc.controller.MPC(self.model)

            

            # Set parameters:
            setup_mpc = {
                'n_horizon': self.N,
                't_step': self.t_s,
                'n_robust': 0,
                'open_loop':0,
                'state_discretization': 'collocation',
                'collocation_type': 'radau',
                'collocation_deg': 2,
                'collocation_ni': 2,
                'store_full_solution': True,
                
            }

            mpc.set_param(**setup_mpc)

            xc = model.x['xc']
            yc = model.x['yc']
            v = model.x['v']

            x_set = model.tvp['x_set']
            y_set = model.tvp['y_set']
            v_set = model.tvp['v_set']

            # Objective function:
            cost = (xc - x_set) ** 2 + (yc - y_set)**2 + (v - v_set)**2
            
            print("Cost ::", cost)

            mterm = cost                                                  # terminal cost
            lterm = cost                                                  # stage cost

            mpc.set_objective(mterm=mterm, lterm=lterm)
            mpc.set_rterm(a= 0.01)                                      #if very large term it would penalize strongly i think it should be large
            mpc.set_rterm(w =0.011) # Scaling for quad. cost.

            ####################### State and input bounds #######################3


            mpc.bounds['lower','_x','v'] = 0 #max reverse speed in m/s
            mpc.bounds['lower','_x','psi'] = -50
            mpc.bounds['lower','_x','delta'] = -np.pi/6
            mpc.bounds['upper','_x','v'] = 2.2 #max forward speed in m/s
            mpc.bounds['upper','_x','psi'] = 50
            mpc.bounds['upper','_x','delta'] = np.pi/6

            mpc.bounds['lower','_u','a'] = -10
            mpc.bounds['lower','_u','w'] = -10
            mpc.bounds['upper','_u','a'] = 10
            mpc.bounds['upper','_u','w'] = 10


            # print('These are intialized global path points(x,y)', path[0][0],path[1][0]) 


            # plotting the path::
            # plt.plot(car_x, car_y, marker ='D')
            # plt.plot(path[0][:30],path[1][:30],color = 'r', marker = 'o')
            # plt.plot(path[0][:10],path[1][:10],color = 'g',marker = 'x')
            # plt.xlabel("x points")
            # plt.ylabel("y points")
            # plt.show()

            #get tvp template
            tvp_struct_mpc = mpc.get_tvp_template()
            tvp_struct_mpc['_tvp',:,'v_set'] = 1.5

            #get tvp function ----which will basically let the cost function to call at iteration and time and change the tvp value
            # Basically for x_ref and y_ref whoose value or set point are going to change with the time
            def tvp_func_mpc(t_now):
                #define sth fro x_ref as we will going to think about it
                for i in range(self.N):

                    
                    # #Hybrid Astar
                    tvp_struct_mpc['_tvp',i,'x_set'] = path[0][0]  + ((path[0][10] - path[0][0])/10) + i*((path[0][10] - path[0][0])/10) 
                    
                    # print(path[0][0]  + ((path[0][10] - path[0][0])/20) + i*((path[0][10] - path[0][0])/20))

                    tvp_struct_mpc['_tvp',i,'y_set'] = path[1][0]  + ((path[1][10] - path[1][0])/10) + i*((path[1][10] - path[1][0])/10)

                    # print(path[1][0]  + ((path[1][10] - path[1][0])/20) + i*((path[1][10] - path[1][0])/20))

                
                return tvp_struct_mpc
            
            mpc.set_tvp_fun(tvp_func_mpc)

            mpc.setup()

            return mpc
    
    
        def mpc_simulator(self, model):
            # Obtain an instance of the do-mpc simulator class
            # and initiate it with the model:
            simulator = do_mpc.simulator.Simulator(self.model)

            # Set parameter(s):
            simulator.set_param(t_step = self.t_s)


            #Setting up for the parameters
            #Setting up for _tv_parameters
            #get tvp template
            tvp_template = simulator.get_tvp_template()

            def tvp_fun(t_now):
        
                #Path from Hybrid Astar
                tvp_template['x_set'] = path[0][0]  + ((path[0][10] - path[0][0])/10) + t_now*((path[0][10] - path[0][0])/10) 
                tvp_template['y_set'] = path[1][0]  + ((path[1][10] - path[1][0])/10) + t_now*((path[1][10] - path[1][0])/10)

                tvp_template['v_set'] = 1.5
                return tvp_template

            simulator.set_tvp_fun(tvp_fun)

            # Setup simulator:
            simulator.setup()

            return simulator


        self.controller = mpc_controller(self, model)
        self.simulator = mpc_simulator(self, model)
        self.estimator = do_mpc.estimator.StateFeedback(model)

        self.simulator.x0 = x_initial
        self.controller.x0 = x_initial
        self.estimator.x0 = x_initial

        self.controller.set_initial_guess()
        self.simulator.set_initial_guess()


        
        acc_pub = rospy.Publisher('/throttle_cmd', Float64, queue_size=10)
        brake_pub = rospy.Publisher('/brake_cmd', Float64, queue_size=10)
        steer_pub = rospy.Publisher('/steering_cmd', Float64, queue_size=10)
        gear_pub = rospy.Publisher('/gear_cmd', UInt8, queue_size=10)

        print("###################'controller is being called'##################")

        
        for k in range(self.N_ref):
            u0 = self.controller.make_step(x_0)

            
            if u0[0][0]>=0:
                acc_pub.publish(u0[0][0])
                print("Throttle:",u0[0][0])
            else:
                force = u0[0][0] * self.m
                torque = -0.32 * force
                brake_pub.publish(torque)
                print("Torque for Brakes:",torque)

            steer += (self.t_s)*u0[1][0]
            steer_pub.publish((self.t_s)*u0[1][0]*17)
            gear_pub.publish(0)

            y_n = self.simulator.make_step(u0)                  # Simulate the next step using the control inputs
            print(" Steer:",steer)

        



def init_node():
    rospy.init_node('control_node', anonymous=True)

    state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, state_callback)
    path_sub = rospy.Subscriber("/path", Path, path_callback)
    vel_sub = rospy.Subscriber("/velocity_array2", Float64MultiArray, vel_callback)
    steer_sub = rospy.Subscriber("/current_steer_angle", Float64, steer_callback)
   

    acc_pub = rospy.Publisher('/throttle_cmd', Float64, queue_size=10)
    brake_pub = rospy.Publisher('/brake_cmd', Float64, queue_size=10)
    steer_pub = rospy.Publisher('/steering_cmd', Float64, queue_size=10)
    gear_pub = rospy.Publisher('/gear_cmd', UInt8, queue_size=10)


    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print("Programme is starting from here--------------------------------------- \n Wait till path points got stored and yaw angle got set!!!")
        while len(path[0])<=100 :
            m = 2
        while a == 1:
            n = 1
        print(len(path[0]), yaw_angle)
        controlObj = NMPController()
        
 
if __name__ == '__main__':
    try:
        init_node()
    except rospy.ROSInterruptException:
        print('ROS Interrupt Exception')
        pass