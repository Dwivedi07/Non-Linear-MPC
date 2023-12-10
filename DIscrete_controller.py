#!/usr/bin/env python3
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
import time

sys.path.append('../../')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


global path, store_cp, steer, c_t_s_in, c_t_in
path = [[],[]]
store_cp = [[],[]]
store_steer = []
store_throttle = []
store_yaw = []
vel = []
x_initial = vertcat([1],[1],[0.01],[1],[1])
state_update = 0
steer = 0
c_t_s_in = 0
c_t_in = 0
plot_path = [[], []]
sol_store = []
n_sols = 5
iteration = 0
avgruntime = 0


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
    roll_angle = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_angle = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_angle = math.atan2(t3, t4)
        
    EuA = EulerAngles(roll_angle, pitch_angle, yaw_angle)
    return EuA.yaw


def path_callback(data):
    global path, yaw_angle, car_x, car_y

    if(len(data.poses)== 0):
        return

    path_1 = [[],[]]
    path_t = [[],[]]

    # Global Frame 
    for j in range(len(data.poses)):
        path_1[0].append((data.poses[j].pose.position.x))
        path_1[1].append((data.poses[j].pose.position.y))
 

    for j in range(len(data.poses)):
        path_t[0].append((path_1[0][j] - 10)*math.cos(yaw_angle) + (path_1[1][j] - 10)*math.sin(yaw_angle) + car_x)
        path_t[1].append((-1)*(path_1[0][j] - 10)*math.sin(yaw_angle) + (path_1[1][j] - 10)*math.cos(yaw_angle) + car_y)
    
    path = path_t


def vel_callback(vel_arr):
    
    vel = np.zeros(len(vel_arr.data))
    for i in range(len(vel_arr.data)):
        vel[i] = vel_arr.data[i]
    


def steer_callback(steer_i):
    global steer, c_t_s_in
    steer = steer_i.data / 17
    # steernw_pub.publish(c_t_s_in)
    # if c_t_in > 0:
    #     acc_pub.publish(c_t_in)
    #     # print("Throttle",c_t_in)
    # else:
    #     brake_pub.publish(0.32*200*c_t_in)
        # print("Brake Torque",0.32*100*c_t_in)    

def state_callback(data):
    global x_0, yaw_angle, car_x, car_y, state_update
    pose_arr = []
    twist_arr = []
    state_update = 1
    yaw_angle = quaternion_to_yaw(Quaternion(data.pose[11].orientation.x,
                                             data.pose[11].orientation.y,
                                             data.pose[11].orientation.z,
                                             data.pose[11].orientation.w))

    car_x = data.pose[11].position.x
    car_y = data.pose[11].position.y


    pose_arr.append(data.pose[-1].position.x)
    pose_arr.append(data.pose[-1].position.y)

    twist_arr.append(data.twist[-1].linear.x)
    twist_arr.append(data.twist[-1].linear.y)
    twist_arr.append(data.twist[-1].angular.z)

    #Global Frame
    # print(steer,'this is steer at input x_0')
    x_0 = vertcat(  [float(pose_arr[0])],
                    [float(pose_arr[1])],
                    [float(math.sqrt(twist_arr[0]**2 + twist_arr[1]**2))],
                    [yaw_angle],             
                    [steer]
                )




class NMPController():
    def __init__(self, N_ref = 1):

        #Model and MPC variables
        self.model_type = 'discrete'  
        self.J = 375            # moment of interia
        self.La = 1.2           # distance of front tires from COM in m
        self.Lb = 1.2           # distance of back tires from COM in m
        self.m = 200            # mass of vehicle in kg
        self.Cy = 0.1           # Tyre stiffness constant
        self.t_s = 0.08         # sample time
        self.N = 25             # Control Horizon
        self.time_steps = 40    # time steps
        self.N_ref = 1          # control iterations


        self.a = 0
        self.w = 0
        self.a_prev = 0
        self.w_prev = 0

        self.xw = 2
        self.yw = 10
        self.vw = 4
        self.daw = 2
        self.dww = 2


    
        # MPC variables
        self.model = self.mpc_model()
        lx = car_x
        ly = car_y
    
        for i in range(40):
            self.control_callback(self.model)
            store_cp[0].append(car_x)
            store_cp[1].append(car_y)
            store_steer.append(c_t_s_in)
            store_throttle.append(c_t_in)
            store_yaw.append(yaw_angle)


        x = np.linspace(10, 110, 500)
        y = [10,10,10,10,10,10,10,10,10,10,10,10]
        X =[]
        Y =[]
        steps = [i for i in range(40)]
        for i in range(10):
            X.append((x[i] - 10) + store_cp[0][0])
            Y.append((x[i] - 10) + store_cp[1][0])

        #plt.plot(store_cp[0], store_cp[1],marker = 'd',label ='tracked')
        #plt.plot(path[0], path[1],marker = 'o',label ='path')
        # plt.plot(lx,ly-2,marker ='o',label ='initial')
        # plt.plot(lx+20,ly-2-1.22,marker = 'o',label ='target at 20 units ahead')
        # plt.plot(lx+40,ly-2-3.95,marker = 'o',label ='target at 40 units ahead')
        # plt.plot(steps, store_steer, marker = 'o',label ='steer')
        # plt.plot(steps, store_throttle, marker = 'o',label ='throttle')
        # plt.plot(steps, store_yaw, marker = 'o',label ='yaw_car_angle')
        #plt.legend()
        #plt.show()
        
        



    def mpc_model(self):

        model = do_mpc.model.Model(self.model_type)

        # States variables of the model
        xc = model.set_variable(var_type='_x',var_name='xc',shape = (1,1))                     # x position
        yc = model.set_variable(var_type='_x',var_name='yc',shape = (1,1))                     # y position
        v = model.set_variable(var_type='_x',var_name='v',shape = (1,1))                       # velocity in x
        psi = model.set_variable(var_type='_x',var_name='psi',shape = (1,1))                   # yaw angle
        delta = model.set_variable(var_type='_x',var_name='delta',shape = (1,1))               # steering angle
    
        # Time varying parameter
        x_set = model.set_variable(var_type='_tvp', var_name='x_set', shape = (1,1))
        y_set = model.set_variable(var_type='_tvp', var_name='y_set', shape = (1,1))
        v_set = model.set_variable(var_type='_tvp', var_name='v_set', shape = (1,1))

        # Control inputs
        a = model.set_variable(var_type='_u',var_name='a',shape = (1,1))                       # acceleration
        w = model.set_variable(var_type='_u',var_name='w',shape = (1,1))                       # steering rate (angular)
    
        # Auxillary Expressions
        # beta = np.arctan2((self.Lb*math.tan(delta)),self.La+self.Lb)

        xn = xc + v*np.cos(psi)*self.t_s
        yn = yc + v*np.sin(psi)*self.t_s
        vn = v + a*self.t_s
        psin= psi + (v/self.Lb) * delta *self.t_s
        deltan = delta + w*self.t_s
        
        model.set_rhs('xc',xn)
        model.set_rhs('yc',yn)
        model.set_rhs('v',vn)
        model.set_rhs('psi',psin)
        model.set_rhs('delta',deltan)

        # model being set up:
        model.setup()
        print("Model has been Set:",model.x.keys())

        return model




    def control_callback(self, model):
    
        global x_0, steer, c_t_s_in, c_t_in, sol_store, iteration, avgruntime


        def mpc_controller(self, model):
            # print('Entered the conroller setup')
            mpc = do_mpc.controller.MPC(self.model)

            

            # Set parameters:
            setup_mpc = {
                'n_robust': 0,
                'n_horizon': self.N,
                't_step': self.time_steps,
                'state_discretization': 'discrete',
                'store_full_solution': True,
                # Use MA27 linear solver in ipopt for faster calculations:
                # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
            }


            mpc.set_param(**setup_mpc)

            xc = model.x['xc']
            yc = model.x['yc']
            v  = model.x['v']

            x_set = model.tvp['x_set']
            y_set = model.tvp['y_set']
            v_set = model.tvp['v_set']


            # Objective function:
            # Cost = self.xw*(xc - x_set) ** 2 + self.yw*(yc - y_set)**2 + self.vw*(v - v_set)**2 + self.daw*(self.a_prev - self.a)**2 + self.dww*(self.w_prev - self.w)**2
            cost = self.xw*(xc - x_set) ** 2 + self.yw*(yc - y_set)**2

            self.a_prev, self.w_prev = self.a, self.w

            mterm = cost                                                  # terminal cost
            lterm = cost                                                  # stage cost

            mpc.set_objective(mterm=mterm, lterm=lterm)
            mpc.set_rterm(a = 1)                                     
            mpc.set_rterm(w = 10) # Scaling for quad. cost.

            ####################### State and input bounds #######################3


            mpc.bounds['lower','_x','v'] = 0 #max reverse speed in m/s
            #mpc.bounds['lower','_x','psi'] = -50
            mpc.bounds['lower','_x','delta'] = -np.pi/6

            mpc.bounds['upper','_x','v'] = 5 #max forward speed in m/s
            #mpc.bounds['upper','_x','psi'] = 50
            mpc.bounds['upper','_x','delta'] = np.pi/6


            mpc.bounds['lower','_u','a'] = -10
            mpc.bounds['lower','_u','w'] = -10
            mpc.bounds['upper','_u','a'] = 10
            mpc.bounds['upper','_u','w'] = 10
            

            #get tvp template
            tvp_struct_mpc = mpc.get_tvp_template()
            tvp_struct_mpc['_tvp',:,'v_set'] = 0.1

            def tvp_func_mpc(t_now):
                for i in range(self.N):
                    # Hybrid Astar
                    tvp_struct_mpc['_tvp',i,'x_set'] = path[0][0]  + ((path[0][self.N] - path[0][0])/self.N) + i*((path[0][self.N] - path[0][0])/self.N) 
                    tvp_struct_mpc['_tvp',i,'y_set'] = path[1][0]  + ((path[1][self.N] - path[1][0])/self.N) + i*((path[1][self.N] - path[1][0])/self.N)

                return tvp_struct_mpc
            
            mpc.set_tvp_fun(tvp_func_mpc)

            mpc.setup()

            return mpc
    
    
        def mpc_simulator(self, model):
            # Obtain an instance of the do-mpc simulator class
            # and initiate it with the model
            simulator = do_mpc.simulator.Simulator(self.model)

            # Set parameter(s):
            simulator.set_param(t_step = self.t_s)


            #Setting up for the parameters
            #Setting up for _tv_parameters
            #get tvp template
            tvp_template = simulator.get_tvp_template()

            def tvp_fun(t_now):
        
                #Path from Hybrid Astar
                tvp_template['x_set'] = path[0][0]  + ((path[0][self.N] - path[0][0])/self.N) + t_now*((path[0][self.N] - path[0][0])/self.N) 
                tvp_template['y_set'] = path[1][0]  + ((path[1][self.N] - path[1][0])/self.N) + t_now*((path[1][self.N] - path[1][0])/self.N)
                tvp_template['v_set'] = 0.1
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

        # print("################### Controller is being called ###################")

        for n_ref in range(self.N_ref):
            start_time = time.time() 
            blockPrint()                
            if sol_store:
                u0 = sol_store.pop(0)
                c_t_in = u0[0]
                c_t_s_in = 17*u0[1]*self.t_s
                self.a, self.w = c_t_in, c_t_s_in

                if u0[0]>=0:
                    acc_pub.publish(u0[0])
                    print("Throttle: ",u0[0])
                else:
                    force = u0[0] * self.m
                    torque = -0.32 * force
                    brake_pub.publish(torque)
                    print("Torque for Brakes: ",torque)

                # steer += self.t_s*u0[1]
                steernw_pub.publish(u0[1][0]*17)
                # steer = self.t_s*u0[1][0]
                # steernw_pub.publish(self.t_s*u0[1][0]*17)
                gear_pub.publish(0)
                print("Steer: ",steer)
                enablePrint()
                end_time = time.time()
                iteration += 1
                print("Runtime for", iteration, "iteration:", end_time - start_time, "s")
                avgruntime += end_time - start_time
            else:
                start_time = time.time()
                u0 = self.controller.make_step(x_0)
                sol_store = list(np.array(self.controller.opt_x_num['_u']).reshape(self.N,2)[:n_sols])
                blockPrint()
                enablePrint()
                end_time = time.time()
                print("Average runtime for every iteration in previous block is", avgruntime/n_sols, "s")
                print()
                print("Runtime for solution store for", n_sols, "solutions is ", end_time - start_time, "s")
                avgruntime = end_time - start_time





def init_node():

    global acc_pub, brake_pub, gear_pub, steernw_pub

    rospy.init_node('control_node', anonymous=True)

    state_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, state_callback)
    path_sub = rospy.Subscriber("/path", Path, path_callback)
    vel_sub = rospy.Subscriber("/best_velocity", Float64MultiArray, vel_callback)
    steer_sub = rospy.Subscriber("/current_steer_angle", Float64, steer_callback)
   
    acc_pub = rospy.Publisher('/throttle_cmd', Float64, queue_size=10)
    brake_pub = rospy.Publisher('/brake_cmd', Float64, queue_size=10)
    steernw_pub = rospy.Publisher('/steering_cmd', Float64, queue_size=10)
    gear_pub = rospy.Publisher('/gear_cmd', UInt8, queue_size=10)


    #rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print("Programme is starting from here-------------------- \n Wait till path points got stored and yaw angle got set !!!")
        while len(path[0])<=26 :
            print("less points recieved. Currently number of path points recieved: ", len(path[0]))
        while state_update == 0:
            print("State not updated")

        controlObj = NMPController()
        
 
if __name__ == '__main__':
    try:
        init_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('ROS Interrupt Exception')
        pass
