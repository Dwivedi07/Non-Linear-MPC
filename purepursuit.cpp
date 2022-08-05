#include "ros/ros.h"
#include "std_msgs/Int8MultiArray.h"
#include <iostream>
#include <vector>
#include <utility>
#include <math.h>
#include <bits/stdc++.h>
#include <fstream>
#include <algorithm>
#include <gazebo_msgs/ModelStates.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include<typeinfo>
#include <cstdlib>
//#include "matplotlib-cpp/matplotlibcpp.h" // use if needed to plot points on a graph instead of rviz
#include <cmath>

using namespace std;
//using namespace cv;
//namespace plt = matplotlibcpp;

double k = 0.1; // propotional constant
double Lfc = 2.0; //Lookahead distance
double Kp = 1.0; //speed_proportional gain
double dt = 0.1; // time step
double WB = 2.4; // wheel base distance
double target_vel = 1; //target velocity
float yaw;
float gear_ratio = 17; //gear ratio between steering and wheel 17:1
ros::Publisher pub_throttle, pub_brake, pub_steer;





struct Quaternion {
    double w, x, y, z;
};

struct Quaternion Q; //global variable for quaternion

//defining structure for states
struct States {
	float x, y, xr, yr, vx, vy;
};

struct States state;   //global variable for states of car

struct EulerAngles {
    double roll, pitch, yaw;
};

struct EulerAngles Eu ;

//quaternion to euler conversion
EulerAngles ToEulerAngles(Quaternion q) {
    EulerAngles angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        angles.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.yaw = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

void ModelStateCallback(const gazebo_msgs::ModelStates::ConstPtr& msg){


	Q.x=msg->pose[11].orientation.x;
	Q.y=msg->pose[11].orientation.y;
	Q.z=msg->pose[11].orientation.z;
	Q.w=msg->pose[11].orientation.w;

	Eu = ToEulerAngles(Q);
	yaw = Eu.yaw;
  cout<<"yaw: "<<yaw<<endl;

	state.x=msg->pose[11].position.x;
	state.y=msg->pose[11].position.y;
  cout<<"car's x: "<<state.x<<endl;
  cout<<"car's y: "<<state.y<<endl;

	state.xr=msg->pose[11].position.x-((WB / 2) *cos(yaw));
	state.yr=msg->pose[11].position.y-((WB / 2) *sin(yaw));

	state.vx=msg->twist[11].linear.x;
	state.vy=msg->twist[11].linear.y;
  cout<<"car's vx: "<<state.vx<<endl;
  cout<<"car's vy: "<<state.vy<<endl;
}

void PathCallback(const nav_msgs::Path::ConstPtr& msg){
		int NumData = msg->poses.size();  //number of points in path
    vector<double> PathDis;
    PathDis.push_back(0.0);

    float l = k*sqrt(pow(state.vx,2)+pow(state.vy,2))+Lfc;                                           //lookahead distance

    for(int i=1; i<=NumData; i++)
    {
        double x1 = msg->poses[i-1].pose.position.x ,x2 = msg->poses[i].pose.position.x;
        double y1 = msg->poses[i-1].pose.position.y ,y2 = msg->poses[i].pose.position.y;
        PathDis.push_back(PathDis[i-1]+sqrt(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1))));
    }

    int index=0;
    for (int i=0; i<NumData; i++){                    //targeting the point just after the lookahead radius is crossed
    	if((PathDis[i]-l)>0){
    		index = i;
    		break;
    	}
    }

    float ty, tx, ty_g, tx_g;
    ty=msg->poses[index-1].pose.position.y;
    tx=msg->poses[index-1].pose.position.x;

    ty=(ty-10);          //local frame with car as origin
    tx=(tx-10);

    tx_g=(tx*cos(Eu.yaw)-ty*sin(Eu.yaw))+state.x;   //transformation of coordinates of target point to global frame
    ty_g=(tx*sin(Eu.yaw)+ty*cos(Eu.yaw))+state.y;
    ty=ty_g;
    tx=tx_g;



    cout<<"target x: "<<tx<<endl;
    cout<<"target y: "<<ty<<endl;

    float alpha, delta;
    alpha = atan((ty - state.yr)/(tx - state.xr)) - Eu.yaw;
    delta = atan(2.0 * WB *sin(alpha) / l);
    cout<<"alpha: "<<alpha<<endl;
    cout<<"delta: "<<delta<<endl;
    cout<<"steer: "<<gear_ratio*delta<<endl;

    //publishing steering angle
    std_msgs::Float64 steer;
    steer.data=(-1)*gear_ratio*delta;
    pub_steer.publish(steer);

    //throttle and brake

    float current_vel = sqrt(pow(state.vx,2)+pow(state.vy,2));
    cout<<"current_vel: "<<current_vel<<endl;
    float a = Kp*(target_vel - current_vel);
    if(a > 0){
        std_msgs::Float64 th;
        th.data = 0.5;
    	pub_throttle.publish(th);}

    else{
        std_msgs::Float64 bk;
        bk.data=500;

        pub_brake.publish(bk);}

}




int main(int argc,char** argv)
{
	ros::init(argc,argv,"purepursuit");
	ros::NodeHandle PathSubNode, ModelStateSubNode, ThrottlePubNode, SteerPubNode, BrakePubNode;

	ros::Subscriber SubPath = PathSubNode.subscribe<nav_msgs::Path>("/path",1,PathCallback);
	ros::Subscriber SubModelState = ModelStateSubNode.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states",1,ModelStateCallback);

	pub_throttle = ThrottlePubNode.advertise<std_msgs::Float64>("/throttle_cmd",1);
	pub_brake = BrakePubNode.advertise<std_msgs::Float64>("/brake_cmd",1);
	pub_steer = SteerPubNode.advertise<std_msgs::Float64>("/steering_cmd",1);


	ros::Rate RosLoopRate(10);
	while(ros::ok())
	{
		ros::spin();//check for incoming messages
		RosLoopRate.sleep();
	}
	return 0;
}
