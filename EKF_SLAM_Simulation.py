#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg
from matplotlib.pyplot import figure


# In[2]:


#Simulation Constants
delta_t = 0.1                     #time increments
t = np.arange(0,100,delta_t)
np.random.seed(1)

inital_state = [0,0,0]
State_size = 3 #theta, x, y
#Landmark_size  = State_size # dimention of the landmarks
landmark1 = np.array([6,10])

W_v_var = 1     #linear velocity variance 
W_omega_var = 0.1 # Standard diviation #Angular velocity variance
M = np.diag([W_omega_var, W_v_var])

angle_var = np.deg2rad(3) **2
range_var = (1) **2
R = np.diag([angle_var, range_var])  # Observation Sensor Noise


# In[ ]:





# In[3]:


class Agent:
    
    def __init__(robot,X):
        '''
        initializes the robot
        :param X:   robot's initial state
        :param P:   
        '''
        robot.X = X
        robot.P_est = np.diag([np.deg2rad(180/math.pi), 1.0, 1.0])**2 # inital state covariance
        
    
    def print_robot_state(robot):
        print(robot.X)
    
    def state_update(robot, u):
        robot.X = motion_model(robot.X,u)
        return robot.X
    
    def observe(robot, landmark_pos, obsv_noise):
        '''
        Returens the relative position of the landmark observed by the 
        robot with added gaussian noise
        h(x) + noise
        '''
        
        delta_x = landmark_pos[0] - robot.X[1]
        delta_y = landmark_pos[1] - robot.X[2]
        d = math.sqrt(delta_x**2 + delta_y **2)  
        angle = pi_2_pi(math.atan2(delta_y, delta_x) - robot.X[0])
    
        #Addthe observation Guassian noise
#         angle = angle + np.random.normal(0, obsv_noise[0,0])
#         d = d + np.random.normal(0,obsv_noise[1,1])
        angle = angle + np.random.randn() * obsv_noise[0,0]
        d = d + np.random.randn() * obsv_noise[1,1]
        z = np.array ([angle, d])
        
        return z
    
    
    def Time_update(robot, u):  
        '''
        Updates the state and state covariance 
        '''
        #Go to the new state
        robot.X = motion_model(robot.X,u) 
        
        #Calculate the Jacobian of f(x,u), F and B
        Fj, Bj = calc_jacob_of_state(robot.X,u)
        
        #Q = B * M B.T  Process noise
        #P_est =F * P * F.T + Q
        P = Fj @ robot.P_est @ Fj.T
        Q = Bj @ M @ Bj.T 
        robot.P_est = P + Q
        
        return robot.X, robot.P_est
    
    
    
    def Observation_update(robot, landmark_pos, z):
        """
        Returns the updated state and covariance for the system
        """  
        delta_x = landmark_pos[0] - robot.X[1]
        delta_y = landmark_pos[1] - robot.X[2]
        d = math.sqrt(delta_x**2 + delta_y **2)
        
        zangle = pi_2_pi(math.atan2(delta_y, delta_x) - robot.X[0])
        zp = np.array([[pi_2_pi(zangle), d]])       
        #zp is the expected measurement based on xEst and the expected landmark position
        
        y = (z - zp).T      # y = z - h(x) innovation/residual
                            #z contains the observation noise
                            #zp contains the input noise
        
        y[0] = pi_2_pi(y[0])   #Convering the residual angle to -pi tp pi
        
        
        H = jacob_of_H(robot.X, landmark_pos)
        
        #calculating the Kalman's Gain
        S = H @ robot.P_est @ H.T + R 
        K = (robot.P_est @ H.T) @ np.linalg.inv(S) #Kalman Gain
        
        #Update the State
        robot.X = robot.X + (K @ y).T [0]  #Converts the observation to state and adds to the current state
       
        #Update the state covariance
        robot.P_est = (np.eye(len(robot.X)) - (K @ H)) @ robot.P_est 
        robot.X[0] = pi_2_pi(robot.X[0])

        return robot.X, robot.P_est
    
    
    
    
    
    
def calc_jacob_of_state(x, u):
    #Jakobian of f wrt x
    jF = delta_t * np.array([[1,              0, 0], 
                            [-u[1] * math.sin(x[0]), 1, 0],
                            [u[1] * math.cos(x[0]),  0, 1]])
    
    
    #Jacobian of f wrt u
    jB = delta_t * np.array([[1,         0],
                             [0, math.cos(x[0])],
                             [0, math.sin(x[0])]])

    return jF, jB 

def jacob_of_H(x, landmark_pos):
    """ compute Jacobian of H matrix where h(x) computes 
    the range and bearing to a landmark for state x 
    """

    px = landmark_pos[0]
    py = landmark_pos[1]
    sq_diff = (px - x[1])**2 + (py - x[2])**2
    distance = math.sqrt(sq_diff)

    H = np.array(
        [[-1, -(py - x[2]) / sq_diff,  -(px - x[1]) / sq_diff],
         [0,  (px - x[1]) / distance,   -(py - x[2]) / distance]])
    
    
    return H


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = delta_t * np.array([[1,   0.0],
                            [0.0, math.cos(x[0])],
                            [0.0, math.sin(x[0])]])

    #x = f(x,u)
    x = F @ x + B @ u   #Calculate the new state
    x[0] = pi_2_pi(x[0]) # Convert the angle to be in [-pi,pi] range
     
    return x


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

    

def get_input(input_cov):
    
    #input velocities
    max_velocity = 5
    u_v_t = np.arange(0,max_velocity,max_velocity/len(t))
    u_omega_t = np.ones(len(t)) * 0.1             #angular velocity
    
    #Noises
    W_v_t =(np.random.normal(0,input_cov[1,1],len(t)))
    W_omega_t = np.random.normal(0,input_cov[0,0],len(t))
    
    #Input + noise
    u_v_t = u_v_t + W_v_t
    u_omega_t = u_omega_t + W_omega_t
    
    return np.vstack((u_omega_t,u_v_t)).T            #To get the the inputs in pars for [u_v,u_w] at time t




# In[4]:


def main():
    print("start!")
    
    
    
    My_robot_true = Agent(inital_state)
    My_robot_DR   = Agent(inital_state)
    My_robot_Est  = Agent(inital_state)
    
    
    #Input velocities
    U_true = get_input(np.zeros([2,2]))
    U_DR   = get_input(M)
    
    
    true_X = np.zeros((len(t),State_size))
    DR_X   = np.zeros((len(t),State_size))
   
    X_est  = np.zeros((len(t),State_size))
    P_est  = np.zeros((len(t),State_size,State_size))
    Z_DR   = np.zeros((len(t),2))
    
   
    print(len(t))
    for i in range(len(t)-1):
        
        true_X[i,:] = My_robot_true.state_update(U_true[i])
        DR_X[i,:] = My_robot_DR.state_update(U_DR[i])
        
        #calc ovservation to get z
        Z = My_robot_true.observe(landmark1,R)  #Observation from the true trajectory to the lnadmrk 
                                                #then observ. noise added here
        
        #Ekf:
        #State predict
        X_est[i,:], P_est[i,:,:] = My_robot_Est.Time_update(U_DR[i])         #time/State Estimate update
        
        if(i%3== 0):
            #Observation Update
            #print(i)
            #initP = np.eye(2)
            X_est[i,:], P_est[i,:,:] = My_robot_Est.Observation_update(landmark1, Z) #Observation estimate update      

    
    
    #Calculate the the difrenses of estimation and dead reckoning here:
    d_tr_DR  = np.sqrt((true_X[:,1]-DR_X[:,1])**2 + (true_X[:,2]-DR_X[:,2])**2)
    theta_tr_dr = 1 - np.cos(true_X[:,0] - DR_X[:,0])
    
    d_tr_est = np.sqrt((true_X[:,1]-X_est[:,1])**2 + (true_X[:,2]-X_est[:,2])**2)
    theta_tr_est = 1- np.cos(true_X[:,0]-X_est[:,0])
    
    
    #Plots
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    
    #Plot of the trajectories 
    ax1.plot(true_X[0:999,1],true_X[0:999,2],label = 'True trajectory') 
    ax1.plot(DR_X[0:999,1],DR_X[0:999,2], label = 'Dead Reckoning')
    ax1.plot(X_est[0:999,1],X_est[0:999,2], label = 'State Estiamte')
    ax1.scatter(landmark1[0],landmark1[1])
    ax1.title.set_text('Trajectories')
    ax1.legend()
    ax1.set_xlabel('x1')
    ax1.set_ylabel ('x2')
    
    
    #Plot of disarnce diffrences 
    ax2.plot(t, d_tr_DR, label ='||tr-DR||')
    ax2.plot(t, d_tr_est, label = '||tr-EST||')
    ax2.title.set_text('Diffrnece of DR and Est trajectories from true trajectory')
    ax2.legend()
    ax2.set_xlabel('t')
    ax2.set_ylabel ('diffrence')
    
    #Plot of angle diffrences 
    ax3.plot(t, theta_tr_dr, label ='angle(tr-DR)')
    ax3.plot(t, theta_tr_est, label = 'angle(tr - EST)')
    ax3.legend()
    ax3.set_xlabel('t')
    ax3.set_ylabel ('Angle_diffrence (rad)')
    ax3.title.set_text('Diffrnece of DR and estimate angles form true angle')
    plt.tight_layout() 
    plt.show()


print("Done!")

main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




