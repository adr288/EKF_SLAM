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
delta_t = 1                   #time increments
t = np.arange(0,100,delta_t)
np.random.seed(1)                 #set the random number seed

inital_state = [0,0,0]
State_size = 3                #theta, x, y
#Landmark_size  = State_size # dimention of the landmarks
landmark1 = np.array([10,20])

W_v_sd = 0.1                     #Linear velocity variance 
W_omega_sd = 0.05                 #Standard diviation #Angular velocity variance

#Observation Variance
Angle_sd = np.deg2rad(2)
Range_sd = 2


M = np.diag([W_omega_sd, W_v_sd]) **2  # Input Noise
R = np.diag([Angle_sd, Range_sd]) **2  # Observation Sensor Noise


# In[ ]:





# In[3]:


class Agent:
    
    def __init__(robot,X):
        '''
        initializes the robot 
        '''
        robot.X = X
        robot.P_est = (np.diag([np.deg2rad(0), 0, 0]))**2 # inital state covariance
        
    
    def print_robot_state(robot):
        print(robot.X)
    
    def print_robot_cov(robot):
        #print(np.linalg.det(robot.P_est))
        print(np.trace(robot.P_est))
    
    def state_update(robot, u):
        robot.X = motion_model(robot.X,u)
        return robot.X
    
    
    def Time_update(robot, u):  
        '''
        Updates the state and state covariance 
        '''
        
        #Calculate the Jacobian of f(x,u), F and B
        Fj, Bj = calc_jacob_of_state(robot.X,u)
    
        #Q = B * M * B.T  Process noise
        Q = Bj @ M @ Bj.T
        
       
        #Fj = np.eye(3,3)    
        #P_est =Fj * P * Fj.T + Q  
        robot.P_est = Fj @ robot.P_est @ Fj.T  + Q 
        
       
        #Go to the new state
        robot.X = motion_model(robot.X,u) 
        robot.X[0] = pi_2_pi(robot.X[0])
        
        return robot.X, robot.P_est
    
    
    def observe(robot, landmark_pos, obsv_noise):
        '''
        Returens the relative position of the landmark observed by the 
        robot with added gaussian noise
        h(x) + noise
        '''
        
        delta_x = landmark_pos[0] - robot.X[1]          
        delta_y = landmark_pos[1] - robot.X[2]          
        d = np.sqrt(delta_x**2 + delta_y **2)  
        angle = pi_2_pi(math.atan2(delta_y, delta_x)- robot.X[0])

        
        #Addthe observation Guassian noise
        stand_div_angle  = np.sqrt(obsv_noise[0,0])
        stand_div_range  = np.sqrt(obsv_noise[1,1])
        
        angle = angle + np.random.normal(0, stand_div_angle)
        d = d + np.random.normal(0, stand_div_range)
        
        z = np.array ([angle, d])
        
        return z
    
    
    def Observation_update(robot, landmark_pos, z):
        """
        Returns the updated state and covariance for the system
        """  
        delta_x = landmark_pos[0] - robot.X[1]    #est pos
        delta_y = landmark_pos[1] - robot.X[2]    #est pos
        d = math.sqrt(delta_x**2 + delta_y **2)     
        
        zangle = pi_2_pi(math.atan2(delta_y, delta_x)- robot.X[0])   
        zp = np.array([[pi_2_pi(zangle), d]])       
        #zp is the expected measurement based on xEst and the expected landmark position
        
        y = (z - zp).T      # y = z - h(x) innovation/residual
                            #z contains the observation noise
                            #zp contains the input noise
        
        y[0] = pi_2_pi(y[0])   #Convering the residual angle to -pi tp pi
        
        
        H = jacob_of_H(robot.X, landmark_pos)
        
        #calculating the Kalman's Gain New'
        S = H @ robot.P_est @ H.T + R 
        
        #Update the state 
        robot.X = (robot.X + (robot.P_est @ (H.T @ (np.linalg.inv(S) @ y))).T)[0]
        robot.X[0] = pi_2_pi(robot.X[0])
        
       
        #Update the state covariance 
        robot.P_est = robot.P_est - robot.P_est @ H.T @ np.linalg.inv(S) @ H @ robot.P_est
        
        #Update the state covariance usinf matrix inversion lemma
        #robot.P_est = np.linalg.inv(np.linalg.inv(robot.P_est)+ H.T @ np.linalg.inv(R) @ H)
        
        
        
        return robot.X, robot.P_est, y


    
    
def calc_jacob_of_state(x, u):
    
    #Jakobian of f wrt x
    jF = delta_t * np.array([[1,                    0, 0], 
                             [-u[1] * np.sin(x[0]), 1, 0],
                             [u[1] * np.cos(x[0]),  0, 1]])

    

    #Jacobian of f wrt u
    jB = delta_t * np.array([[1,         0],
                             [0, np.cos(x[0])],
                             [0, np.sin(x[0])]])
 
    return jF, jB 


def jacob_of_H(x, landmark_pos):
    """
    compute Jacobian of H matrix where h(x) computes 
    the range and bearing to a landmark for state x 
    
    """
    px = landmark_pos[0]
    py = landmark_pos[1]
    sq_diff = (px - x[1])**2 + (py - x[2])**2
    distance = math.sqrt(sq_diff)

    H = np.array(
        [[-1, (py - x[2]) / sq_diff,  (x[1] - px) / sq_diff],
         [0,  (x[1] - px) / distance,   (x[2] - py) / distance]])
    
    
    return H


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B =  np.array([[1,   0],
                            [0, np.cos(x[0])],
                            [0, np.sin(x[0])]])

    #x = f(x,u)
    x = F @ x + (delta_t * B) @ u  #Calculate the new state
    x[0] = pi_2_pi(x[0]) # Convert the angle to be in [-pi,pi] range
     
    return x


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

    
def get_input(input_cov): 
    
    #input velocities
 
    u_v_t = np.ones(len(t)) * 5
    u_omega_t = np.ones(len(t)) * 0.1             #angular velocity
    
    #Noises
    W_v_t =(np.random.normal(0,np.sqrt(input_cov[1,1]),len(t)))
    W_omega_t = np.random.normal(0,np.sqrt(input_cov[0,0]),len(t))
    
    #Input + noise
    u_v_t = u_v_t + W_v_t
    u_omega_t = u_omega_t + W_omega_t
    
    return np.vstack((u_omega_t,u_v_t)).T            #To get the the inputs in pars for [u_v,u_w] at time t


# def get_input(input_cov):
    
#     #input velocities
#     max_velocity = 5
#     u_v_t = np.arange(0,max_velocity,max_velocity/len(t))
#     u_omega_t = np.ones(len(t)) * 0.1             #angular velocity
    
#     #Noises
#     W_v_t =(np.random.normal(0,np.sqrt(input_cov[1,1]),len(t)))
#     W_omega_t = np.random.normal(0,np.sqrt(input_cov[0,0]),len(t))
    
#     #Input + noise
#     u_v_t = u_v_t + W_v_t
#     u_omega_t = u_omega_t + W_omega_t
    
#     return np.vstack((u_omega_t,u_v_t)).T            #To get the the inputs in pars for [u_v,u_w] at time t


# In[4]:



print("start!")

My_robot_true = Agent(inital_state)
My_robot_DR   = Agent(inital_state)
My_robot_Est  = Agent(inital_state)


#Input velocities
U_DR = get_input(np.zeros([2,2]))
U_true = get_input(M)



#initialize the varibles
true_X = np.zeros((len(t),State_size))
DR_X   = np.zeros((len(t),State_size))
X_est  = np.zeros((len(t),State_size))

Z   = np.zeros((len(t),2))

P_est  = np.zeros((len(t),State_size,State_size))
P_est_trace = np.zeros((len(t),1))
P_est_det = np.zeros((len(t),1))

cov_t = list()
cov_list_trace = list()
cov_list_det = list()




for i in range(len(t)-1):
    
    
    true_X[i,:] = My_robot_true.state_update(U_true[i])
    DR_X[i,:]   = My_robot_DR.state_update(U_DR[i])


    #Ekf:
    #State predict
    X_est[i,:], P_est[i,:,:] = My_robot_Est.Time_update(U_DR[i])         #time/State Estimate update
    
    cov_t = np.append(cov_t,t[i])
    cov_list_trace = np.append(cov_list_trace, np.sqrt(np.trace(P_est[i,:,:])))
    cov_list_det = np.append(cov_list_det,np.sqrt(np.linalg.det(P_est[i,:,:])))
    

    if(i%1== 0):
        #calc ovservation to get z
        Z[i] = My_robot_true.observe(landmark1,R)  #Observation from the true trajectory to the lnadmrk 
                                                   #then observ. noise added here
        #Observation Update
        X_est[i,:], P_est[i,:,:], y_res = My_robot_Est.Observation_update(landmark1, Z[i]) #Observation estimate update

        cov_t = np.append(cov_t,t[i])
        cov_list_trace = np.append(cov_list_trace, np.sqrt(np.trace(P_est[i,:,:])))
        cov_list_det = np.append(cov_list_det,np.sqrt(np.linalg.det(P_est[i,:,:])))
       
    



#Calculate the the difrenses of estimation and dead reckoning here:
d_tr_DR  = np.sqrt((true_X[:,1]-DR_X[:,1])**2 + (true_X[:,2]-DR_X[:,2])**2)
theta_tr_dr = 1 - np.cos(true_X[:,0] - DR_X[:,0])

d_tr_est = np.sqrt((true_X[:,1]-X_est[:,1])**2 + (true_X[:,2]-X_est[:,2])**2)
theta_tr_est = 1- np.cos(true_X[:,0]-X_est[:,0])

print("Done!")


# In[ ]:





# In[5]:


#Plots 
#Plot of the trajectories 
fig,ax1 = plt.subplots(figsize=(10, 10))

ax1.plot(true_X[0:-1,1],true_X[0:-1,2],label = 'True trajectory') 
ax1.plot(DR_X[0:-1,1],DR_X[0:-1,2], label = 'Dead Reckoning')
ax1.plot(X_est[0:-1,1],X_est[0:-1,2], label = 'State Estiamte')

# #Plot the direction vectors
#coef = 3
#freq = 20
#ax1.quiver(true_X[0:999:freq,1], true_X[0:999:freq,2], coef * np.cos(true_X[0:999:freq,0]), coef *np.sin(true_X[0:999:freq,0]))
#ax1.quiver(true_X[0:999:freq,1], true_X[0:999:freq,2], Z[0:999:freq,1] * np.cos(Z[0:999:freq,0]), Z[0:999:freq,1] * np.sin(Z[0:999:freq,0]))
#ax1.quiver(DR_X[0:999:freq,1], DR_X[0:999:freq,2], coef * np.cos(DR_X[0:999:freq,0]), coef *np.sin(DR_X[0:999:freq,0]))
#ax1.quiver(X_est[0:999:freq,1], X_est[0:999:freq,2], coef * np.cos(X_est[0:999:freq,0]), coef *np.sin(X_est[0:999:freq,0]))

ax1.scatter(landmark1[0],landmark1[1])
ax1.title.set_text('Trajectories')
ax1.legend()
ax1.set_xlabel('x1')
ax1.set_ylabel ('x2')


# In[ ]:





# In[6]:


#Plot of disarnce diffrences
fig, ax2 = plt.subplots(figsize=(7, 7))
ax2.plot(t, d_tr_DR, label ='||tr-DR||')
ax2.plot(t, d_tr_est, label = '||tr-EST||')


ax2.title.set_text('Diffrnece of DR and Est trajectories from true trajectory')
ax2.legend()
ax2.set_xlabel('t')
ax2.set_ylabel ('diffrence')

#Plot of angle diffrences 
fig, ax3 = plt.subplots(figsize=(7, 7))
ax3.plot(t, theta_tr_dr, label ='angle(tr-DR)')
ax3.plot(t, theta_tr_est, label = 'angle(tr - EST)')
ax3.legend()
ax3.set_xlabel('t')
ax3.set_ylabel ('Angle_diffrence (rad)')
ax3.title.set_text('Diffrnece of DR and estimate angles form true angle')
plt.tight_layout() 
plt.show()


# In[7]:


#Plots of trace and determinant of P_est
beginig_time = 8
last_time = -1



fig, ax4 = plt.subplots(figsize = (10,10))
ax4.plot(cov_t[beginig_time:last_time], cov_list_trace[beginig_time:last_time])

ax4.set_xlabel('t')
ax4.set_ylabel('P_est_trace')
ax4.grid(True)
plt.tight_layout()
plt.show()

fig, ax5 = plt.subplots(figsize = (10,10))
ax5.plot(cov_t[beginig_time:last_time], cov_list_det[beginig_time:last_time])

ax5.set_xlabel('t')
ax5.set_ylabel('P_est_determinant')
ax5.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[8]:


# x = np.arange(0,np.pi/3,np.pi/12)
# u = 5
# F = delta_t * np.array([[1,                    0, 0], 
#                         [-u * np.sin(x), 1, 0],
#                         [u * np.cos(x),  0, 1]])



# plt.plot(np.rad2deg(x))


# In[9]:


Q = np.array([[ 0.1, 0, 0],
                     [ 0,  0.1, -0.1],
                     [ 0, -0.1,  0.01]])
   


# In[ ]:





# In[ ]:




