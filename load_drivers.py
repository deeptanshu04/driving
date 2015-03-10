import os
import glob
import csv
import math
import random
import numpy as np 
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from scipy.interpolate import UnivariateSpline


rnd1 = random.randrange(1,201) #add 1 to the required range

data_root = '/home/dexter/Documents/ML/Driving/drivers'
driver_id = random.choice(os.listdir(data_root))
file = str(rnd1)+'.csv'
#driver_id = str(180)
#file = '59.csv'

driver_path = os.path.join(data_root,driver_id)



print('Driver Selected  '  + driver_id+'  File Selected  ' + file)
#for file in os.listdir(driver_path):
#	if file.endswith('.csv'):
#		print(file)



def import_file(file_name):
	csv_f = csv.reader(open(file_name))
	a = []
	for row in csv_f:
		if row[0]!='x':
			#print(row)
			z = map(float,row)
			a.append(z)
	b = np.array(a)
	return b


aa = import_file(os.path.join(driver_path,file))
#print(str(len(aa)))

v_x = 3.6*np.gradient(aa[:,0])
v_y = 3.6*np.gradient(aa[:,1])
v_arr1 = []#np.vstack((v_x,v_y))

for i in range(len(aa)):
	#print('i=  '+str(i)+'   v_y  '+str(v_y[i]))
	v_arr1.append([v_x[i],v_y[i]])

va = (np.asarray(v_arr1))

mx = np.mean(va[:,0])
#print mx
sdx = np.std(va[:,0])
#print sdx
my = np.mean(va[:,1])
sdy = np.std(va[:,1])
std_thresh = 5
c=[e for e in va if ((mx- std_thresh*sdx<e[0]<mx+ std_thresh*sdx)&(my- std_thresh*sdy<e[1]<my+ std_thresh*sdy))]
v_arr = np.asarray(c)

print(len(v_arr))
#print(v_x)
t = np.arange(len(v_arr))
t1 = np.arange(len(v_x))
imat = [[1.0,0.0],[0.0,1.0]]
cvar = 20;
kf = KalmanFilter(transition_matrices = imat, observation_matrices = imat, transition_covariance = cvar*np.identity(2), observation_covariance = cvar*np.identity(2))
#kf = kf.em(v_x, n_iter=5)
(v_smooth, smoothed_state_covariances) = kf.smooth(v_arr)

di = (v_smooth[:,1]*v_smooth[:,1]+v_smooth[:,0]*v_smooth[:,0])**0.5
#Remove Stops
stop_thresh = 1;
v_stop = []


#Take care of case where velocity is zero only for a small while - try moving average filter on absolute value
for i in range(0,len(v_smooth)):
	if (di[i]>stop_thresh):
		v_stop.append([v_smooth[i,0],v_smooth[i,1]])

v_f = np.asarray(v_stop)
t2 = np.arange(len(v_f))

bf,af = butter(6,0.1)
vx_butter = filtfilt(bf,af,v_f[:,0])
vy_butter = filtfilt(bf,af,v_f[:,1])

tangent_vector = np.asarray(map(lambda x,y: [x/((x**2+y**2)**0.5),y/((x**2+y**2)**0.5)],vx_butter,vy_butter))
n_x = np.gradient(tangent_vector[:,0])
n_y = np.gradient(tangent_vector[:,1])
normal_vector = np.asarray(map(lambda x,y: [x/((x**2+y**2)**0.5),y/((x**2+y**2)**0.5)],n_x,n_y))

ax_smooth = np.gradient(vx_butter)# np.gradient(v_smooth[:,0])
ay_smooth = np.gradient(vy_butter)

t_vel = map(lambda x,y: (x**2+y**2)**0.5,vx_butter,vy_butter) #Velocity(Tangential)
t_acc = map(lambda x,y,z : x*z[0]+y*z[1],ax_smooth,ay_smooth,tangent_vector)
n_acc = map(lambda x,y,z : x*z[0]+y*z[1],ax_smooth,ay_smooth,normal_vector)

#print(len(tangent_vector),len(tangent_vector[0]))
#print(dd[:,1])
#print(t)
x_smooth = np.cumsum(v_f,0)


stop_ratio = (len(v_smooth)-len(v_f))*100.0/len(v_smooth)
print('Stop Ratio = '+str(stop_ratio))

plt.plot(t2,t_acc,'r',t2,n_acc,'g')
#,t2,v_f[:,0],'b.')#,t1,np.gradient(v_y),'g')#,'r',t,dd[:,1],'g')#,t,smoothed_state_means,'b')#,t,v_x,'g')#[:,0],'r.',t,vel_arr[:,1],'b-')
plt.show()

 
