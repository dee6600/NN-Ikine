import numpy as np
from numpy.core.fromnumeric import _squeeze_dispatcher
import pandas as pd
import math 


#create dataframe file with header x,y,z,q1,q2,q3
df_new = pd.DataFrame(columns=['x','y','z','q1','q2','q3'])

#generate random number x
x_val = np.random.uniform(low=-200, high=0, size=10000)
y_val = np.random.uniform(low=0, high=200, size=10000)
z_val = np.random.uniform(low=0, high=200, size=10000)


df_new['x'] = x_val
df_new['y'] = y_val
df_new['z'] = z_val

#inverse kinematics
def inv_kinematics(x,y,z):
    d1 = 5
    q1 = np.arctan2(y,-x)
    q2 = z-d1
    q3 = math.sqrt(x**2+y**2)
    return q1,q2,q3

q1_list=[]
q2_list=[]
q3_list=[]

for x,y,z in zip(x_val,y_val,z_val):
    q1,q2,q3 = inv_kinematics(x,y,z)
    q1_list.append(q1)
    q2_list.append(q2)
    q3_list.append(q3)

df_new['q1'] = q1_list
df_new['q2'] = q2_list
df_new['q3'] = q3_list

print(df_new)

#save df to csv file
df_new.to_csv('train10000test.csv',index=False)