clear

%%% PROGRAM FOR RPP ROBOTIC ARM USING IKINE FUNCTION
%prompt for user to get end effector pose

% run requrement file 
system('pip install -r requirements.txt');

%prompt for user to get end effector pose
prompt = 'Enter end Effector X,Y,Z';
EE_user1 = input(prompt)

% transformation matrix
T = [1 0 0 EE_user1(1);0 1 0 EE_user1(2); 0 0 1 EE_user1(3); 0 0 0 1]

q0 = [0 0 0];
% extracting each x y z seperatly
x = EE_user1(1)
y = EE_user1(2)
z = EE_user1(3)
% predifining initial values and pose
theta1 = deg2rad(30)
d1 = 5
d2 = 5
d3 = 5
%defining robot discription links
L(1) = Link([theta1,d1,0,0,0], 'standard');
L(2) = Link([0,d2,0,-pi/2,1], 'standard');
L(2).qlim = [1,200]
L(3) = Link([0,d3,0,0,1], 'standard');
L(3).qlim = [1,200]
threelink = SerialLink(L, 'name', 'my_RPP');

%write EEpose to file
csvwrite('dataset/EE_pose_rpp.csv',EE_user1)
%run py_rrr.py
system('python py_rpp.py');
%read JS_pos from file
JS_pos = csvread('dataset/JS_pos_rpp.csv')
csvwrite('dataset/JS_pos_rpp.csv',q0)
% predicted plot parameters
q1=rad2deg(JS_pos(1))  %test with 
q2=JS_pos(2)
q3=JS_pos(3)
% initial plot parameters
threelink.plot([q1,q2,q3])
view(45,30)
%threelink.teach()

%Animating Arm Movement/ Transformations
try
for qi1 = deg2rad(0:100:q1)
    threelink.animate([qi1,0,0])
end

for qi2 = (0:10:q2)
    threelink.animate([1,qi2,0])
end

for qi3 =(5:10:q3)
    threelink.animate([1,qi2,qi3])
end

catch ME
    fprintf('Singularity POSE , Cannot Reach These Cordinate!!!')
    f = msgbox( 'Singularity POSE , Cannot Reach These Cordinate!!!')
    close all
end


%standard input [200 200 100]



