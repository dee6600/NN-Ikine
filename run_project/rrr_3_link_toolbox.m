clear
%%% PROGRAM FOR RRR ROBOTIC ARM USING IKINE FUNCTION
%prompt for user to get end effector pose
prompt = 'Enter end Effector X,Y,Z : ';
EE_user1 = input(prompt)

% transformation matrix
T = [1 0 0 EE_user1(1);0 1 0 EE_user1(2); 0 0 1 EE_user1(3); 0 0 0 1]

q0 = [0 0 0];
% extracting each x y z seperatly
x = EE_user1(1)
y = EE_user1(2)
z = EE_user1(3)

% predifining initial values and pose
d2 = 10
d3 = 10
theta1 = 0
theta2 = 0
theta3 = 0

%defining robot discription links
L(1) = Link([theta1,0,0,pi/2,0], 'standard');
L(2) = Link([theta2,0,d2,0,0], 'standard');
L(3) = Link([theta3,0,d3,0,0],'standard');
threelink = SerialLink(L, 'name', 'three link');

% call for inverse kinematics functions from robotics toolbox
IK_TOOLBOX = threelink.ikine(T,q0,'mask',[1 1 1 0 0 0],'ilimit',1000)

% computed plot parameters
q1=rad2deg(IK_TOOLBOX(1))  
q2=rad2deg(IK_TOOLBOX(2))
q3=rad2deg(IK_TOOLBOX(3))

% initial plot parameters
threelink.plot([1,1,1])
view(45,30)
%threelink.teach()

%Animating Arm Movement/ Transformations
try 
for qi1 = deg2rad(0:5:q1)
    threelink.animate([qi1,0,0])
end
for qi2 = deg2rad(0:5:q2)
    threelink.animate([qi1,qi2,0])
end
for qi3 = deg2rad(0:5:q3)
    threelink.animate([qi1,qi2,qi3])
end

catch ME
    fprintf('Singularity POSE , Cannot Reach These Cordinate!!!')
    f = msgbox( 'Singularity POSE , Cannot Reach These Cordinate!!!')
    close all
end


%standard input [1 4 9]



