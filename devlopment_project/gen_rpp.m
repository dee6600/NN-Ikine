
min_dig = deg2rad(0)
max_dig = deg2rad(360)
min = 5
max = 200
x = (max_dig-min_dig).*rand(10000,1) + min_dig
y = (max-min).*rand(10000,1) + min
z = (max-min).*rand(10000,1) + min

EE_poses = [x y z]


q0 = [0 0 0]

theta1 = 0
d1 = 5
d2 = 0
d3 = 0

L(1) = Link([theta1,d1,0,0,0], 'standard');
L(2) = Link([0,d2,0,-pi/2,1], 'standard');
L(2).qlim = [1,200]
L(3) = Link([0,d3,0,0,1], 'standard');
L(3).qlim = [1,200]

threelink = SerialLink(L, 'name', 'my_RPP');
IK_result = []
% for i=1:length(EE_poses)
for i=1:10000
    
    ix = EE_poses(i,1)
    iy = EE_poses(i,2)
    iz = EE_poses(i,3)

    T = [1 0 0 ix;0 1 0 iy; 0 0 1 iz; 0 0 0 1]
    
    
    IK_TOOLBOX = threelink.ikine(T,q0,'mask',[1 1 1 0 0 0],'ilimit',1000);
    if isempty(IK_TOOLBOX) == 0
        IK_result = [IK_result;ix iy iz IK_TOOLBOX]
    end    
    
%      try
%      IK_TOOLBOX = threelink.ikine(T,q0,'mask',[1 1 1 0 0 0],'ilimit',1000);
%      if IK_TOOLBOX ~= []
%          IK_result = [IK_result;IK_TOOLBOX]
%      end
%      catch ME
%          fprintf('error');
%          continue;
%          
%      end
                 
end

csvwrite('train_rpp_rad.csv',IK_result)
    