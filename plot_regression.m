%init_pos = [0 0 10 0 45 0 0];


plot_joint_num = 6;

joint_pos_upper_limit = [30,45,34,190,175,40];
joint_pos_lower_limit = [-30,-14,-34,-80,-85,-40];

delta_angle = 1;

x = joint_pos_lower_limit(plot_joint_num):delta_angle:joint_pos_upper_limit(plot_joint_num);

y = zeros(size(x));
y2 = zeros(size(x));
%net =  net_list{plot_joint_num};
for i=1:size(x,2)
    config = deg2rad(init_pos);
    config(plot_joint_num) = deg2rad(x(i));
    output = CAD_model(config.');
    output2 = net(config(1:6).');
    y(:,i) = output(plot_joint_num);
    y2(:,i) = output2;
end

hold on
plot(x,y)
plot(x,y2)
hold off


