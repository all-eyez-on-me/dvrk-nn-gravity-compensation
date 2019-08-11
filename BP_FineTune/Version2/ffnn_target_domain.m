load('Real_MTMR_pos_4096.mat');
load('Real_MTMR_tor_4096.mat');
input_mat = input_mat(1:6,:);
output_mat = output_mat(1:6,:);

% config training network
goal = 1e-6;
showWindow = true;
showCommandLine = true;
max_fail = 100;

% config network
optimal_neuron_num = 20; %number of hidden neurons
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.



train_perform_list = [];
validation_perform_list = [];
test_perform_list = [];
net_list = {}; %cell list with network for each joint
tr_stop_list = {};%cell list with stop train result for each joint
hiddenLayerSize = optimal_neuron_num;

joint_num = size(input_mat,1);
for i = 1:joint_num
    net = fitnet(hiddenLayerSize,trainFcn);

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivision
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error

    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate', ...
        'plotregression'};

    % Train the Network
    net.trainParam.goal = goal;
    net.trainParam.showWindow = showWindow;
    net.trainParam.showCommandLine = showCommandLine;
    net.trainParam.max_fail = max_fail;

    x_input = input_mat;
    t_input = output_mat(i, :);
    net = configure(net, x_input, t_input);
   
    net = init(net);
    [net,tr] = train(net,x_input,t_input);

    tr_stop_list{end+1} = tr.stop; 
    net_list{end+1} = net;
end

save target_domain_Real_MTMR_4096_model.mat net_list tr_stop_list

