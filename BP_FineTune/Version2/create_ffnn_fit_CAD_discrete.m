is_useParallelCPU = false;
is_useParallelGPU = true;

load('./data/CAD_sim_1e6/CAD_sim_1e6_pos.mat');
load('./data/CAD_sim_1e6/CAD_sim_1e6_tor.mat');
train_input_mat = input_mat(1:6,:);
train_output_mat = output_mat(1:6,:);

load('./data/Real_traj_test_10/MTMR_28002_traj_test_10_pos.mat');
load('./data/Real_traj_test_10/MTMR_28002_traj_test_10_tor.mat');
test_input_mat = input_mat(1:6,:);
test_output_mat = output_mat(1:6,:);

% config training network
goal = 1e-6;
showWindow = true;
showCommandLine = true;
max_fail = 100;

% config network
optimal_neuron_num = 20; %number of hidden neurons
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

if(is_useParallelCPU)
    pool = parpool
end
if(is_useParallelGPU)
    gpu1 = gpuDevice(1)
end

train_perform_list = [];
validation_perform_list = [];
test_perform_list = [];
net_list = {}; %cell list with network for each joint
tr_stop_list = {};%cell list with stop train result for each joint
hiddenLayerSize = optimal_neuron_num*[1,1];

abs_RMS_vec = [];
rel_RMS_vec = [];

for i = 1:6
%for i = 1:size(train_input_mat,1)
    net = fitnet(hiddenLayerSize,trainFcn);

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    if(is_useParallelGPU)
        net.input.processFcns = {'mapminmax'};
        net.output.processFcns = {'mapminmax'};
    else
        net.input.processFcns = {'removeconstantrows','mapminmax'};
        net.output.processFcns = {'removeconstantrows','mapminmax'};
    end

    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivision
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

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

    net = configure(net, train_input_mat, train_output_mat(i,:));
   
    net = init(net);
    
    if(is_useParallelCPU)
        [net,tr] = train(net,train_input_mat, train_output_mat(i,:),'useParallel','yes');
    elseif(is_useParallelGPU)
        [net,tr] = train(net,train_input_mat, train_output_mat(i,:),'useGPU','yes');
    else
        [net,tr] = train(net,train_input_mat, train_output_mat(i,:));
    end

    tr_stop_list{end+1} = tr.stop; 
    net_list{end+1} = net;
    
    % test
    y = net(test_input_mat);
    e = gsubtract(test_output_mat(i,:),y);
    abs_RMS = sqrt(sum(e.^2));
    rel_RMS = abs_RMS/sqrt(sum(test_output_mat(i,:).^2));
    abs_RMS_vec = [abs_RMS_vec;abs_RMS];
    rel_RMS_vec = [rel_RMS_vec;rel_RMS];
end
if(is_useParallelCPU)
    delete(gcp('nocreate'))
end
%save source_domain_CAD_10_e5_ffnn_model.mat net_list tr_stop_list

