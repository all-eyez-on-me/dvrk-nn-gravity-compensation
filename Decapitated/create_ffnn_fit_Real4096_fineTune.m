is_useParallelCPU = false;
is_useParallelGPU = true;

load('./data/Real_all_couple_4096/Real_MTMR_pos_4096.mat');
load('./data/Real_all_couple_4096/Real_MTMR_tor_4096.mat');
train_input_mat = input_mat(1:6,:);
train_output_mat = output_mat(1:6,:);

load('./model/CAD/CAD_10_e5_ffnn_model.mat');
source_net = net;

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
optimal_neuron_num = 40; %number of hidden neurons
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

if(is_useParallelCPU)
     pool = parpool
end
if(is_useParallelGPU)
    gpu1 = gpuDevice(1)
end

hiddenLayerSize = optimal_neuron_num;

abs_RMS_vec = [];
rel_RMS_vec = [];


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
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression'};

% Train the Network
net.trainParam.goal = goal;
net.trainParam.showWindow = showWindow;
net.trainParam.showCommandLine = showCommandLine;
net.trainParam.max_fail = max_fail;

net = configure(net, train_input_mat, train_output_mat);

net = init(net);

net.IW{1,1} = source_net.IW{1,1};
net.IW{2,1} = source_net.IW{2,1};
net.b{1} = source_net.b{1};
net.b{2} = source_net.b{2};

if(is_useParallelCPU)
    [net,tr] = train(net,train_input_mat, train_output_mat,'useParallel','yes');
elseif(is_useParallelGPU)
    [net,tr] = train(net,train_input_mat, train_output_mat,'useGPU','yes');
else
    [net,tr] = train(net,train_input_mat, train_output_mat);
end


% test
y = net(test_input_mat);
[abs_RMS_vec, rel_RMS_vec] = RMS(test_output_mat, y)


if(is_useParallelCPU)
    delete(gcp('nocreate'))
end
save ./model/CAD/CAD_10_e5_ffnn_model.mat net tr

