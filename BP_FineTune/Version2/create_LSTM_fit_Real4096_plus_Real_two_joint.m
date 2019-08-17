train_input_cell = {};
train_output_cell = {};
test_input_cell = {};
test_output_cell = {};
fixWindowLength = 3;
numHiddenUnits = 100;
maxEpochs = 3000;
miniBatchSize = 2000;


[input_cell, output_cell] = load_data('./data/Real_all_couple_4096/Real_MTMR_pos_4096.mat',...
                                      './data/Real_all_couple_4096/Real_MTMR_tor_4096.mat',...
                                       fixWindowLength,...
                                       'repeat');
 train_input_cell = vertcat(train_input_cell,input_cell);
 train_output_cell = vertcat(train_output_cell,output_cell);
 
 [input_cell, output_cell] = load_data('./data/Real_all_couple_4096/Real_MTMR_pos_4096_reverse.mat',...
                                      './data/Real_all_couple_4096/Real_MTMR_tor_4096_reverse.mat',...
                                       fixWindowLength,...
                                       'repeat');
 train_input_cell = vertcat(train_input_cell,input_cell);
 train_output_cell = vertcat(train_output_cell,output_cell);

 [input_cell, output_cell] = load_data('./data/Real_two_joint_moving_6286/Real_two_joint_moving_6286_pos.mat',...
                                      './data/Real_two_joint_moving_6286/Real_two_joint_moving_6286_tor.mat',...
                                       fixWindowLength,...
                                       'order');
 train_input_cell = vertcat(train_input_cell,input_cell);
 train_output_cell = vertcat(train_output_cell,output_cell);

 [input_cell, output_cell] = load_data('./data/Real_traj_test_10/MTMR_28002_traj_test_10_pos.mat',...
                                      './data/Real_traj_test_10/MTMR_28002_traj_test_10_tor.mat',...
                                       fixWindowLength,...
                                       'repeat');
 test_input_cell = vertcat(test_input_cell,input_cell);
 test_output_cell = vertcat(test_output_cell,output_cell);


% data pre-process for input
mu_input = mean([train_input_cell{:}],2);
sig_input = std([train_input_cell{:}],0,2);
for i = 1:numel(train_input_cell)
    train_input_cell{i} = (train_input_cell{i} - mu_input) ./ sig_input;
end
% data pre-process for output
mu_output = mean([train_output_cell{:}],2);
sig_output = std([train_output_cell{:}],0,2);
for i = 1:numel(train_output_cell)
    train_output_cell{i} = (train_output_cell{i} - mu_output) ./ sig_output;
end

% define LSTM architechture
numFeatures = size(train_input_cell{1},1);
numResponses = size(train_output_cell{1},1);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

% train LSTM
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);
net = trainNetwork(train_input_cell,train_output_cell,layers,options);


%%
%test
for i = 1:numel(test_input_cell)
    test_input_cell{i} = (test_input_cell{i} - mu_input) ./ sig_input;
end

YPred = predict(net, test_input_cell,'MiniBatchSize',1);
y_mat = []
for i = 1:numel(YPred)
    y_mat = [y_mat, YPred{i}(:,end).*sig_output+mu_output];
end
test_output_mat = [];
for i=1:1:numel(test_output_cell)
    test_output_mat = [test_output_mat test_output_cell{i}(:,end)]
end
[abs_RMS_vec, rel_RMS_vec] = RMS(test_output_mat, y_mat)


%save LSTM_fit_4096_dual_add_mlse4pol_sim.mat net fixWindowLength mu_input mu_output sig_input sig_output
