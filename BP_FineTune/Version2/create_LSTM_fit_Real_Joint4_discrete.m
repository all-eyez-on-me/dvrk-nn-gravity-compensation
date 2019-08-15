load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_pos.mat');
load('./data/Real_Joint4_10Reps/Real_Joint4_10Reps_tor.mat');
train_input_mat = input_mat(4,:);
train_output_mat = output_mat(4,:);

load('./data/Real_traj_test_10/MTMR_28002_traj_test_10_pos.mat');
load('./data/Real_traj_test_10/MTMR_28002_traj_test_10_tor.mat');
test_input_mat = input_mat(1:6,:);
test_output_mat = output_mat(1:6,:);


fixWindowLength = 10;
numHiddenUnits = 200;
maxEpochs = 200;
miniBatchSize = 20;

train_input_cell = {};
train_output_cell = {};
test_input_cell = {};
test_output_cell = {};

% cut data into cells with fix window
for i = 1:size(train_input_mat,2)-fixWindowLength+1
    train_input_cell = vertcat(train_input_cell, {train_input_mat(:,i:i+fixWindowLength-1)});
    train_output_cell = vertcat(train_output_cell, {train_output_mat(:,i:i+fixWindowLength-1)});
end

% cut data into cells with fix window
test_input_temp =zeros(6,fixWindowLength);
test_output_temp =zeros(6,fixWindowLength);
for i = 1:size(test_input_mat,2)
    for j = 1:fixWindowLength
        test_input_temp(:,j) = test_input_mat(:,i);
        test_output_temp(:,j) = test_output_mat(:,i);
    end
    test_input_cell = vertcat(test_input_cell, test_input_temp);
    test_output_cell = vertcat(test_output_cell, test_output_temp);
end

% data pre-process
mu = mean([train_input_cell{:}],2);
sig = std([train_input_cell{:}],0,2);

for i = 1:numel(train_input_cell)
    train_input_cell{i} = (train_input_cell{i} - mu) ./ sig;
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
% data pre-process
YPred = predict(net, train_input_cell,'MiniBatchSize',1);
y_mat = [];
for i = 1:numel(YPred)
    y_mat = [y_mat, YPred{i}(:,end)];
end

x= train_input_mat;
y_predit = y_mat;
y_measure = train_output_mat;


sz = 10;
figure;
hold on
scatter(x,y_measure,sz,'k', 'filled');
plot(x(fixWindowLength:end),y_predit);
hold off
% save ./model/CAD/CAD_10_e5_ffnn_model.mat net tr


