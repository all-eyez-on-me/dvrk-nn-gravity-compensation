clear;clc;
close all;
format long;

data = load('data1.mat');
net = data.net;
% rule1 = data.rule1;
% rule2 = data.rule2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 读取数据并构造数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataFeature = load('train_input data_target domain.mat');
trainFeature = dataFeature.input_mat;
trainFeature = trainFeature';
dataLabel = load('train_output data_target domain.mat');
trainLabel = dataLabel.output_mat;
trainLabel = trainLabel';
trainLabel = trainLabel(:,3);

dataFeature = load('test_input data_source domain.mat');
testFeature = dataFeature.input_mat;
testFeature = testFeature';
dataLabel = load('test_output data_source domain.mat');
testLabel = dataLabel.output_mat;
testLabel = testLabel';
testLabel = testLabel(:,3);

trainFeature = trainFeature';
trainLabel = trainLabel';
testFeature = testFeature';
testLabel = testLabel';

%归一化
% % [trainFeature,rule1] = mapminmax(trainFeature);
% trainFeature = mapminmax('apply',trainFeature,rule1);
% testFeature = mapminmax('apply',testFeature,rule1);
% % [trainLabel,rule2] = mapminmax(trainLabel);
% trainLabel = mapminmax('apply',trainLabel,rule2);
% testLabel = mapminmax('apply',testLabel,rule2);

XTrain = trainFeature;
YTrain = trainLabel;
XTest = testFeature;
YTest = testLabel;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 定义LSTM网络架构
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 定义迁移学习网络
layersTransfer = net.Layers(1:end-2);

inputSize = 6;
numResponses = 1;
numHiddenUnits = 3;

%定义新网络
layers = [ ...
    layersTransfer
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% Specify the training options. Set the solver to 'adam' and train for 250 epochs. To prevent the gradients from exploding, set the gradient threshold to 1. Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs by multiplying by a factor of 0.2.
opts = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 训练LSTM网络
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the LSTM network with the specified training options by using trainNetwork.
net = trainNetwork(XTrain,YTrain,layers,opts);
% 
% save data1.mat net rule1 rule2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对训练集进行预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTrain);
rmse = sqrt(mean((YPred-YTrain).^2));
% 反归一化
% trainLabel = mapminmax('reverse',trainLabel,rule2);
% predict_trainLabel = mapminmax('reverse',YPred,rule2);
predict_trainLabel = YPred;
% 训练集回归指标分析
[MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(trainLabel,predict_trainLabel);
strLog = sprintf('[LSTM迁移学习训练集]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',MSE,RMSE,R2,MAE,MAPE,RPD);
disp(strLog);
error = trainLabel(1,:)-predict_trainLabel(1,:);
error = error.^2;
tempLabel = trainLabel(1,:).^2;
output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
strLog = sprintf('[迁移学习训练集][输出%d]error=%f',1,output);
disp(strLog)
% 作图
figure;
plot(trainLabel(1:1000),'r*-')
hold on;
plot(predict_trainLabel(1:1000),'bo-')
legend('实际值','预测值')
title('[迁移学习]LSTM训练集实际值与预测值的对比')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对测试集进行预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTest);
rmse = sqrt(mean((YPred-YTest).^2));
% 反归一化
% testLabel = mapminmax('reverse',testLabel,rule2);
% predict_testLabel = mapminmax('reverse',YPred,rule2);
predict_testLabel = YPred;
% 训练集回归指标分析
[MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(testLabel,predict_testLabel);
strLog = sprintf('[LSTM测试集]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',MSE,RMSE,R2,MAE,MAPE,RPD);
disp(strLog);
error = testLabel(1,:)-predict_testLabel(1,:);
error = error.^2;
tempLabel = testLabel(1,:).^2;
output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
strLog = sprintf('[测试集][输出%d]error=%f',1,output);
disp(strLog)
% 作图
figure;
plot(testLabel(1:100),'r*-')
hold on;
plot(predict_testLabel(1:100),'bo-')
legend('实际值','预测值')
title('[迁移学习]LSTM测试集实际值与预测值的对比')
