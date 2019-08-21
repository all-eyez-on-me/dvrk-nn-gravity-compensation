clear;clc;
close all;
format long;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 读取数据并构造数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataFeature = load('input data_source domain.mat');
Feature = dataFeature.input_mat;
Feature = Feature';
dataLabel = load('output data_source domain.mat');
Label = dataLabel.output_mat;
Label = Label';
% Label = Label(:,2);

Feature = Feature(1:3000,:);
Label = Label(1:3000,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 随机划分训练测试集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m,n] = size(Feature);
K = randperm(m);
trainSize = int32(0.8*m);
trainFeature = Feature(K(1:trainSize),:);
trainLabel = Label(K(1:trainSize),:);
testFeature = Feature(K(trainSize+1:end),:);
testLabel = Label(K(trainSize+1:end),:);

trainFeature = trainFeature';
trainLabel = trainLabel';
testFeature = testFeature';
testLabel = testLabel';

%归一化
[trainFeature,rule1] = mapminmax(trainFeature);
testFeature = mapminmax('apply',testFeature,rule1);
[trainLabel,rule2] = mapminmax(trainLabel);
testLabel = mapminmax('apply',testLabel,rule2);

XTrain = trainFeature;
YTrain = trainLabel;
XTest = testFeature;
YTest = testLabel;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 定义LSTM网络架构
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputSize = 6;
numResponses = 6;
numHiddenUnits = 6;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% Specify the training options. Set the solver to 'adam' and train for 250 epochs. To prevent the gradients from exploding, set the gradient threshold to 1. Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs by multiplying by a factor of 0.2.
opts = trainingOptions('adam', ...
    'MaxEpochs',100, ...
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

save data1.mat net rule1 rule2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对训练集进行预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTrain);
rmse = sqrt(mean((YPred-YTrain).^2));
% 反归一化
trainLabel = mapminmax('reverse',trainLabel,rule2);
predict_trainLabel = mapminmax('reverse',YPred,rule2);
% 训练集回归指标分析
[MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(trainLabel,predict_trainLabel);
strLog = sprintf('[LSTM训练集]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',MSE,RMSE,R2,MAE,MAPE,RPD);
disp(strLog);
error = trainLabel(1,:)-predict_trainLabel(1,:);
error = error.^2;
tempLabel = trainLabel(1,:).^2;
output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
strLog = sprintf('[训练集][输出%d]error=%f',1,output);
disp(strLog)
% 作图
figure;
plot(trainLabel(1:1000),'r*-')
hold on;
plot(predict_trainLabel(1:1000),'bo-')
legend('实际值','预测值')
title('LSTM训练集实际值与预测值的对比')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 对测试集进行预测
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTest);
rmse = sqrt(mean((YPred-YTest).^2));
% 反归一化
testLabel = mapminmax('reverse',testLabel,rule2);
predict_testLabel = mapminmax('reverse',YPred,rule2);
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
title('LSTM训练集实际值与预测值的对比')
