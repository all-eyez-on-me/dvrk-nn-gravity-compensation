clear;clc;
close all;
format long;

data = load('data1.mat');
net = data.net;
% rule1 = data.rule1;
% rule2 = data.rule2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ��ȡ���ݲ��������ݼ�
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

%��һ��
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
% ����LSTM����ܹ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ����Ǩ��ѧϰ����
layersTransfer = net.Layers(1:end-2);

inputSize = 6;
numResponses = 1;
numHiddenUnits = 3;

%����������
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
% ѵ��LSTM����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the LSTM network with the specified training options by using trainNetwork.
net = trainNetwork(XTrain,YTrain,layers,opts);
% 
% save data1.mat net rule1 rule2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ѵ��������Ԥ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTrain);
rmse = sqrt(mean((YPred-YTrain).^2));
% ����һ��
% trainLabel = mapminmax('reverse',trainLabel,rule2);
% predict_trainLabel = mapminmax('reverse',YPred,rule2);
predict_trainLabel = YPred;
% ѵ�����ع�ָ�����
[MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(trainLabel,predict_trainLabel);
strLog = sprintf('[LSTMǨ��ѧϰѵ����]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',MSE,RMSE,R2,MAE,MAPE,RPD);
disp(strLog);
error = trainLabel(1,:)-predict_trainLabel(1,:);
error = error.^2;
tempLabel = trainLabel(1,:).^2;
output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
strLog = sprintf('[Ǩ��ѧϰѵ����][���%d]error=%f',1,output);
disp(strLog)
% ��ͼ
figure;
plot(trainLabel(1:1000),'r*-')
hold on;
plot(predict_trainLabel(1:1000),'bo-')
legend('ʵ��ֵ','Ԥ��ֵ')
title('[Ǩ��ѧϰ]LSTMѵ����ʵ��ֵ��Ԥ��ֵ�ĶԱ�')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �Բ��Լ�����Ԥ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[net,YPred] = predictAndUpdateState(net,XTest);
rmse = sqrt(mean((YPred-YTest).^2));
% ����һ��
% testLabel = mapminmax('reverse',testLabel,rule2);
% predict_testLabel = mapminmax('reverse',YPred,rule2);
predict_testLabel = YPred;
% ѵ�����ع�ָ�����
[MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(testLabel,predict_testLabel);
strLog = sprintf('[LSTM���Լ�]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',MSE,RMSE,R2,MAE,MAPE,RPD);
disp(strLog);
error = testLabel(1,:)-predict_testLabel(1,:);
error = error.^2;
tempLabel = testLabel(1,:).^2;
output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
strLog = sprintf('[���Լ�][���%d]error=%f',1,output);
disp(strLog)
% ��ͼ
figure;
plot(testLabel(1:100),'r*-')
hold on;
plot(predict_testLabel(1:100),'bo-')
legend('ʵ��ֵ','Ԥ��ֵ')
title('[Ǩ��ѧϰ]LSTM���Լ�ʵ��ֵ��Ԥ��ֵ�ĶԱ�')
