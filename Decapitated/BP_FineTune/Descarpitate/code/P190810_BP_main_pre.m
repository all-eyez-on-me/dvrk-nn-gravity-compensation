clear;clc;
close all;
format long;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 读取数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataFeature = load('input data_source domain.mat');
Feature = dataFeature.input_mat;
feature = Feature';
feature = feature(1:end,:);
dataLabel = load('output data_source domain.mat');
Label = dataLabel.output_mat;
label = Label';
label = label(1:end,2);
% if exist('data.mat')
%     data = load('data.mat');
%     label = data.newLabel';
% end
% 随机划分训练测试集
[m,n] = size(feature);
K = randperm(m);
trainSize = int32(0.8*m);
trainFeature = feature((1:trainSize),:);
trainLabel = label((1:trainSize),:);
testFeature = feature((trainSize+1:end),:);
testLabel = label((trainSize+1:end),:);

[trainFeature,rule1] = mapminmax(trainFeature');
testFeature = mapminmax('apply',testFeature',rule1);
[trainLabel,rule2] = mapminmax(trainLabel');
testLabel = mapminmax('apply',testLabel',rule2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RBF神经网络训练
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = newff(trainFeature,trainLabel,10);
% net.trainParam.goal = 1e-10;
% net.trainParam.epochs = 100;
% net.trainParam.lr = 0.05;
% net.trainParam.showWindow = 1;
net = train(net,trainFeature,trainLabel);
% net = newrbe(trainFeature,trainLabel,50);
% 训练集
predicted_trainLabel = net(trainFeature);
errors = predicted_trainLabel - trainLabel;
trainLabel = mapminmax('reverse',trainLabel,rule2);
predicted_trainLabel = mapminmax('reverse',predicted_trainLabel,rule2);
% figure;
% plot(predicted_trainLabel,'ro-'),hold on;
% plot(trainLabel,'g*-')
% legend('预测值','实际值')
% title('RBF神经网络训练集预测值与实际值对比')
for i=1:1
    [MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(trainLabel(i,:),predicted_trainLabel(i,:));
    strLog = sprintf('[训练集][输出%d]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',i,MSE,RMSE,R2,MAE,MAPE,RPD);
    disp(strLog)
    
    error = trainLabel(i,:)-predicted_trainLabel(i,:);
    error = error.^2;
    tempLabel = trainLabel(i,:).^2;
    output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
    strLog = sprintf('[训练集][输出%d]error=%f',i,output);
    disp(strLog)
end
% 测试集
predicted_testLabel = net(testFeature);
errors = predicted_testLabel - testLabel;
testLabel = mapminmax('reverse',testLabel,rule2);
predicted_testLabel = mapminmax('reverse',predicted_testLabel,rule2);
% figure;
% plot(predicted_testLabel,'ro-'),hold on;
% plot(testLabel,'g*-')
% legend('预测值','实际值')
% title('RBF神经网络测试集预测值与实际值对比')
for i=1:1
    [MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(testLabel(i,:),predicted_testLabel(i,:));
    strLog = sprintf('[测试集][输出%d]MSE=%f,RMSE=%f,R2=%f,MAE=%f,MAPE=%f,RPD=%f',i,MSE,RMSE,R2,MAE,MAPE,RPD);
    disp(strLog)
    
    error = testLabel(i,:)-predicted_testLabel(i,:);
    error = error.^2;
    tempLabel = testLabel(i,:).^2;
    output = sqrt(sum(error(:)))/sqrt(sum(tempLabel(:)));
    strLog = sprintf('[测试集][输出%d]error=%f',i,output);
    disp(strLog)
end
% if ~exist('data.mat')
%     newLabel = [predicted_trainLabel predicted_testLabel];
%     save data.mat newLabel;
% end
save data.mat rule1 rule2 net;

