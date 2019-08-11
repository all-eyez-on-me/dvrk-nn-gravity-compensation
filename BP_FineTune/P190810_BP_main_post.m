clear;clc;
close all;
format long;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 读取数据并构造数据集
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataFeature = load('train_data_target domain.mat');
trainData = dataFeature.input_mat;
trainData = trainData';

[m,n] = size(trainData);
trainData = trainData(randperm(m),:);

[m,n] = size(trainData);
Index = 1;
trainFeature = trainData(1:m-100,1:n/2);
trainLabel = trainData(1:m-100,n/2+Index);
testFeature = trainData(m-99:m,1:n/2);
testLabel = trainData(m-99:m,n/2+Index);

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
%% BP神经网络训练
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('data.mat');
net1 = data.net;

net = newff(trainFeature,trainLabel,10);
%网络权值赋值
net.iw{1,1} = net1.iw{1,1};
net.lw{2,1} = net1.lw{2,1};
net.b{1} = net1.b{1};
net.b{2} = net1.b{2};
% net.trainParam.goal = 1e-10;
% net.trainParam.epochs = 100;
% net.trainParam.lr = 0.05;
% net.trainParam.showWindow = 1;
net = train(net,trainFeature,trainLabel);
% net = newrbe(trainFeature,trainLabel,50);

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
% save data.mat rule1 rule2 net;

