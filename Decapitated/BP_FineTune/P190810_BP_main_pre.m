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
label = label(1:end,1);

[feature,rule1] = mapminmax(feature');
[label,rule2] = mapminmax(label');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RBF神经网络训练
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = newff(feature,label,10);
% net.trainParam.goal = 1e-10;
% net.trainParam.epochs = 100;
% net.trainParam.lr = 0.05;
% net.trainParam.showWindow = 1;
net = train(net,feature,label);
% net = newrbe(trainFeature,trainLabel,50);
% 训练集

save data.mat rule1 rule2 net;

