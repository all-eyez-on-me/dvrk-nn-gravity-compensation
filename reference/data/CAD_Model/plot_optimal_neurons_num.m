% Solve an Input-Output Fitting problem with a Neural Network
% Script generated by Neural Fitting app
% Created 03-Aug-2019 08:28:09
%
% This script assumes these variables are defined:
%
%   input_mat - input data.
%   output_mat_1 - target data.
pool = parpool
x = input_mat;
t = output_mat(5,:);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

train_perform_list = [];
validation_perform_list = [];
test_perform_list = [];
tr_stop_list = {};
for i=1:20
    % Create a Fitting Network
    hiddenLayerSize = i;
    net = fitnet(hiddenLayerSize,trainFcn);

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

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
     net.trainParam.goal = 1e-10;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;
    net.trainParam.max_fail = 4;
    [net,tr] = train(net,x,t,'useParallel','yes');

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y)

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y)
    train_perform_list = [train_perform_list, trainPerformance];
    valPerformance = perform(net,valTargets,y)
    validation_perform_list = [validation_perform_list,valPerformance];
    testPerformance = perform(net,testTargets,y)
    test_perform_list = [test_perform_list, testPerformance];
    tr_stop_list{end+1} = tr.stop; 
end


%%

figure
hold on
y_lim_max = max(test_perform_list)*1.2;
plot(1:size(test_perform_list,2), test_perform_list,'-ok',...
'LineWidth',2,...
'MarkerFaceColor','b',...
'MarkerSize',10)

set(gca,'FontSize',20)
xlabel('Number of neurons');
ylabel('E_{RMS}');
ylim([0,y_lim_max])


optimal_neuron_index = 9;
plot([optimal_neuron_index,optimal_neuron_index],[0, y_lim_max],'--k','LineWidth',2)
hold off

