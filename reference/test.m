load('nn_data.mat');
net = Neural_Network([7, 601, 7]);
% net.SGD(obj, training_data_cell_mat, epochs, mini_batch_size, eta, testing_data_cell_mat)
net.SGD(training_data_cell_mat,          5,         300,         0.5, testing_data_cell_mat);
