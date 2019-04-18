data_path = 'D:\Ben\Machine Learning\dvrk-nn-gravity-compensation\data\MTML\December-05-2018-12_15_44\dataCollection_info.json'
Torques_data_mat = mlse(data_path);
position_mat = [];
torque_mat = [];
for i = 1:size(Torques_data_mat,3)
     position_mat(:,i) = Torques_data_mat(:,1,i);
     torque_mat(:,i) = Torques_data_mat(:,2,i);
end
data_cell_mat = [reshape(num2cell(position_mat,1),[size(position_mat,2),1]),reshape(num2cell(torque_mat,1),[size(torque_mat,2),1])];

random_index_list = randperm(size(data_cell_mat,1));

train_sets_rate = 0.88;
train_sets_size = round(size(data_cell_mat,1)*train_sets_rate);

training_data_cell_mat = data_cell_mat(random_index_list(1:train_sets_size),:);
testing_data_cell_mat = data_cell_mat(random_index_list(1:(size(data_cell_mat,1) - train_sets_size)),:);

save('nn_data', 'training_data_cell_mat', 'testing_data_cell_mat')