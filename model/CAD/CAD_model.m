function output_mat = CAD_model(input_mat)
output_mat = [];
g = 9.81;
for i = 1:size(input_mat,2)
    q1 = input_mat(1,i);
    q2 = input_mat(2,i);
    q3 = input_mat(3,i);
    q4 = input_mat(4,i);
    q5 = input_mat(5,i);
    q6 = input_mat(6,i);
    output_mat(:,end+1) = CAD_analytical_regressor(g,q1,q2,q3,q4,q5,q6)*CAD_dynamic_vec; 
end
end