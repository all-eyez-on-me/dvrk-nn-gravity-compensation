function [abs_RMS_vec, rel_RMS_vec] = RMS(t_mat, output_mat)

e = gsubtract(t_mat,output_mat);
abs_RMS_vec = sqrt(sum(e.^2,2));
rel_RMS_vec = abs_RMS_vec./sqrt(sum(t_mat.^2, 2));


end

