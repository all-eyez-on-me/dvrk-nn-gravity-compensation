function Torques_data_mat = mlse(dataCollection_info_str)
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    argument_checking(dataCollection_info_str)

    % General Setting
    output_file_str = '';

    % Read JSON config input file dataCollection_info_str
    fid = fopen(dataCollection_info_str);
    if fid<3
        error('Cannot read file %s', dataCollection_info_str)
    end
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    config = jsondecode(str);

    % Get the path
    [input_data_path_with_date, ~, ~] = fileparts(dataCollection_info_str);

    % display data input root path
    disp(' ');
    fprintf('data path for MLSE : ''%s'' \n', input_data_path_with_date);
    disp(' ');


    % create config_LSE objects
    config_lse_list=setting_lse(config,input_data_path_with_date);

    Torques_data_mat = [];
    % Multi-steps MLSE from Joint#6 to Joint#1.
    for i=6:-1:1
        if i==6
            Torques_data_tmp = lse_mtm_one_joint(config_lse_list{i});
            Torques_data_mat = cat(3, Torques_data_mat, Torques_data_tmp);
        elseif i==1
            Torques_data_tmp = lse_mtm_one_joint(config_lse_list{i},config_lse_list{i+1});
            Torques_data_mat = cat(3, Torques_data_mat, Torques_data_tmp);
        else
            Torques_data_tmp = lse_mtm_one_joint(config_lse_list{i},config_lse_list{i+1});
            Torques_data_mat = cat(3, Torques_data_mat, Torques_data_tmp);
        end
    end

end

function argument_checking(input_data_path_with_date)
    if ischar(input_data_path_with_date) ==0
        error('%s is not a char object', input_data_path_with_date)
    end
end

function Torques_data_mat = lse_mtm_one_joint(config_lse_joint, previous_config)
    %  Institute: The Chinese University of Hong Kong
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    
    fprintf('LSE for joint %d started..\n', config_lse_joint.Joint_No);

    if ~exist('previous_config')
         Torques_data_mat = lse_model(config_lse_joint);
    else
        % if there is 'previous_config', we pass the path to the result of previous step of LSE to lse_model
        Torques_data_mat = lse_model(config_lse_joint,...
            previous_config.output_param_path);
    end
end

function  Torques_data_mat = lse_model(config_lse_joint1,...
                                            old_param_path)

    %  Institute: The Chinese University of Hong Kong
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05


    lse_obj = lse_preparation(config_lse_joint1);
    
    Torques_data_mat = cat(3,lse_obj.Torques_pos_data, lse_obj.Torques_neg_data); 

end


function lse_obj = lse_preparation(config_lse_joint)

    % create lse_obj inheriting config_lse_joint
    lse_obj.Joint_No = config_lse_joint.Joint_No;
    lse_obj.std_filter = config_lse_joint.std_filter;
    lse_obj.g_constant = config_lse_joint.g_constant;
    lse_obj.Is_Plot = config_lse_joint.Is_Plot;
    lse_obj.issave_figure = config_lse_joint.issave_figure;
    lse_obj.Input_Pos_Data_Path = config_lse_joint.input_pos_data_path;
    lse_obj.Input_Neg_Data_Path = config_lse_joint.input_neg_data_path;
    lse_obj.input_pos_data_files = config_lse_joint.input_pos_data_files;
    lse_obj.input_neg_data_files = config_lse_joint.input_neg_data_files;
    lse_obj.new_param_save_path = config_lse_joint.output_param_path;
    lse_obj.new_fig_pos_save_path = config_lse_joint.output_pos_fig_path;
    lse_obj.new_fig_neg_save_path = config_lse_joint.output_neg_fig_path;
    lse_obj.prior_param_index = config_lse_joint.prior_param_index;
    lse_obj.prior_param_values = config_lse_joint.prior_param_values;
    lse_obj.Output_Param_Joint_No = config_lse_joint.Output_Param_Joint_No;
    lse_obj.std_filter = config_lse_joint.std_filter;
    lse_obj.fit_method = config_lse_joint.fit_method;

    % check the given joint path exist
    if exist(lse_obj.Input_Pos_Data_Path)==0
        error('Cannot find input data folder: %s. Please check that input data folder exists.', lse_obj.Input_Pos_Data_Path);
    end
    if exist(lse_obj.Input_Neg_Data_Path)==0
        error('Cannot find input data folder: %s. Please check that input data folder exists.', lse_obj.Input_Neg_Data_Path);
    end

    data_path_struct_list = dir(lse_obj.input_pos_data_files);
    lse_obj.Torques_pos_data_list = {};
    lse_obj.theta_pos_list = {};
    for i=1:size(data_path_struct_list,1)
        load(strcat(data_path_struct_list(i).folder,'/',data_path_struct_list(i).name));
        lse_obj.Torques_pos_data_list{end+1} = torques_data_process(current_position,...
            desired_effort,...
            'mean',...
            lse_obj.std_filter);
        lse_obj.theta_pos_list{end+1} = int32(Theta);
    end

    data_path_struct_list = dir(lse_obj.input_neg_data_files);
    lse_obj.Torques_neg_data_list = {};
    lse_obj.theta_neg_list = {};
    for i=1:size(data_path_struct_list,1)
        load(strcat(data_path_struct_list(i).folder,'/',data_path_struct_list(i).name));
        lse_obj.Torques_neg_data_list{end+1} = torques_data_process(current_position,...
            desired_effort,...
            'mean',...
            lse_obj.std_filter);
        lse_obj.theta_neg_list{end+1} = int32(Theta);
    end

    % Append List Torques Data
    lse_obj.Torques_pos_data = [];
    for j = 1:size(lse_obj.Torques_pos_data_list,2)
        lse_obj.Torques_pos_data = cat(3,lse_obj.Torques_pos_data,lse_obj.Torques_pos_data_list{j});
    end
    lse_obj.Torques_neg_data = [];
    for j = 1:size(lse_obj.Torques_neg_data_list,2)
        lse_obj.Torques_neg_data = cat(3,lse_obj.Torques_neg_data,lse_obj.Torques_neg_data_list{j});
    end

end

function Torques_data = torques_data_process(current_position, desired_effort, method, std_filter)
    %current_position = current_position(:,:,1:10);
    %desired_effort = desired_effort(:,:,1:10);
    d_size = size(desired_effort);
    Torques_data = zeros(7,2,d_size(2));
    %First Filter out Point out of 1 std, then save the date with its index whose value is close to mean
    for i=1:d_size(2)
        for j=1:d_size(1)
            for k=1:d_size(3)
                effort_data_array(k)=desired_effort(j,i,k);
                position_data_array(k)=current_position(j,i,k);
            end
            effort_data_std = std(effort_data_array);
            effort_data_mean = mean(effort_data_array);
            if effort_data_std<0.0001
                effort_data_std = 0.0001;
            end
            %filter out anomalous data out of 1 standard deviation
            select_index = (effort_data_array <= effort_data_mean+effort_data_std*std_filter)...
                &(effort_data_array >= effort_data_mean-effort_data_std*std_filter);

            effort_data_filtered = effort_data_array(select_index);
            position_data_filtered = position_data_array(select_index);
            if size(effort_data_filtered,2) == 0
                effort_data_filtered =effort_data_array;
                position_data_filtered = position_data_array;
            end
            effort_data_filtered_mean = mean(effort_data_filtered);
            position_data_filtered_mean = mean(position_data_filtered);
            for e = 1:size(effort_data_filtered,2)
                if e==1
                    final_index = 1;
                    min_val =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                else
                    abs_result =abs(effort_data_filtered(e)-effort_data_filtered_mean);
                    if(min_val>abs_result)
                        min_val = abs_result;
                        final_index = e;
                    end
                end
            end
            if(strcmpi(method,'mean'))
                Torques_data(j,1,i) = position_data_filtered_mean;
                Torques_data(j,2,i) = effort_data_filtered_mean;
            elseif(strcmpi(method,'min_abs_error'))
                Torques_data(j,1,i) = current_position(j,i,final_index);
                Torques_data(j,2,i) = desired_effort(j,i,final_index);
            else
                error('Method argument is wrong, please pass: mean or min_abs_error.')
            end
        end
    end

    % Tick out the data collecting from some joint configuration which reaches limits and have cable force effect.
    Torques_data = Torques_data(:,:,3:end-1);
end

function  [R2_augmented, T2_augmented] = data2augmat(Torques_data,...
        Joint_No,...
        direction,...
        g)
    R2_augmented = [];
    T2_augmented = [];

    for i=1:size(Torques_data,3)
        R = analytical_regressor_mat_dual_dir(direction,g,Torques_data(:,1,i)');
        R2_augmented = [R2_augmented;R(Joint_No,:)];
        T2_augmented = [T2_augmented;Torques_data(Joint_No,2,i)];
    end

end

function plot_fitting_curves(lse_obj,direction,dynamic_parameters_vec)
    if(strcmp(direction,'pos'))
        Torques_data_list = lse_obj.Torques_pos_data_list;
        theta_list = lse_obj.theta_pos_list;
        fig_save_path = lse_obj.new_fig_pos_save_path;
    elseif(strcmp(direction,'neg'))
        Torques_data_list = lse_obj.Torques_neg_data_list;
        theta_list = lse_obj.theta_neg_list;
        fig_save_path = lse_obj.new_fig_neg_save_path;
    end
    for j = 1:size(Torques_data_list,2)
        if(size(Torques_data_list{j},3)~=0  )
            plot_fit_joint(Torques_data_list{j},...
                dynamic_parameters_vec,...
                theta_list{j},...
                direction,...
                lse_obj.Joint_No, ...
                lse_obj.g_constant,...
                lse_obj.Is_Plot, ...
                lse_obj.issave_figure,...
                fig_save_path, ...
                j);
        end
    end
end

function plot_fit_joint(Torques_data,...
        dynamic_parameters,...
        theta,...
        dir,...
        Joint_No,...
        g,...
        isplot,...
        issave,...
        save_path,...
        save_file_index)
    %  Institute: The Chinese University of Hong Kong
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05


    for i=1:size(Torques_data,3)
        Regressor_Matrix = analytical_regressor_mat_dual_dir(dir,g,Torques_data(:,1,i));
        F = Regressor_Matrix(Joint_No,:)*dynamic_parameters;

        x(i) = Torques_data(Joint_No,1,i);
        y1(i) = Torques_data(Joint_No,2,i);
        y2(i) = F;
    end
    x= x.';
    x=x*180/pi;
    y1= y1.';
    y2= y2.';

    if isplot
        figure;
    else
        figure('visible', 'off')
    end
    title_string = sprintf('Actual and Predicted Torque of Joint%d at theta=%d', Joint_No, theta);
    xlabel_string = sprintf('Joint %d Angle',Joint_No);
    ylabel_string = sprintf('Joint %d Torque',Joint_No);
    scatter(x,y1,100);
    hold on
    plot(x,y2)
    title(title_string);
    xlabel(xlabel_string);
    ylabel(ylabel_string);

    if issave == 1
        if exist(save_path)~=7
            mkdir(save_path)
        end
        saveas(gcf, strcat(save_path,'/Figure_',int2str(save_file_index),'_',title_string,'.png'));
        fprintf(strcat('Figure, [',title_string,'.png] saved.\n'));
    end
end


% Gravity compensation test


function [joint_position_upper_limit, joint_position_lower_limit] =  generate_joint_angle_limit(config)
    joint_position_upper_limit = zeros(1,7);
    joint_position_lower_limit = zeros(1,7);
    
    if strcmp(config.ARM_NAME, 'MTML')
        joint_position_upper_limit(1) = config.data_collection.joint1.train_angle_max.MTML;
        joint_position_upper_limit(2) = config.data_collection.joint2.train_angle_max;
        joint_position_upper_limit(3) = config.data_collection.joint3.train_angle_max;
        joint_position_upper_limit(4) = config.data_collection.joint4.train_angle_max.MTML;
        joint_position_upper_limit(5) = config.data_collection.joint5.train_angle_max;
        joint_position_upper_limit(6) = config.data_collection.joint6.train_angle_max;
        joint_position_upper_limit(7) = 400;

        joint_position_lower_limit(1) = config.data_collection.joint1.train_angle_min.MTML;
        joint_position_lower_limit(2) = config.data_collection.joint2.train_angle_min;
        joint_position_lower_limit(3) = config.data_collection.joint3.train_angle_min;
        joint_position_lower_limit(4) = config.data_collection.joint4.train_angle_min.MTML;
        joint_position_lower_limit(5) = config.data_collection.joint5.train_angle_min;
        joint_position_lower_limit(6) = config.data_collection.joint6.train_angle_min;
        joint_position_lower_limit(7) = -400;
    end
    
   
    if strcmp(config.ARM_NAME, 'MTMR')
        joint_position_upper_limit(1) = config.data_collection.joint1.train_angle_max.MTMR;
        joint_position_upper_limit(2) = config.data_collection.joint2.train_angle_max;
        joint_position_upper_limit(3) = config.data_collection.joint3.train_angle_max;
        joint_position_upper_limit(4) = config.data_collection.joint4.train_angle_max.MTMR;
        joint_position_upper_limit(5) = config.data_collection.joint5.train_angle_max;
        joint_position_upper_limit(6) = config.data_collection.joint6.train_angle_max;
        joint_position_upper_limit(7) = 400;

        joint_position_lower_limit(1) = config.data_collection.joint1.train_angle_min.MTMR;
        joint_position_lower_limit(2) = config.data_collection.joint2.train_angle_min;
        joint_position_lower_limit(3) = config.data_collection.joint3.train_angle_min;
        joint_position_lower_limit(4) = config.data_collection.joint4.train_angle_min.MTMR;
        joint_position_lower_limit(5) = config.data_collection.joint5.train_angle_min;
        joint_position_lower_limit(6) = config.data_collection.joint6.train_angle_min;
        joint_position_lower_limit(7) = -400;
    end
end

function [torque_upper_limit, torque_lower_limit] = traning_data_tor_limit(Torques_data_mat)
    measure_tor_mat=[];
    for i=1:size(Torques_data_mat,3)
        measure_tor_mat = cat(2,measure_tor_mat,Torques_data_mat(:,2,i));
    end
    torque_upper_limit = max(measure_tor_mat,[],2);
    torque_lower_limit = min(measure_tor_mat,[],2);
end

