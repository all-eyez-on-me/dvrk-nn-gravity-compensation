function config_lse_list=setting_lse(config, data_input_root_path)
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    %General Setting
    Is_Plot = false;
    issave_figure = false;
    std_filter = 0.3;
    g_constant = 9.8;
    fit_method = '4POL';
    %Generate config obj for lse
    Joint_No = 6;
    config_lse_joint6 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint6'],fit_method,g_constant);

    Joint_No = 5;
    config_lse_joint5 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint5'],fit_method,g_constant);

    Joint_No = 4;
    config_lse_joint4 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint4'],fit_method,g_constant);

    Joint_No = 3;
    config_lse_joint3 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint3'],fit_method,g_constant);

    Joint_No = 2;
    config_lse_joint2 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint2'],fit_method,g_constant);

    Joint_No = 1;
    config_lse_joint1 =  config_lse(Joint_No,std_filter,Is_Plot,...
        issave_figure,[data_input_root_path,'/Train_Joint1'],fit_method,g_constant);
    
    config_lse_list = {config_lse_joint1,config_lse_joint2,config_lse_joint3,config_lse_joint4,config_lse_joint5,config_lse_joint6};

end