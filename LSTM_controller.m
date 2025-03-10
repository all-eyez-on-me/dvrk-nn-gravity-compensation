classdef LSTM_controller < handle
    %  Author(s):  Hongbin LIN, Vincent Hui, Samuel Au
    %  Created on: 2018-10-05
    %  Copyright (c)  2018, The Chinese University of Hong Kong
    %  This software is provided "as is" under BSD License, with
    %  no warranty. The complete license can be found in LICENSE

    properties(Access = public)
        pub_tor
        sub_pos
        safe_upper_torque_limit
        safe_lower_torque_limit
        Zero_Output_Joint_No
        mtm_arm
        msg_counter_buffer = 0
        is_disp_init_info = false
        ARM_NAME
        net
        q_buff_mat
        Fix_window
        mu_input
        sig_input
        mu_output 
        sig_output
    end

    methods(Access = public)
        % Class constructor
        function obj = LSTM_controller(mtm_arm,...
                net,...
                safe_upper_torque_limit,...
                safe_lower_torque_limit,...
                ARM_NAME,...
                Fix_window,...
                mu_input, sig_input,...
                mu_output, sig_output,...
                Zero_Output_Joint_No)
            obj.net = net;
            obj.q_buff_mat = [];
            obj.Fix_window = Fix_window;
            obj.safe_upper_torque_limit = safe_upper_torque_limit;
            obj.safe_lower_torque_limit = safe_lower_torque_limit;
            obj.mtm_arm = mtm_arm;
            obj.ARM_NAME = ARM_NAME;
            obj.mu_input = mu_input;
            obj.sig_input = sig_input;
            obj.mu_output =  mu_output;
            obj.sig_output = sig_output;
            obj.pub_tor = rospublisher(['/dvrk/',ARM_NAME,'/set_effort_joint']);
            obj.sub_pos = rossubscriber(['/dvrk/',ARM_NAME,'/state_joint_current']);
            if exist('Zero_Output_Joint_No')
                obj.Zero_Output_Joint_No = Zero_Output_Joint_No;
            end
        end

        % Callback function of pose subscriber when start gc controller
        function callback_gc_publisher(obj, q)
            if(~obj.is_disp_init_info)
                fprintf('GC of %s starts, you can move %s now. If you need to stop gc controller, call "mtm_gc_controller.stop_gc()".\n',obj.ARM_NAME,obj.ARM_NAME);
                obj.is_disp_init_info = true;
            end

            if(obj.msg_counter_buffer==0)
                fprintf('.');
            end

            if(obj.msg_counter_buffer == 100)
                obj.msg_counter_buffer = 0;
            else
                obj.msg_counter_buffer = obj.msg_counter_buffer+1;
            end
            % Calculate predict torques
            Torques = obj.base_controller(q);

            % Publish predict torques
            msg = rosmessage(obj.pub_tor);
            for i =1:7
                msg.Effort(i) = Torques(i);
            end
            send(obj.pub_tor, msg);
        end

        % Base controller to calculate the predict torque
        function Torques = base_controller(obj, q)
            q = q(1:6,:);
            q_input = (q-obj.mu_input)./obj.sig_input;
            if size(obj.q_buff_mat,2)==obj.Fix_window
                obj.q_buff_mat = [obj.q_buff_mat(:,2:end) q_input];
            else
                obj.q_buff_mat = [obj.q_buff_mat q_input];
            end
         
            input_cell = {obj.q_buff_mat};
            
            Torques_cell = predict(obj.net, input_cell, 'MiniBatchSize',1);
            Torques_mat = Torques_cell{end};
            Torques = Torques_mat(:,end);
            
            Torques = Torques.*obj.sig_output+obj.mu_output;
            
            Torques(7,:) = 0.0; 
            for i =1:6
                % Set upper and lower torque limit, if output value exceed limits, just keep the limit value for output
                if Torques(i)>=obj.safe_upper_torque_limit(i)
                    Torques(i)=obj.safe_upper_torque_limit(i);
                elseif Torques(i)<=obj.safe_lower_torque_limit(i)
                    Torques(i)=obj.safe_lower_torque_limit(i);
                end
            end
            Torques(obj.Zero_Output_Joint_No) = 0;
        end

        % call this function to start the gc controller
        function start_gc(obj)
            % Apply GC controllers
            callback_MTM = @(src,msg)(obj.callback_gc_publisher(msg.Position));
            obj.sub_pos = rossubscriber(['/dvrk/',obj.ARM_NAME,'/state_joint_current'],callback_MTM,'BufferSize',10);
        end

        % call this function to stop the gc controller and move to origin pose
        function stop_gc(obj)
            obj.sub_pos = rossubscriber(['/dvrk/',obj.ARM_NAME,'/state_joint_current']);
            obj.mtm_arm.move_joint([0,0,0,0,0,0,0]);
            disp('gc_controller stopped');
        end
  
        function set_zero_tor_output_joint(obj, Zero_Output_Joint_No)
            obj.Zero_Output_Joint_No = Zero_Output_Joint_No
        end
    end
end

