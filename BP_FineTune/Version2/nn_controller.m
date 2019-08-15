classdef nn_controller < handle
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
        nn_dynamicModel
    end

    methods(Access = public)
        % Class constructor
        function obj = nn_controller(mtm_arm,...
                nn_dynamicModel,...
                safe_upper_torque_limit,...
                safe_lower_torque_limit,...
                ARM_NAME,...
                Zero_Output_Joint_No)
            obj.nn_dynamicModel = nn_dynamicModel;
            obj.safe_upper_torque_limit = safe_upper_torque_limit;
            obj.safe_lower_torque_limit = safe_lower_torque_limit;
            obj.mtm_arm = mtm_arm;
            obj.ARM_NAME = ARM_NAME;
            obj.pub_tor = rospublisher(['/dvrk/',ARM_NAME,'/set_effort_joint']);
            obj.sub_pos = rossubscriber(['/dvrk/',ARM_NAME,'/state_joint_current']);
            if exist('Zero_Output_Joint_No')
                obj.Zero_Output_Joint_No = Zero_Output_Joint_No;
            end
        end

        % Callback function of pose subscriber when start gc controller
        function callback_gc_publisher(obj, q, q_dot)
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
            Torques = obj.base_controller(q, q_dot);

            % Publish predict torques
            msg = rosmessage(obj.pub_tor);
            for i =1:7
                msg.Effort(i) = Torques(i);
            end
            send(obj.pub_tor, msg);
        end

        % Base controller to calculate the predict torque
        function Torques = base_controller(obj, q, q_dot)
            vel = q_dot;
            Torques = obj.nn_dynamicModel(q);
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
            callback_MTM = @(src,msg)(obj.callback_gc_publisher(msg.Position,...
                msg.Velocity));
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

