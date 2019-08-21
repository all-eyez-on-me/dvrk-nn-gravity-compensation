classdef Neural_Network
   properties(Access = public)
      weight_mat_list
      bias_vec_list 
      network_size
      activation_function
      activation_derivative_func
      cost_function_derivative
   end
   methods(Access = public)
       function obj = Neural_Network(Neural_Network_size_arr, varargin)
            p = inputParser;
            is_array = @(x) size(x,1) == 1; 
            is_function = @(x) isa(x,'function_handle');
            default_activation_function = @(x) 1.0/(1.0+exp(-x));
            default_activation_derivative_func = @(x) exp(-x)/((1+exp(-x))^2);
           addRequired(p,'Neural_Network_size_arr',is_array);
           addOptional(p,'activation_function',default_activation_function,is_function); 
           addOptional(p,'activation_derivative_func',default_activation_derivative_func,is_function); 
            parse(p,Neural_Network_size_arr,varargin{:});
            % initiate function
            obj.weight_mat_list = {};
            obj.bias_vec_list = {};
            for i = 2:size(Neural_Network_size_arr,2)
                obj.weight_mat_list{end+1} = randn(Neural_Network_size_arr(:,i),Neural_Network_size_arr(:,i-1));
                obj.bias_vec_list{end+1} = randn(Neural_Network_size_arr(:,i),1);
            end
            obj.network_size = size(Neural_Network_size_arr,2);
            obj.activation_function = p.Results.activation_function;
            obj.activation_derivative_func = p.Results.activation_derivative_func;
            
            % Cost function derivative using MSE
            obj.cost_function_derivative = @(x,y) x -y;
       end
       function a = feedforward(obj, a_in)
            a = a_in; 
            for i=1:obj.network_size-1
                a = arrayfun(obj.activation_function,obj.weight_mat_list{i}*a + obj.bias_vec_list{i});
            end
       end    
      function [a_list, z_list] = feedforward_list(obj, a_in)
            a_list = {a_in};
            z_list = {};
            for i=1:obj.network_size-1
                % z = wx + b
                z = obj.weight_mat_list{i}*a_list{i} + obj.bias_vec_list{i};
                % activation = act_func(z)
                activation = arrayfun(obj.activation_function, z);
                z_list{end+1} = z;
                a_list{end+1} = activation;
            end
      end
        function obj = SGD(obj, training_data_cell_mat, epochs, mini_batch_size, eta, testing_data_cell_mat)
            for j=1:epochs
                data_size = size(training_data_cell_mat,1);
                ridx = randperm(data_size);
                mini_batch_idx_list = {};
                for i=mini_batch_size:mini_batch_size:data_size
                    mini_batch_idx_list{end+1} = ridx(i-mini_batch_size+1:i);
                end
                % If there is also some residual tail data
                if i~=data_size
                     mini_batch_idx_list{end+1} = ridx(i+1:end);
                end
                for i=1:size(mini_batch_idx_list,2)
                    obj = obj.update_mini_batches(training_data_cell_mat(mini_batch_idx_list{i},:), eta);
                end
                success_rate = obj.evaluate(testing_data_cell_mat);
                  disp(sprintf('epochs: %d / %d  success rate: %.1f %%', j, epochs, success_rate*100));
            end
        end
        function obj = update_mini_batches(obj, training_data_cell_mat, eta)
            grad_weight_mat_list = obj.weight_mat_list;
            grad_bias_vec_list = obj.bias_vec_list;
            % initialize with zeros
            for i = 1:size(grad_weight_mat_list,2)
                grad_weight_mat_list{i} = zeros(size(grad_weight_mat_list{i})); 
                grad_bias_vec_list{i} = zeros(size(grad_bias_vec_list{i})); 
            end
            
            % calculate gradient decent for weight matrix and bias vector
            for i = 1:size(training_data_cell_mat,1)
                [delta_weight_mat_list,delta_bias_vec_list] = back_prob(obj, training_data_cell_mat{i,1}, training_data_cell_mat{i,2});
                for j = 1:size(grad_weight_mat_list,2)
                    grad_weight_mat_list{j} = grad_weight_mat_list{j} + delta_weight_mat_list{j}; 
                    grad_bias_vec_list{j} = grad_bias_vec_list{j} + delta_bias_vec_list{j}; 
                end                
            end
            
            % Update parameters
            for i = 1:size(grad_weight_mat_list,2)
                 sum_deta_weight_mat_list = grad_weight_mat_list{i}.*eta./size(training_data_cell_mat,1);
                 sum_deta_bias_vec_list = grad_bias_vec_list{i}.*eta./size(training_data_cell_mat,1);
                 w_new = obj.weight_mat_list{i} - sum_deta_weight_mat_list;
                 b_new = obj.bias_vec_list{i} - sum_deta_bias_vec_list;
                 obj.weight_mat_list{i} = w_new;
                 obj.bias_vec_list{i} = b_new;
            end  
             %disp(sprintf('%.4f', obj.bias_vec_list{1}(end)));
        end
        
        function [delta_weight_mat_list,delta_bias_vec_list] = back_prob(obj, x, y)
            delta_weight_mat_list = obj.weight_mat_list;
            delta_bias_vec_list = obj.bias_vec_list;
            [a_list, z_list] = obj.feedforward_list(x);
            % initialize with zeros
            for i = 1:size(delta_weight_mat_list,2)
                delta_weight_mat_list{i} = zeros(size(delta_weight_mat_list{i})); 
                delta_bias_vec_list{i} = zeros(size(delta_bias_vec_list{i})); 
            end
            
            % For the last layer
             delta_vec = dot(arrayfun(obj.activation_derivative_func, z_list{end}), arrayfun(obj.cost_function_derivative,a_list{end}, y), 2); 
             delta_bias_vec_list{end} = delta_vec;
             delta_weight_mat_list{end} = delta_vec*a_list{end-1}.';
             
            % From the last second layer to first layer
             for i =  size(delta_weight_mat_list,2)-1:-1:1
                delta_vec = dot(arrayfun(obj.activation_derivative_func, z_list{i}), obj.weight_mat_list{i+1}.'*delta_bias_vec_list{i+1}, 2);
                delta_bias_vec_list{i} = delta_vec;
                delta_weight_mat_list{i} = delta_vec*a_list{i}.';
             end
        end
        
        function success_rate = evaluate(obj, testing_data_cell_mat)
            success_count = 0;
            for i = 1:size(testing_data_cell_mat,1)
                [~, y_out] = max(obj.feedforward(testing_data_cell_mat{i,1}));
                % the index range using python defaut 0~9
                y_t= testing_data_cell_mat{i,2}+1;
                if y_t == y_out
                    success_count = success_count+1;
                end
            end
            success_rate = success_count / size(testing_data_cell_mat,1);
        end
   end
end