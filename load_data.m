function [input_cell, output_cell] = load_data(input_path, output_path, fixWindowLength ,pattern, range_index)
    load(input_path);
    load(output_path);
    if ~exist('pattern')
        pattern = 'repeat';
    end
    if ~exist('range_index')
        range_index = 1:size(input_mat,2);
    end
    input_cell = {};
    output_cell = {};
    input_mat= input_mat(1:6,range_index);
    output_mat= output_mat(1:6,range_index);
    input_temp =zeros(6,fixWindowLength);
    output_temp =zeros(6,fixWindowLength);
    if(strcmp(pattern, 'repeat'))
        for i = 1:size(input_mat,2)
            for j = 1:fixWindowLength
                % repeat pattern
                    input_temp(:,j) = input_mat(:,i);
                    output_temp(:,j) = output_mat(:,i);
            end
            input_cell = vertcat(input_cell, input_temp);
            output_cell = vertcat(output_cell, output_temp);
        end
    elseif (strcmp(pattern, 'order'))
        % cut data into cells with fix window
        for i = 1:size(input_mat,2)-fixWindowLength+1
            % order pattern
            input_cell = vertcat(input_cell, {input_mat(:,i:i+fixWindowLength-1)});
            output_cell = vertcat(output_cell, {output_mat(:,i:i+fixWindowLength-1)});
        end
    end
end