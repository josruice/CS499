% TODO: Write proper documentation.

function [matrix, num_descriptors] = descriptors_cell_to_single_matrix(descriptors, total_descriptors)

    % Create the matrix where the descriptors will be stored by columns.
    num_rows = size(descriptors{1,1}, 1);
    matrix = zeros(num_rows, total_descriptors, 'single');

    % Get the size of the descriptors cell.
    [num_rows num_columns] = size(descriptors);

    % Store the number of descriptors of each image in a mirror cell array.
    num_descriptors = cell(num_rows, num_columns);

    % Cover all the cells.
    next = 1;
    for i = 1:num_rows,
        for j = 1:num_columns,
            num_descriptors{i,j} = size(descriptors{i,j}, 2);
            matrix(:, next : next+num_descriptors{i,j}-1) = descriptors{i,j};
            next = next + num_descriptors{i,j};
    end

end