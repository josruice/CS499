% TODO: Write proper documentation.

function [M] = extract_from_cell (cell_array, DIM)
    num_cells = length(cell_array(:));
    [num_rows num_columns] = size(cell_array{1}{DIM});

    if num_rows > 1 && num_columns > 1
        % Matrix case.
        M = zeros(num_rows, num_columns, num_cells);
        for i = 1:num_cells,
                M(:,:,i) = cell_array{i}{DIM};
        end
    else
        % Vector case. 
        vectors_cell = cell(num_cells, 1);
        for i = 1:num_cells,
            vectors_cell{i} = cell_array{i}{DIM};
        end

        % Row vector and single element case.
        M = cell2mat(vectors_cell);

        if num_rows > 1
            % Column vector case.
            M = reshape(M, [num_rows num_cells]);
        end
    end
end