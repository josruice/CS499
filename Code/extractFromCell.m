function [M] = extractFromCell (cellArray, dimension)
    nCells = length(cellArray(:));
    [nRows nColumns] = size(cellArray{1}{dimension});

    if nRows > 1 && nColumns > 1
        % Matrix case.
        M = zeros(nRows, nColumns, nCells);
        for i = 1:nCells,
            M(:,:,i) = cellArray{i}{dimension};
        end
    else
        % Vector case. 
        vectorsCellArray = cell(nCells, 1);
        for i = 1:nCells,
            vectorsCellArray{i} = cellArray{i}{dimension};
        end

        % Row vector and single element case.
        M = cell2mat(vectorsCellArray);

        if nRows > 1
            % Column vector case.
            M = reshape(M, [nRows nCells]);
        end
    end
end