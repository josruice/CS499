% TODO: Write proper documentation

% image_part can be 'FIRST HALF', 'SECOND HALF', 'COMPLETE'.
% feature_method can be 'PHOW', 'SIFT', 'DSIFT'.

function [descriptors, total_descriptors] = get_descriptors(root_path, folders, file_names, img_format, image_part, feature_method)
% Variables to improve code legibility.
num_folders = length(folders);
num_file_names = length(file_names);

% Create a counter and a cell array to store the descriptor matrices.
total_descriptors = 0;
descriptors = cell(num_folders, num_file_names);

% Iterate through all the folder images.
    for i = 1:num_folders,
        for j = 1:num_file_names,
            % Build image path.
            image_path = sprintf('%s/%s/%s.%s', root_path, folders{i}, file_names{j}, img_format);

            % Load the image.
            image = imread(image_path);

            % Split the image so that only the required part is used.
            [h w d] = size(image); % Height, width and depth.
            half_w = floor(w/2);

            switch upper(image_part)
            case 'FIRST HALF'
                image = image(:, 1:half_w, :);
            case 'SECOND HALF'
                image = image(:, half_w+1:end, :);
            case 'COMPLETE'
                % Do nothing, since the image is already complete.
            end

            % Convert the image to SINGLE (feature extractors requirements).
            image = im2single(image);

            % Get frames and descriptors using a feature extraction method.
            %  - The size of matrix D will be [128 num_descriptors].
            switch upper(feature_method)
            case 'PHOW'
                [F D] = vl_phow(image);
            case 'SIFT'
                if ndims(image) == 3
                    image = rgb2gray(image); % Gray scale. SIFT requirement.
                end
                [F D] = vl_sift(image);
            case 'DSIFT'
                if ndims(image) == 3
                    image = rgb2gray(image); % Gray scale. DSIFT requirement.
                end
                [F D] = vl_dsift(image);
            end

            % Update the counter and store the descriptors.
            total_descriptors = total_descriptors + size(D,2);
            descriptors{i,j} = D;
        end
    end
end