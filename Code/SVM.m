% TODO: Write proper documentation.

function [est_properties, svms] = SVM (features_3d_in, cell_real_properties_in, method_in, varargin)

    % Constants.
    TRAINING = 'Training';
    TEST = 'Test';
    SOLVER_PARAM = 'SOLVER';

    DEFAULT_LAMBDA = 0.01;
    DEFAULT_SOLVER = 'SDCA';
    DEFAULT_VERBOSE = 0;

    PROPERTY_NAME_DELIM = '-';

    % Create arguments parser.
    parser = inputParser;

    % Add required and parametrized arguments.
    parser.addRequired('Features', @(x) length(x)>0);
    parser.addRequired('RealProperties', @(x) length(x)>0);
    parser.addRequired('Method', @isstr);

    parser.addParamValue('SVMs', {}, @(x) length(x)>0);
    parser.addParamValue('Lambda', DEFAULT_LAMBDA, @isnumeric);
    parser.addParamValue('Solver', DEFAULT_SOLVER, @isstr);
    parser.addParamValue('Verbose', DEFAULT_VERBOSE, @isnumeric);

    % Parse input arguments.
    parser.parse(features_3d_in, cell_real_properties_in, method_in, varargin{:});

    % Read the arguments.
    inputs = parser.Results;
    features_3d = inputs.Features;
    cell_real_properties = inputs.RealProperties; 
    method = inputs.Method;

    svms = inputs.SVMs;
    LAMBDA = inputs.Lambda;
    SOLVER = inputs.Solver;
    VERBOSE = inputs.Verbose;

    % Variables to improve code legibility.
    [num_materials, num_file_names, real_num_clusters] = size(features_3d);
    num_images = num_materials * num_file_names;
    num_properties = length(cell_real_properties);

    % Permute and reshape the features to fit the Support Vector Machine (SVM)
    % requirements: one column per example.
    features_2d = permute(features_3d, [3 2 1]);
    features_2d = reshape(features_2d, [real_num_clusters, num_images]);
    features_2d( find(features_2d) ) = 1; % Binary histograms.

    if not(strcmp(method, TEST))
        % Cell array formed by cell arrays each with 4 elements: 
        % Scale string, feature string, SVM weight vector, SVM bias.
        svms = cell(num_properties, 1);
    end

    % 3D matrix to store the binary properties vectors of the images. The rows
    % represent the materials, the columns the file names and the depth the 
    % properties with value 0 (not present) or 1 (present).
    est_properties = zeros(num_materials, num_file_names, num_properties);

    % Variables used to test the accuracy.
    min_accuracy = 1;
    max_accuracy = 0;
    mean_accuracy = 0;

    % If given, use the SVMs to classify the specified feature vectors. 
    % Otherwise, use SVMs to create the linear classifiers for the properties.
    for i = 1:num_properties,
        prop = cell_real_properties{i};
        prop_name = prop{1};
        binary_labels = prop{2}; % 0 or 1.

        labels = binary_labels;
        labels( find(binary_labels == 0) ) = -1; % -1 or 1 (SVMs requirement).

        if strcmp(method, TEST)
            % Use the SVM linear classifiers to classify the test data.
            % SVM cell structure: 'scale', 'feature', weight vector (W), bias (B).
            W = svms{i}{3};
            B = svms{i}{4};
            scores = (W' * features_2d) + B;
        else
            if not( strcmp(method, TRAINING) )
                fprintf(1, 'Wrong method. Default execution.\n');
            end

            % Get scale and feature of the property.
            scale_and_feature = strsplit(prop_name, PROPERTY_NAME_DELIM);
            scale = scale_and_feature(1);
            feature = scale_and_feature(2);

            % Build the classifier for this property.
            [W, B, ~, scores] = vl_svmtrain(features_2d, labels, LAMBDA, SOLVER_PARAM, SOLVER);

            % Store everything in the cell data structure.
            svms{i} = {scale, feature, W, B};
        end

        % Elements with score 0 (on the line of the linear classifier) are
        % the same as negative (don't have the property).
        estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
        estimated_labels( find(estimated_labels == -1) ) = 0;

        % Store the estimated properties of each image using binary vectors.
        est_properties(:,:,i) = reshape(estimated_labels, [num_file_names, num_materials])'; 

        % Computing accuracy in the classification process.
        accuracy = sum(binary_labels == sign(estimated_labels')) / ...
                   length(binary_labels);
        mean_accuracy = mean_accuracy + accuracy;
        min_accuracy = min(min_accuracy, accuracy);
        max_accuracy = max(max_accuracy, accuracy);
    end

    if VERBOSE >= 1
        % Print the resulting accuracies.
        mean_accuracy = (mean_accuracy / num_properties);
        fprintf(1, '%s:\n', method);
        fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
        fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
        fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);

        if VERBOSE >= 2
            % Print some statistics about the current estimated properties.
            print_estimated_properties_stats (materials, FILE_NAMES, num_properties, NUM_PROPERTIES_PER_IMAGE, est_properties);
        end
    end
end