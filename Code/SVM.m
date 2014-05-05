% TODO: Write proper documentation.

function [estimated_labels, svms] = SVM (features_3d, cell_real_labels, modality, varargin)

    % Constants.
    TRAINING = 'Training';
    TEST = 'Test';
    SOLVER_PARAM = 'Solver';
    LOSS_PARAM = 'Loss';

    DEFAULT_SVMS = {};
    DEFAULT_BINARY_HISTOGRAMS = false;
    DEFAULT_LAMBDA = 0.01;
    DEFAULT_SOLVER = 'SDCA';
    DEFAULT_LOSS = 'Logistic';
    DEFAULT_VERBOSE = 0;

    % Create arguments parser.
    parser = inputParser;

    % Add required and parametrized arguments.
    parser.addRequired('Features', @(x) length(x)>0);
    parser.addRequired('Labels', @(x) length(x)>0);
    parser.addRequired('Modality', @isstr);

    parser.addParamValue('SVMs', DEFAULT_SVMS, @(x) length(x)>0);
    parser.addParamValue('BinaryHistograms', DEFAULT_BINARY_HISTOGRAMS, @islogical);
    parser.addParamValue('Lambda', DEFAULT_LAMBDA, @isnumeric);
    parser.addParamValue('Solver', DEFAULT_SOLVER, @isstr);
    parser.addParamValue('Loss', DEFAULT_LOSS, @isstr);
    parser.addParamValue('Verbose', DEFAULT_VERBOSE, @isnumeric);

    % Parse input arguments.
    parser.parse(features_3d, cell_real_labels, modality, varargin{:});

    % Read the arguments.
    inputs = parser.Results;
    features_3d = inputs.Features;
    cell_real_labels = inputs.Labels; 
    modality = inputs.Modality;

    svms = inputs.SVMs;
    binary_histograms = inputs.BinaryHistograms;
    lambda = inputs.Lambda;
    solver = inputs.Solver;
    loss = inputs.Loss;
    verbose = inputs.Verbose;

    % Variables to improve code legibility.
    [num_categories, num_images_per_category, num_features_per_image] = size(features_3d);
    num_images = num_categories * num_images_per_category;
    num_classes = length(cell_real_labels);

    % Permute and reshape the features to fit the Support Vector Machine (SVM)
    % requirements: one column per example.
    features_2d = permute(features_3d, [3 2 1]);
    features_2d = reshape(features_2d, [num_features_per_image, num_images]);

    % Use binary histograms if required.
    if binary_histograms
        features_2d( find(features_2d) ) = 1;
    end

    if not(strcmp(modality, TEST))
        % Cell array formed by cell arrays each with 3 elements: 
        % {'Class name', SVM weight vector (W), SVM bias (B)}.
        svms = cell(num_classes, 1);
    end

    % 3D matrix to store the estimated labels of the images. The rows represent
    % the categories, the columns the file names and the depth the labels with
    % value 0 (it does not belong to this class) or 1 (it belongs to this class).
    estimated_labels = zeros(num_categories, num_images_per_category, num_classes);

    % Variables used to test the accuracy.
    min_accuracy = 1;
    max_accuracy = 0;
    sum_of_accuracies = 0;

    % If the modality is test, use the given SVMs to classify the specified 
    % feature vectors. Otherwise, use SVMs to create the linear classifiers.
    for i = 1:num_classes,
        class_cell = cell_real_labels{i};
        class_name = class_cell{1};
        class_real_binary_labels = class_cell{2}; % 0 or 1.

        % Convert binary labels to -1 or 1 (SVMs requirement).
        class_real_labels = class_real_binary_labels;
        class_real_labels( find(class_real_binary_labels == 0) ) = -1; 

        if strcmp(modality, TEST)
            % Use the SVM linear classifiers to classify the test data.
            % Cell structure: {'Class name', SVM weight vector (W), SVM bias (B)}.
            W = svms{i}{2};
            B = svms{i}{3};
            scores = (W' * features_2d) + B;
        else
            if not( strcmp(modality, TRAINING) )
                fprintf(1, 'Wrong modality. Default execution.\n');
            end

            % Build the classifier for this class.
            [W, B, ~, scores] = vl_svmtrain(features_2d, class_real_labels, lambda, SOLVER_PARAM, solver, LOSS_PARAM, loss);

            % Store everything in the cell data structure.
            svms{i} = {class_name, W, B};
        end

        % Elements with score 0 (on the line of the linear classifier) are
        % the same as negative (don't belong to the class).
        class_est_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
        class_est_binary_labels = class_est_labels;
        class_est_binary_labels( find(class_est_labels == -1) ) = 0;
        
        % Store the estimated labels of each image using binary vectors.
        estimated_labels(:,:,i) = reshape(class_est_binary_labels, ...
            [num_images_per_category, num_categories])'; 

        % Computing accuracy in the classification process.
        correct = sum(class_real_binary_labels == sign(class_est_binary_labels'));
        accuracy = correct / length(class_est_binary_labels);

        sum_of_accuracies = sum_of_accuracies + accuracy;
        min_accuracy = min(min_accuracy, accuracy);
        max_accuracy = max(max_accuracy, accuracy);
    end

    if verbose >= 1
        % Print the resulting accuracies.
        mean_accuracy = (sum_of_accuracies / num_classes);
        fprintf(1, 'Mean accuracy: %.2f. ', mean_accuracy * 100);
        %fprintf(1, '%s:\n', modality);
        %fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
        %fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
        %fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);
    end
end