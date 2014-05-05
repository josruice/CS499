% TODO: Write proper documentation.

function [bayes] = NB (properties, materials, varargin)

    % Constants.
    DEFAULT_VERBOSE = 0;
    DEFAULT_FIGURENAME = 'Naive Bayes Confusion';

    DISTRIBUTION_PARAM = 'Distribution';
    DISTRIBUTION = 'mn';

    % Create arguments parser.
    parser = inputParser;

    % Add required and parametrized arguments.
    parser.addRequired('Properties', @(x) length(x)>0);
    parser.addRequired('Materials', @(x) length(x)>0);
    parser.addParamValue('Verbose', DEFAULT_VERBOSE, @isnumeric);
    parser.addParamValue('FigureName', DEFAULT_FIGURENAME, @isstr);

    % Parse input arguments.
    parser.parse(properties, materials, varargin{:});

    % Read the arguments.
    inputs = parser.Results;
    properties = inputs.Properties;
    materials = inputs.Materials; 
    VERBOSE = inputs.Verbose;
    FIGURE_NAME = inputs.FigureName;

    % Variables to improve code legibility.
    [num_materials, num_file_names, num_properties] = size(properties);
    num_images = num_materials * num_file_names;

    % Permute and reshape the class labels and the properties vectors to fit the
    % Naive Bayes fitting function requirements.
    material_labels = repmat(materials, num_file_names, 1);
    properties = permute(properties, [3 2 1]);
    properties = reshape(properties, [num_properties, num_images]);

    % Train the Naive Bayes classifier with the given data.
    bayes = NaiveBayes.fit(properties', material_labels(:), ...
        DISTRIBUTION_PARAM, DISTRIBUTION);

    if VERBOSE >= 1
        % Check and print the accuracy of the results.
        I = cellfun(@(x) find(strcmp(x, materials)), bayes.predict(properties'));
        num_correctly_classified = sum(I == ceil([1:num_images]./num_file_names)');

        fprintf(1, 'Correct with Naive Bayes: %d out of %d (%.2f %%)\n', num_correctly_classified, num_images, num_correctly_classified*100/num_images);

        % Compute and plot the confusion matrix.
        confusion = accumarray([ceil([1:num_images]./num_file_names); I']', 1) ./ num_file_names;
        plot_confusion(confusion, FIGURE_NAME, materials);
    end
end