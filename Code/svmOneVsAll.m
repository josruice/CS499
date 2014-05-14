function [estimatedLabelsMatrix, confusionMatrix] = ...
    svmOneVsAll (svmsCellArray, featuresMatrix, realClassLabelsMatrix, varargin)

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(SVMS_PARAM, @(x) length(x)>0);
parser.addRequired(FEATURES_PARAM, @(x) length(x)>0);
parser.addRequired(LABELS_PARAM, @(x) length(x)>0);
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(svmsCellArray, featuresMatrix, realClassLabelsMatrix, varargin{:});
inputs = parser.Results;

% Read the arguments.
svmsCellArray = inputs.(SVMS_PARAM);
featuresMatrix = inputs.(FEATURES_PARAM);
realClassLabelsMatrix = inputs.(LABELS_PARAM); 
verbose = inputs.(VERBOSE_PARAM);

% Variables to improve code legibility.
[nCategories, nSamplesPerCategory, nFeaturesPerSample] = size(featuresMatrix);
nSamples = nCategories * nSamplesPerCategory;

% Permute and reshape the features to have one column per sample.
realClassLabelsArray = reshape(realClassLabelsMatrix', [], 1);
featuresMatrix = reshape(permute(featuresMatrix, [3 2 1]), ...
                         [nFeaturesPerSample, nSamples]);

% Matrix with the weight vectors of the SVMs as columns.
weights = extractFromCell (svmsCellArray, SMVS_CELL_ARRAY_WEIGHT_VECTOR_INDEX);

% Column vector with the bias of the SVMs.
biases = extractFromCell (svmsCellArray, SMVS_CELL_ARRAY_BIAS_INDEX);

% Compute the scores applying a standard linear classifier.
scores = (weights' * featuresMatrix) + repmat(biases, [1, nSamples]);
[~, estimatedLabelsArray] = max(scores);
estimatedLabelsMatrix = reshape(estimatedLabelsArray, [nSamplesPerCategory, ...
                                                       nCategories])';

% Compute the confusion matrix.
confusionMatrix = accumarray([realClassLabelsArray'; ...
                              estimatedLabelsArray]',1) ./ nSamplesPerCategory;

if verbose >= 1
    % Compute and print the obtained accuracy.
    nCorrectlyClassified = sum(realClassLabelsArray' == estimatedLabelsArray);

    fprintf(STDOUT, ['Correct with SVMs (one vs. all): %d out of %d ' ...
                     '(%.2f %%)\n'], nCorrectlyClassified, nSamples,  ...
                                     nCorrectlyClassified*100/nSamples);
end