function [estimatedLabelsArray, confusionMatrix] =                     ...
    svmOneVsAll (svmsCellArray, featuresMatrix, realClassLabelsArray,  ...
                 nSamplesPerClassArray, varargin)
% svmOneVsAll Classifies the given samples using SVMs with One vs. All.
%
%   [estimatedLabelsArray, confusionMatrix] = svmOneVsAll (svmsCellArray,
%       featuresMatrix, realClassLabelsArray, nSamplesPerClassArray, varargin)
%   returns the estimated labels and the confusion matrix that result of the 
%   classification process of the given feature vectors using the specified SVMs
%   with One vs. All.
%
%   svmOneVsAll() accepts the following options:
%
%   Verbose:: 0
%       Determines the level of verbosity of the execution.

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(SVMS_PARAM, @(x) length(x)>0);
parser.addRequired(FEATURES_PARAM, @(x) length(x)>0);
parser.addRequired(LABELS_PARAM, @(x) length(x)>0);
parser.addRequired(NUMBER_OF_SAMPLES_PER_CLASS_PARAM, @(x) length(x)>0);
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(svmsCellArray, featuresMatrix, realClassLabelsArray, ...
             nSamplesPerClassArray, varargin{:});
inputs = parser.Results;

% Read the arguments.
svmsCellArray = inputs.(SVMS_PARAM);
featuresMatrix = inputs.(FEATURES_PARAM);
realClassLabelsArray = inputs.(LABELS_PARAM); 
nSamplesPerClassArray = inputs.(NUMBER_OF_SAMPLES_PER_CLASS_PARAM);
verbose = inputs.(VERBOSE_PARAM);

% Variables to improve code legibility.
nClasses = length(nSamplesPerClassArray);
[nSamples, nFeaturesPerSample] = size(featuresMatrix);

% Permute and reshape the features to have one column per sample.
featuresMatrix = reshape(permute(featuresMatrix, [3 2 1]), ...
                         [nFeaturesPerSample, nSamples]);

% Matrix with the weight vectors of the SVMs as columns.
weights = extractFromCell (svmsCellArray, SMVS_CELL_ARRAY_WEIGHT_VECTOR_INDEX);

% Column vector with the bias of the SVMs.
biases = extractFromCell (svmsCellArray, SMVS_CELL_ARRAY_BIAS_INDEX);

% Compute the scores applying a standard linear classifier.
scores = (weights' * featuresMatrix) + repmat(biases, [1, nSamples]);
[~, estimatedLabelsArray] = max(scores);

% Compute the confusion matrix.
confusionMatrix = accumarray([realClassLabelsArray';            ...
                              estimatedLabelsArray]',           ...
                              1, [nClasses,nClasses])  ./       ...
                              repmat(nSamplesPerClassArray, [1,nClasses]);

if verbose >= 1
    % Compute and print the obtained accuracy.
    nCorrectlyClassified = sum(realClassLabelsArray' == estimatedLabelsArray);

    fprintf(STDOUT, ['Correct with SVMs (one vs. all): %d out of %d ' ...
                     '(%.2f %%)\n'], nCorrectlyClassified, nSamples,  ...
                                     nCorrectlyClassified*100/nSamples);
end