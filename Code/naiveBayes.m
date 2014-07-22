function [estimatedLabelsMatrix, bayesClassifier, confusionMatrix] = ...
    naiveBayes (featuresMatrix, realClassLabelsMatrix, modality, varargin)
% naiveBayes Classifies the given samples using Naive Bayes.
%
%   [estimatedLabelsMatrix, bayesClassifier, confusionMatrix] =
%       naiveBayes (featuresMatrix, realClassLabelsMatrix, modality)
%   returns the estimated labels, bayes classifier and confusion matrix that 
%   result of the classification process of the given feature vectors using the 
%   specified SVMs with One vs. All.
%   
%   naiveBayes() accepts the following options:
%   
%   
%   
%   
%   
%   

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(FEATURES_PARAM, @(x) length(x)>0);
parser.addRequired(LABELS_PARAM, @(x) length(x)>0);
parser.addRequired(MODALITY_PARAM, @isstr);

parser.addParamValue(BAYES_CLASSIFIER_PARAM, DEFAULT_BAYES_CLASSIFIER, ...
                     @(x) length(x)>0);
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(featuresMatrix, realClassLabelsMatrix, modality, varargin{:});
inputs = parser.Results;

% Read the arguments.
featuresMatrix = inputs.(FEATURES_PARAM);
realClassLabelsMatrix = inputs.(LABELS_PARAM); 
modality = inputs.(MODALITY_PARAM);

bayesClassifier = inputs.(BAYES_CLASSIFIER_PARAM);
verbose = inputs.(VERBOSE_PARAM);

% Check if the modality given is allowed.
modalityAllowed = any(strcmpi(modality, ALLOWED_MODALITIES));
if not(modalityAllowed)
    error(INVALID_PARAM_ERROR, MODALITY_PARAM);
end

% Variables to improve code legibility.
[nCategories, nSamplesPerCategory, nFeaturesPerSample] = size(featuresMatrix);
nSamples = nCategories * nSamplesPerCategory;

% Permute and reshape the class labels and features to fit the Naive Bayes 
% fitting function requirements: samples by rows.
realClassLabelsArray = reshape(realClassLabelsMatrix', [], 1);
featuresMatrix = reshape(permute(featuresMatrix, [3 2 1]), ...
                         [nFeaturesPerSample, nSamples])';


if strcmpi(modality, TRAINING_MODALITY)
    % Train a Naive Bayes classifier with the given data.
    bayesClassifier = NaiveBayes.fit(featuresMatrix, realClassLabelsArray, ...
        BAYES_DISTRIBUTION_INTERNAL_PARAM, BAYES_MULTINOMIAL_DISTRIBUTION);
end

% Test the Naive Bayes classifier with the given data.
estimatedLabelsArray = bayesClassifier.predict(featuresMatrix);
estimatedLabelsMatrix = reshape(estimatedLabelsArray, [nSamplesPerCategory, ...
                                                       nCategories])';

% Compute the confusion matrix.
confusionMatrix = accumarray([realClassLabelsArray'; ...
                              estimatedLabelsArray']',1) ./ nSamplesPerCategory;

if verbose >= 1
    % Check and print the accuracy of the results.
    nCorrectlyClassified = sum(realClassLabelsArray == estimatedLabelsArray);

    fprintf(STDOUT, 'Correct with Naive Bayes: %d out of %d (%.2f %%)\n', nCorrectlyClassified, nSamples, nCorrectlyClassified*100/nSamples);
end