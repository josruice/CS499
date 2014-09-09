function [estimatedLabelsArray, bayesClassifier, confusionMatrix] =          ...
    naiveBayes (featuresMatrix, realClassLabelsArray, nClasses, modality,    ...
                varargin)
% naiveBayes Classifies the given samples using Naive Bayes.
%
%   [estimatedLabelsArray, bayesClassifier, confusionMatrix] = naiveBayes(
%       featuresMatrix, realClassLabelsArray, nClasses, modality)
%   returns the estimated labels, bayes classifier and confusion matrix that 
%   result of the classification process of the given feature vectors using ...
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
parser.addRequired(NUMBER_OF_CLASSES_PARAM, @isnumeric);
parser.addRequired(MODALITY_PARAM, @isstr);

parser.addParamValue(BAYES_CLASSIFIER_PARAM, DEFAULT_BAYES_CLASSIFIER, ...
                     @(x) length(x)>0);
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(featuresMatrix, realClassLabelsArray, nClasses, modality, ...
             varargin{:});
inputs = parser.Results;

% Read the arguments.
featuresMatrix = inputs.(FEATURES_PARAM);
realClassLabelsArray = inputs.(LABELS_PARAM); 
nClasses = inputs.(NUMBER_OF_CLASSES_PARAM);
modality = inputs.(MODALITY_PARAM);

bayesClassifier = inputs.(BAYES_CLASSIFIER_PARAM);
verbose = inputs.(VERBOSE_PARAM);

% Check if the modality given is allowed.
modalityAllowed = any(strcmpi(modality, ALLOWED_MODALITIES));
if not(modalityAllowed)
    error(INVALID_PARAM_ERROR, MODALITY_PARAM);
end

% Variables to improve code legibility.
nSamplesPerClassArray = accumarray(realClassLabelsArray, 1, [nClasses,1]);
[nSamples, nFeaturesPerSample] = size(featuresMatrix);

if strcmpi(modality, TRAINING_MODALITY)
    % Train a Naive Bayes classifier with the given data.
    bayesClassifier = NaiveBayes.fit(featuresMatrix, realClassLabelsArray, ...
        BAYES_DISTRIBUTION_INTERNAL_PARAM, BAYES_MULTINOMIAL_DISTRIBUTION);
end

% Test the Naive Bayes classifier with the given data.
estimatedLabelsArray = bayesClassifier.predict(featuresMatrix);

% Compute the confusion matrix.
confusionMatrix = accumarray([realClassLabelsArray';            ...
                              estimatedLabelsArray']',          ...
                              1, [nClasses,nClasses])  ./       ...
                              repmat(nSamplesPerClassArray, [1,nClasses]);

if verbose >= 1
    % Check and print the accuracy of the results.
    nCorrectlyClassified = sum(realClassLabelsArray == estimatedLabelsArray);

    fprintf(STDOUT, 'Correct with Naive Bayes: %d out of %d (%.2f %%)\n', nCorrectlyClassified, nSamples, nCorrectlyClassified*100/nSamples);
end