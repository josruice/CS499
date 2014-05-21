%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CS 499 - Senior thesis (Spring 2014)  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%% Material dataset classification using SVM.
%
%%% Advisor:   Dr. David Alexander Forsyth  (daf -at- uiuc.edu)
%%% Coadvisor: Dr. Roberto Paredes Palacios (rparedes -at- dsic.upv.es)
%%% Student:       Jose Vicente Ruiz Cepeda (ruizcep2 -at- illinois.edu)
%
% Algorithm outline:
%   1. Get the features of the training images using a feature extraction 
%      method.
%   2. Quantize the feature vectors:
%       - Partition the feature descriptors in clusters using k-means.
%       - Define the image feature vector as an array with size the number of
%         clusters where index i represents the number of image features whose
%         nearest neighbour is cluster i.
%   3. Train one linear classifier per material property using Support Vector
%      Machines (SVM) and the labels of the manual markup.
%   4. Get the features of the test images and quantize them using the clusters
%      of the training data.
%   5. Use the previous linear classifiers to obtain the vector of properties of
%      the test images.
%   6. Train a Naive Bayes classifier and a SVMs with the vector of properties
%      of the training images and use them to classify the vector of properties  
%      of the test images.
%   [... In progress ...]
%

% --------------------------------------------------------------------------- %

%%%%%%%%%%%%%%%%%%
%%%   Script   %%%
%%%%%%%%%%%%%%%%%%

% Load constants file.
loadConstants;

% Load constants that specify this execution.
loadControlPanel;

% Load VLFeat library.
run(VLFEAT_LIBRARY_PATH);

% Variables to improve code legibility.
nClasses = length(CLASSES_CELL_ARRAY);
nSamplesPerClass = length(SAMPLE_FILENAMES_CELL_ARRAY);
nSamples = nClasses * nSamplesPerClass;

% Read markup file and store the properties in a cell array.
propertiesCellArray = readFeatures(MARKUP_FILE, nClasses, nSamplesPerClass);
nProperties = length(propertiesCellArray);

% Display the file names of the sample images with every property if required.
if VERBOSE >= 2
    for iProperty = 1:nProperties,
        disp(propertiesCellArray{iProperty}{1});
        [J,I] = find(propertiesCellArray{iProperty}{2}');
        disp(strcat(IMAGES_ROOT_PATH, '/', CLASSES_CELL_ARRAY(I)', '/', ...
                   SAMPLE_FILENAMES_CELL_ARRAY(J)', '.', SAMPLE_IMAGES_FORMAT));
    end
end

% --------------------------------------------------------------------------- %

disp('Execution data:');
fprintf(STDOUT, [' - Descriptors: %s\n - Num clusters: %d (datatype %s, '    ...
        'hierarchical %d, branching %d)\n - SVM solver: %s (%s loss, '       ...
        'lambda %f)\n\n'], FEATURE_EXTRACTOR, NUM_CLUSTERS, KMEANS_DATATYPE, ...
        KMEANS_HIERARCHICAL, KMEANS_BRANCHING_FACTOR, SOLVER, LOSS, LAMBDA);

% --------------------------------------------------------------------------- %

%%%
%%% Training SVMs for properties detection. 
%%%

% Build the images path from the constants.
imagePathsCellArray = buildImagesPaths (IMAGES_ROOT_PATH,               ...
                                        CLASSES_CELL_ARRAY,             ...
                                        SAMPLE_FILENAMES_CELL_ARRAY,    ...
                                        SAMPLE_IMAGES_FORMAT);

% Get the descriptors of the first half of each image in this set of images.
disp('Training');
fprintf(STDOUT, ' - Descriptors: '); tic;
trainingDescriptorsCellArray = computeDescriptors(imagePathsCellArray,  ...
    IMAGE_PART_PARAM,                TRAINING_IMAGE_PART,               ...
    FEATURE_EXTRACTOR_PARAM,         FEATURE_EXTRACTOR,                 ...
    MAX_DESCRIPTORS_PER_IMAGE_PARAM, MAX_DESCRIPTORS_PER_IMAGE); 
fprintf('%d descriptors. ', sum(cellfun(@(x) size(x,2), ...
                                        trainingDescriptorsCellArray(:)))); toc;

% Quantize the feature descriptors using k-means.
fprintf(STDOUT, ' - Kmeans: '); tic;
[quantizedTrainDescriptorsCellArray, clusterCenterMatrix, nClusters] =      ...
quantizeVectors (trainingDescriptorsCellArray, NUM_CLUSTERS,                ...
    KMEANS_DATATYPE_PARAM,          KMEANS_DATATYPE,                        ...
    KMEANS_HIERARCHICAL_PARAM,      KMEANS_HIERARCHICAL,                    ...
    KMEANS_BRANCHING_FACTOR_PARAM,  KMEANS_BRANCHING_FACTOR);
fprintf('%d clusters. ', nClusters); toc;

% Use SVM to create the linear clasifiers for the properties.
% SVM cell structure: 'scale-feature', weight vector (W), bias (B).
fprintf(STDOUT, ' - SMVs: '); tic;
[estimatedTrainPropertiesCellArray, propertiesSvmsCellArray] =      ...
svm (quantizedTrainDescriptorsCellArray, propertiesCellArray,       ...
    TRAINING_MODALITY,                                              ...
    LAMBDA_PARAM,  LAMBDA,                                          ...
    SOLVER_PARAM,  SOLVER,                                          ...
    LOSS_PARAM,    LOSS,                                            ...
    VERBOSE_PARAM, VERBOSE);
toc; fprintf(STDOUT, '\n');


% --------------------------------------------------------------------------- %

%%%
%%% Testing SVMs for properties detection.
%%%

% Get the descriptors of the second half of each image in this set of images.
disp('Testing: ');
fprintf(STDOUT, ' - Descriptors: '); tic;
testDescriptorsCellArray = computeDescriptors(imagePathsCellArray,  ...
    IMAGE_PART_PARAM,                TEST_IMAGE_PART,               ...
    FEATURE_EXTRACTOR_PARAM,         FEATURE_EXTRACTOR,             ...
    MAX_DESCRIPTORS_PER_IMAGE_PARAM, MAX_DESCRIPTORS_PER_IMAGE); 
fprintf('%d descriptors. ', sum(cellfun(@(x) size(x,2), ...
                                        testDescriptorsCellArray(:)))); toc;

% Quantize the feature descriptors using k-means.
fprintf(STDOUT, ' - Kmeans: '); tic;
quantizedTestDescriptorsCellArray = quantizeVectors (           ...
    testDescriptorsCellArray,       nClusters,                  ...
    CLUSTERS_CENTERS_PARAM,         clusterCenterMatrix,        ...
    KMEANS_DATATYPE_PARAM,          KMEANS_DATATYPE,            ...
    KMEANS_HIERARCHICAL_PARAM,      KMEANS_HIERARCHICAL,        ...
    KMEANS_BRANCHING_FACTOR_PARAM,  KMEANS_BRANCHING_FACTOR);
toc;

% Use the previously created SVMs with the training data to estimate the 
% vectors of properties of the test images.
fprintf(STDOUT, ' - SVMs: '); tic;
estimatedTestPropertiesCellArray =                                  ...
svm (quantizedTestDescriptorsCellArray, propertiesCellArray,        ...
    TESTING_MODALITY,                                               ...
    SVMS_PARAM,    propertiesSvmsCellArray,                         ...
    VERBOSE_PARAM, VERBOSE+1);
toc;


% --------------------------------------------------------------------------- %

% Build matrices with real material labels, real properties and estimated
% properties both from training and testing.
realMaterialLabelsMatrix = repmat(1:nClasses, [nSamplesPerClass, 1])';
realPropertiesMatrix = extractFromCell(propertiesCellArray, 2);
estimatedTrainPropertiesMatrix = cell2mat(cellfun(@(x) reshape(x, 1, 1, []), ...
                                         estimatedTrainPropertiesCellArray,  ...
                                         'UniformOutput', false));
estimatedTestPropertiesMatrix = cell2mat(cellfun(@(x) reshape(x, 1, 1, []), ...
                                         estimatedTestPropertiesCellArray,  ...
                                         'UniformOutput', false));


% --------------------------------------------------------------------------- %

%%%
%%% Train Naive Bayes for material labeling with predicted data.
%%%
[~, estimatedBayesClassifier] = naiveBayes (estimatedTrainPropertiesMatrix,  ...
                                            realMaterialLabelsMatrix,        ...
                                            TRAINING_MODALITY,               ...
                                            VERBOSE_PARAM, VERBOSE);

%%%
%%% Train Naive Bayes for material labeling with ground truth data.
%%%
[~, realBayesClassifier] = naiveBayes (realPropertiesMatrix,       ...
                                       realMaterialLabelsMatrix,   ...
                                       TRAINING_MODALITY,          ...
                                       VERBOSE_PARAM, VERBOSE);

% --------------------------------------------------------------------------- %

%%%
%%% Testing Naive Bayes trained with predicted data for material labeling.
%%%

%% Predicted data.
if VERBOSE >= 1
    fprintf(STDOUT, '\nNaive Bayes trained with predicted data:\n');
    fprintf(STDOUT, '[Predicted data] ');
end
[estimatedMaterialLabelsMatrix, ~, confusionMatrix] =                       ...
    naiveBayes (estimatedTestPropertiesMatrix, realMaterialLabelsMatrix,    ...
               TESTING_MODALITY,                                            ...
               BAYES_CLASSIFIER_PARAM, estimatedBayesClassifier,            ...
               VERBOSE_PARAM, VERBOSE);

%% Ground truth data.
if VERBOSE >= 1
    fprintf(STDOUT, '[Ground truth data] ');
end
[estimatedMaterialLabelsMatrix, ~, confusionMatrix] =                       ...
    naiveBayes (realPropertiesMatrix, realMaterialLabelsMatrix,             ...
               TESTING_MODALITY,                                            ...
               BAYES_CLASSIFIER_PARAM, estimatedBayesClassifier,            ...
               VERBOSE_PARAM, VERBOSE);

%%%
%%% Testing Naive Bayes trained with ground truth data for material labeling.
%%%

%% Ground truth data.
if VERBOSE >= 1
    fprintf(STDOUT, '\nNaive Bayes trained with ground truth data:\n');
    fprintf(STDOUT, '[Predicted data] ');
end
[estimatedMaterialLabelsMatrix, ~, confusionMatrix] =                       ...
    naiveBayes (estimatedTestPropertiesMatrix, realMaterialLabelsMatrix,    ...
               TESTING_MODALITY,                                            ...
               BAYES_CLASSIFIER_PARAM, realBayesClassifier,                 ...
               VERBOSE_PARAM, VERBOSE);

%% Predicted data.
if VERBOSE >= 1
    fprintf(STDOUT, '[Ground truth data] ');
end
[estimatedMaterialLabelsMatrix, ~, confusionMatrix] =                       ...
    naiveBayes (realPropertiesMatrix, realMaterialLabelsMatrix,             ...
               TESTING_MODALITY,                                            ...
               BAYES_CLASSIFIER_PARAM, realBayesClassifier,                 ...
               VERBOSE_PARAM, VERBOSE);


% --------------------------------------------------------------------------- %

% Build a cell array only with the real properties of the materials.
realPropertiesCellArray = mat2cell(realPropertiesMatrix, ...
                                    ones(nClasses,1),    ...
                                    ones(nSamplesPerClass, 1), nProperties);
realPropertiesCellArray = cellfun(@(x) reshape(x, [], 1),   ...
                                   realPropertiesCellArray, ...
                                   'UniformOutput', false);

% Build a cell array with one cell per material with its name and a binary vector
% with the images of that material.
materialsCellArray = cell(nClasses, 1);
for i = 1:nClasses,
    materialLabels = zeros(nSamples, 1);
    materialLabels((i-1)*nSamplesPerClass+1 : i*nSamplesPerClass) = 1;
    materialsCellArray{i} = {CLASSES_CELL_ARRAY{i}, materialLabels};
end

% --------------------------------------------------------------------------- %

%%%
%%% Train SVMs for material labeling with estimated data.
%%%
[~, estimatedMaterialsSvmsCellArray] =                      ...
svm (estimatedTrainPropertiesCellArray, materialsCellArray, ...
    TRAINING_MODALITY,                                      ...
    LAMBDA_PARAM,  LAMBDA,                                  ...
    SOLVER_PARAM,  SOLVER,                                  ...
    LOSS_PARAM,    LOSS,                                    ...
    VERBOSE_PARAM, VERBOSE-1);

%%%
%%% Train SVMs for material labeling with ground truth data.
%%%
[~, realMaterialsSvmsCellArray] =                   ...
svm (realPropertiesCellArray, materialsCellArray,   ...
    TRAINING_MODALITY,                              ...
    LAMBDA_PARAM,  LAMBDA,                          ...
    SOLVER_PARAM,  SOLVER,                          ...
    LOSS_PARAM,    LOSS,                            ...
    VERBOSE_PARAM, VERBOSE-1);

% --------------------------------------------------------------------------- %

%%%
%%% Testing SVMs (one vs. all) trained with predicted data for material 
%%% labeling. 
%%%

%% Predicted data.
if VERBOSE >= 1
    fprintf(STDOUT, '\nSVMs trained with predicted data:\n');
    fprintf(STDOUT, '[Predicted data] ');
end
[estimatedMaterialLabelsMatrix, confusionMatrix] =  ...
    svmOneVsAll (estimatedMaterialsSvmsCellArray,   ...
                 estimatedTestPropertiesMatrix,     ...
                 realMaterialLabelsMatrix, VERBOSE_PARAM, VERBOSE);

%% Ground truth data.
if VERBOSE >= 1
    fprintf(STDOUT, '[Ground truth data] ');
end
[estimatedMaterialLabelsMatrix, confusionMatrix] =                      ...
    svmOneVsAll (estimatedMaterialsSvmsCellArray, realPropertiesMatrix, ...
                 realMaterialLabelsMatrix, VERBOSE_PARAM, VERBOSE);


%%%
%%% Testing SVMs (one vs. all) trained with ground truth data for material 
%%% labeling. 
%%%

%% Predicted data.
if VERBOSE >= 1
    fprintf(STDOUT, '\nSVMs trained with ground truth data:\n');
    fprintf(STDOUT, '[Predicted data] ');
end
[estimatedMaterialLabelsMatrix, confusionMatrix] =                          ...
    svmOneVsAll (realMaterialsSvmsCellArray, estimatedTestPropertiesMatrix, ...
                 realMaterialLabelsMatrix, VERBOSE_PARAM, VERBOSE);

%% Ground truth data.
if VERBOSE >= 1
    fprintf(STDOUT, '[Ground truth data] ');
end
[estimatedMaterialLabelsMatrix, confusionMatrix] =                 ...
    svmOneVsAll (realMaterialsSvmsCellArray, realPropertiesMatrix, ...
                 realMaterialLabelsMatrix, VERBOSE_PARAM, VERBOSE);


% --------------------------------------------------------------------------- %

