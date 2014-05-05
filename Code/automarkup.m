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
%   1. Get the features of the training images using a feature extraction method.
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
%   6. Train a Naive Bayes classifier and a SVMs with the vector of properties of
%      the training images and use them to classify the vector of properties of 
%      the test images.
%   [... In progress ...]
%

%%%%%%%%%%%%%%%%%%%%%
%%%   Constants   %%%
%%%%%%%%%%%%%%%%%%%%%

%%% General.
VERBOSE = 1; % Verbose level.

%%% Data.
ROOT_PATH = '../Dataset'; % Without last slash.

MATERIALS = {'Birch', 'Brick', 'Concrete', 'Corduroy', 'Denim', 'Elm', 'Feathers', 'Fur', 'Hair', 'KnitAran', 'KnitGuernsey', 'Leather', 'Marble', 'Scale', 'Silk', 'Slate', 'Stucco', 'Velour'};

FILE_NAMES = {'01','02','03','04','05','06','07','08','09','10','11','12'};

IMG_FORMAT = 'png';

MARKUP_FILE = '../Markups/Machine-Markup-(1.0).txt';

NUM_PROPERTIES_PER_IMAGE = 6;

%%% Descriptors.
FEATURE_METHOD = 'PHOW';            % PHOW, SIFT or DSIFT.
MAX_DESCRIPTORS_PER_IMAGE = 1000;  % 0 means no maximum.

%%% K-means
NUM_CLUSTERS = 300;     % Min number of clusters obtained.
DATATYPE = 'single';      % Datatype of the descriptors matrix: single or uint8.
HIERARCHICAL = false;     % Hierarchical (only with uint8).
BRANCHING_FACTOR = 2;    % Branching factor (only with HIERARCHICAL).

%%% Support Vector Machines.
SOLVER = 'SDCA';    % Solver method: SGD or SDCA.
LOSS = 'Logistic';  % Loss function: Hinge, Hinge2, L1, L2 or LOGISTIC.
LAMBDA = 0.000001;  % Lambda value of the SVM.

%%% Others (do NOT change).
% Function parameters names.
CLUSTERS_PARAM = 'Clusters';
DATATYPE_PARAM = 'Datatype';
HIERARCHICAL_PARAM = 'Hierarchical';
BRANCHING_PARAM = 'Branching';

TRAINING_METHOD = 'Training';
TEST_METHOD = 'Test';
SMVS_PARAM = 'SVMs';
SOLVER_PARAM = 'Solver';
LOSS_PARAM = 'Loss';
LAMBDA_PARAM = 'Lambda';

VERBOSE_PARAM = 'Verbose';
FIGURE_NAME_PARAM = 'FigureName';


%%%%%%%%%%%%%%%%%%
%%%   Script   %%%
%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');

% Variables to improve code legibility.
num_materials = length(MATERIALS);
num_file_names = length(FILE_NAMES);
num_images = num_materials * num_file_names;

%%% Read markup %%%

% The structure of the output cell array will be a column vector of cell arrays
% with one row per different property, where each of the elements store the name
% of the property (scale-property_name) and a column vector with the images that 
% have the property. 
[cell_real_properties, real_properties, num_properties] = read_markup(MARKUP_FILE, num_materials, num_file_names);


%%% TRAINING SVMs %%%

% Get the descriptors of the first half of each image in this set of images.
disp('Execution data:');
fprintf(1,' - Descriptors: %s\n - Num clusters: %d (datatype %s, hierarchical %d, branching %d)\n - SVM solver: %s (%s loss, lambda %f)\n\n', FEATURE_METHOD, NUM_CLUSTERS, DATATYPE, HIERARCHICAL, BRANCHING_FACTOR, SOLVER, LOSS, LAMBDA);
disp('Training');
fprintf(1, ' - Descriptors: '); tic;
[training_descriptors, total_training_descriptors] = get_descriptors( ...
    ROOT_PATH, MATERIALS, FILE_NAMES, IMG_FORMAT,                     ...
    'FIRST HALF', FEATURE_METHOD, MAX_DESCRIPTORS_PER_IMAGE);
fprintf('%d descriptors. ', total_training_descriptors); toc;

% Quantize the feature descriptors using k-means.
fprintf(1, ' - Kmeans: '); tic;
[training_features_3d, clusters, real_num_clusters] = quantize_feature_vectors(...
    training_descriptors, total_training_descriptors, NUM_CLUSTERS,            ...
    DATATYPE_PARAM,     DATATYPE,                                              ...
    HIERARCHICAL_PARAM, HIERARCHICAL,                                          ...
    BRANCHING_PARAM,    BRANCHING_FACTOR);
fprintf('%d clusters. ', real_num_clusters); toc;

% Use SVM to create the linear clasifiers for the properties.
% SVM cell structure: 'scale', 'feature', weight vector (W), bias (B).
fprintf(1, ' - SMVs: '); tic;
[~, svms] = SVM (training_features_3d, cell_real_properties, TRAINING_METHOD, ...
    LAMBDA_PARAM,  LAMBDA,                                                    ...
    SOLVER_PARAM,  SOLVER,                                                    ...
    LOSS_PARAM,    LOSS,                                                      ...
    VERBOSE_PARAM, VERBOSE);
toc;


%%% TESTING SVMs %%%

% Get the descriptors of the second half of each image in this set of images.
disp('Testing: ');
fprintf(1, ' - Descriptors: '); tic;
[test_descriptors, total_test_descriptors] = get_descriptors(   ...
    ROOT_PATH, MATERIALS, FILE_NAMES, IMG_FORMAT,               ...
    'SECOND HALF', FEATURE_METHOD, MAX_DESCRIPTORS_PER_IMAGE);
fprintf('%d descriptors. ', total_test_descriptors); toc;

% Quantize the feature descriptors using k-means.
fprintf(1, ' - Kmeans: '); tic;
test_features_3d = quantize_feature_vectors (                            ...
    test_descriptors, total_test_descriptors, real_num_clusters,         ...
    CLUSTERS_PARAM,     clusters,                                        ...
    DATATYPE_PARAM,     DATATYPE,                                        ...
    HIERARCHICAL_PARAM, HIERARCHICAL,                                    ...
    BRANCHING_PARAM,    BRANCHING_FACTOR);
toc;

% Use the previous created SVMs with the training data to estimate the 
% vectors of properties of the test images.
fprintf(1, ' - SVMs: '); tic;
est_properties = SVM (test_features_3d, cell_real_properties, TEST_METHOD, ...
    SMVS_PARAM,  svms,                                                     ...
    VERBOSE_PARAM, VERBOSE);
toc;

%keyboard;

%%% Naive Bayes %%%

%% Predicted data.
if VERBOSE >= 1
    fprintf(1, '\n[Predicted data] ');
end
bayes = NB (est_properties, MATERIALS, VERBOSE_PARAM, VERBOSE, ...
            FIGURE_NAME_PARAM, 'Naive Bayes Confusion - Predicted Data');

%% Ground truth data.
if VERBOSE >= 1
    fprintf(1, '[Ground truth data] ');
end
bayes = NB (real_properties, MATERIALS, VERBOSE_PARAM, VERBOSE, ...
            FIGURE_NAME_PARAM, 'Naive Bayes Confusion - Ground Truth');

%keyboard;

%%% SVMs %%%

% Permute and reshape the class labels and the properties vectors to fit the
% SVMs requirements.
material_labels = repmat(MATERIALS, num_file_names, 1);
est_properties = permute(est_properties, [3 2 1]);
est_properties = reshape(est_properties, [num_properties, num_images]);
real_properties = permute(real_properties, [3 2 1]);
real_properties = reshape(real_properties, [num_properties, num_images]);

%% Predicted data.

% Cell array formed by cell arrays each with 3 elements: 
% Material name string, SVM weight vector, SVM bias.
est_material_svms = cell(num_materials, 1);

% Matrix with the weight vectors of the material svms as columns.
est_weights_material_svms = zeros(num_properties, num_materials);

% Column vector with the bias of the materials svms.
est_bias_material_svms = zeros(num_materials, 1);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use one vs. all multiclass classification.
for i = 1:num_materials,
    % Build the column vector with the labels.
    labels = -ones(num_images, 1);
    labels( (1:num_file_names) + ((i-1)*num_file_names) ) = 1;

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(est_properties, labels, LAMBDA, SOLVER_PARAM, SOLVER);

    % Store everything in the cell data structure and the matrix of weights
    % and vector of bias.
    est_material_svms{i} = {MATERIALS(i), W, B};
    est_weights_material_svms(:,i) = W;
    est_bias_material_svms(i) = B;

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == 0) ) = -1;

    % Testing accuracy in the training set.
    accuracy = sum(labels == sign(estimated_labels')) / length(labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

% Check the accuracy of the results.
mean_accuracy = (mean_accuracy / num_materials);
fprintf(1, '\n[Predicted data] Correct with SVMs: %d out of %d (%.2f %%)\n', floor(mean_accuracy*num_images), num_images, mean_accuracy*100);

%% Ground truth data.

% Cell array formed by cell arrays each with 3 elements: 
% Material name string, SVM weight vector, SVM bias.
real_material_svms = cell(num_materials, 1);

% Matrix with the weight vectors of the material svms as columns.
real_weights_material_svms = zeros(num_properties, num_materials);

% Column vector with the bias of the materials svms.
real_bias_material_svms = zeros(num_materials, 1);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use one vs. all multiclass classification.
for i = 1:num_materials,
    % Build the column vector with the labels.
    labels = -ones(num_images, 1);
    labels( (1:num_file_names) + ((i-1)*num_file_names) ) = 1;

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(real_properties, labels, LAMBDA, SOLVER_PARAM, SOLVER);

    % Store everything in the cell data structure and the matrix of weights
    % and vector of bias.
    real_material_svms{i} = {MATERIALS(i), W, B};
    real_weights_material_svms(:,i) = W;
    real_bias_material_svms(i) = B;

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == 0) ) = -1;

    % Testing accuracy in the training set.
    accuracy = sum(labels == sign(estimated_labels')) / length(labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

% Check the accuracy of the results.
mean_accuracy = (mean_accuracy / num_materials);
fprintf(1, '[Ground truth data] Correct with SVMs: %d out of %d (%.2f %%)\n', floor(mean_accuracy*num_images), num_images, mean_accuracy*100);


%%% SVMs with real one vs. all %%%

%% Predicted data.

est_scores = (est_weights_material_svms' * est_properties) + repmat(est_bias_material_svms, [1, num_images]);
[~,I] = max(est_scores);
confusion = accumarray([ceil([1:num_images]./num_file_names); I]', 1) ./ num_file_names;
plot_confusion(confusion, 'SMVs Confusion - Predicted Data', MATERIALS); 
[~,I] = max(est_scores);
indices_well_classified = cellfun(@strcmp, MATERIALS(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, '\n[Predicted data] Correct with SVMs (one vs. all): %d out of %d (%.2f %%)\n', num_correctly_classified, num_images, num_correctly_classified*100/num_images);

%% Ground truth data.

real_scores = (real_weights_material_svms' * real_properties) + repmat(real_bias_material_svms, [1, num_images]);
[~,I] = max(real_scores);
confusion = accumarray([ceil([1:num_images]./num_file_names); I]', 1) ./ num_file_names;
plot_confusion(confusion, 'SMVs Confusion - Ground Truth', MATERIALS); 
indices_well_classified = cellfun(@strcmp, MATERIALS(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, '[Ground truth data] Correct with SVMs (one vs. all): %d out of %d (%.2f %%)\n', num_correctly_classified, num_images, num_correctly_classified*100/num_images);

%keyboard;