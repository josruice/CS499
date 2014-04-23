%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  CS 499 - Senior thesis (Spring 2014)  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Material dataset classification using SVM.
%
% Advisor:   Dr. David Alexander Forsyth  (daf -at- uiuc.edu)
% Coadvisor: Dr. Roberto Paredes Palacios (rparedes -at- dsic.upv.es)
% Student:       Jose Vicente Ruiz Cepeda (ruizcep2 -at- illinois.edu)
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
%   [... In progress ...]
%

%%%%%%%%%%%%%%%%%%%%%
%%%   Constants   %%%
%%%%%%%%%%%%%%%%%%%%%

%%% General.
verbose = 1; % Verbose level.

%%% Data.
root_path = '../Dataset'; % Without last slash.

materials = {'Birch', 'Brick', 'Concrete', 'Corduroy', 'Denim', 'Elm', 'Feathers', 'Fur', 'Hair', 'KnitAran', 'KnitGuernsey', 'Leather', 'Marble', 'Scale', 'Silk', 'Slate', 'Stucco', 'Velour'};

file_names = {'01','02','03','04','05','06','07','08','09','10','11','12'};

img_format = 'png';

markup_file = '../Markups/Machine-Markup-(1.0).txt';

num_properties_per_image = 6;

%%% Descriptors.
feature_method = 'SIFT'; % PHOW, SIFT or DSIFT.

%%% K-means
num_clusters = 100;     % Min number of clusters obtained.
datatype = 'uint8';     % Datatype of the descriptors matrix: single or uint8.
hierarchical = true;    % Hierarchical (only with uint8).
branching_factor = 10;  % Branching factor (only with hierarchical.

%%% Suppor Vector Machines.
solver = 'SDCA';    % Solver method: SGD or SDCA.
lambda = 0.05;      % Lambda value of the SVM.

%%% Others (do NOT change).
% Function parameters names.
datatype_param = 'Datatype';
hierarchical_param = 'Hierarchical';
branching_param = 'Branching';

%%%%%%%%%%%%%%%%%%
%%%   Script   %%%
%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');

% Variables to improve code legibility.
num_materials = length(materials);
num_file_names = length(file_names);
num_images = num_materials * num_file_names;


%%% TRAINING  SVMs %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'FIRST HALF', feature_method);

% Quantize the feature descriptors using k-means.
keyboard;
[features_3d, clusters, num_clusters] = quantize_feature_vectors (  ...
    descriptors, total_descriptors, num_clusters,                   ...
    datatype_param,     datatype,                                   ...
    hierarchical_param, hierarchical,                               ...
    branching_param,    branching_factor);
keyboard;

% Read markup data. The structure of the output cell array will be a column
% vector of cell arrays with one row per different property, where each of the
% elements store the name of the property (scale-property_name) and a column
% vector with the images that have the property. 
[cell_real_properties] = read_markup(markup_file, num_materials, num_file_names);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_images]);
features_2d( find(features_2d) ) = 1; % Binary histograms.

% Cell array formed by cell arrays each with 4 elements: 
% Scale string, feature string, SVM weight vector, SVM bias.
num_properties = length(cell_real_properties);
svms = cell(num_properties, 1);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use SVM to create the linear clasifiers for the properties.
for i = 1:num_properties,
    prop = cell_real_properties{i};
    prop_name = prop{1};
    labels = prop{2};

    % Get scale and feature of the property.
    scale_and_feature = strsplit(prop_name);
    scale = scale_and_feature(1);
    feature = scale_and_feature(1);

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(features_2d, labels, lambda, 'Solver', solver);

    % Store everything in the cell data structure.
    svms{i} = {scale, feature, W, B};

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

if verbose >= 1
    % Print the resulting training accuracies.
    mean_accuracy = (mean_accuracy / num_properties);
    fprintf(1, 'TRAINING SET:\n');
    fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
    fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
    fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);
end


%%% TESTING SMVs %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'SECOND HALF', feature_method);

% Quantize the feature descriptors using k-means.
[features_3d, ] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

% Permute and reshape the features to classify all of them using matrix 
% operations: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_images]);
features_2d( find(features_2d) ) = 1; % Binary histograms.

% 3D matrices to store the binary properties vectors of the images. The rows
% represent the materials, the columns the file names and the depth the 
% properties with value 0 (not present) or 1 (present).
est_properties = zeros(num_materials, num_file_names, num_properties);
real_properties = zeros(num_materials, num_file_names, num_properties);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use one vs. all multiclass classification.
for i = 1:num_properties,
    % Get real labels of the images for this property.
    real_labels = cell_real_properties{i}{2}';

    % Use the SVM linear classifiers to classify the test data.
    % SVM cell structure: 'scale', 'feature', weight vector (W), bias (B).
    W = svms{i}{3};
    B = svms{i}{4};
    scores = (W' * features_2d) + B;

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == -1) ) = 0;
    real_labels( find(real_labels == -1) ) = 0;

    % Store the estimated and real properties of each image using binary vectors.
    est_properties(:,:,i) = reshape(estimated_labels, [num_file_names, num_materials])'; 
    real_properties(:,:,i) = reshape(real_labels, [num_file_names, num_materials])';

    % Testing accuracy in the test set.
    accuracy = sum(real_labels == sign(estimated_labels)) / length(real_labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

if verbose >= 1
    % Print the resulting test accuracies.
    mean_accuracy = (mean_accuracy / num_properties);
    fprintf(1, 'TEST SET:\n');
    fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
    fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
    fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);
end

if verbose >= 2
    % Print some statistics about the current estimated properties.
    print_estimated_properties_stats (materials, file_names, num_properties, num_properties_per_image, est_properties);
end


%%% Naive Bayes %%%

%% Predicted data.

% Permute and reshape the class labels and the estimated properties vectors to 
% fit the Naive Bayes fitting function requirements.
material_labels = repmat(materials, 12, 1);
est_properties = permute(est_properties, [3 2 1]);
est_properties = reshape(est_properties, [num_properties, num_images]);

% Train the Naive Bayes classifier with the predicted data.
est_bayes = NaiveBayes.fit(est_properties', material_labels(:), 'Distribution', 'mn');

% Check the accuracy of the results.
indices_well_classified = cellfun(@strcmp, est_bayes.predict(est_properties'), material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, 'Naive Bayes with predicted data:\n')
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', num_correctly_classified, num_correctly_classified*100/num_images);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', num_images-num_correctly_classified, 100-(num_correctly_classified*100/num_images));


%% Ground truth data.

% Permute and reshape the real properties vectors to fit the Naive Bayes fitting
% function requirements.
real_properties = permute(real_properties, [3 2 1]);
real_properties = reshape(real_properties, [num_properties, num_images]);

% Train the Naive Bayes classifier with the predicted data.
real_bayes = NaiveBayes.fit(real_properties', material_labels(:), 'Distribution', 'mn');

% Check the accuracy of the results.
indices_well_classified = cellfun(@strcmp, real_bayes.predict(real_properties'), material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, 'Naive Bayes with ground truth data:\n')
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', num_correctly_classified, num_correctly_classified*100/num_images);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', num_images-num_correctly_classified, 100-(num_correctly_classified*100/num_images));
fprintf(1, '\n');

%keyboard;

%%% SVMs %%%

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
    [W,B,~,scores] = vl_svmtrain(est_properties, labels, lambda, 'Solver', solver);

    % Store everything in the cell data structure and the matrix of weights
    % and vector of bias.
    est_material_svms{i} = {materials(i), W, B};
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
fprintf(1, 'SVMs with predicted data:\n')
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', floor(mean_accuracy*num_images), mean_accuracy*100);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', ceil((1-mean_accuracy)*num_images), 100-(mean_accuracy*100));


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
    [W,B,~,scores] = vl_svmtrain(real_properties, labels, lambda, 'Solver', solver);

    % Store everything in the cell data structure and the matrix of weights
    % and vector of bias.
    real_material_svms{i} = {materials(i), W, B};
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
fprintf(1, 'SVMs with real data:\n');
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', floor(mean_accuracy*num_images), mean_accuracy*100);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', ceil((1-mean_accuracy)*num_images), 100-(mean_accuracy*100));
fprintf(1, '\n');


%%% SVMs with real one vs. all %%%

%% Predicted data.

est_scores = (est_weights_material_svms' * est_properties) + repmat(est_bias_material_svms, [1, num_images]);
[~,I] = max(est_scores);
indices_well_classified = cellfun(@strcmp, materials(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, 'SVMs with one vs. all and predicted data:\n');
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', num_correctly_classified, num_correctly_classified*100/num_images);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', num_images-num_correctly_classified, 100-(num_correctly_classified*100/num_images));

%% Ground truth data.

real_scores = (real_weights_material_svms' * real_properties) + repmat(real_bias_material_svms, [1, num_images]);
[~,I] = max(real_scores);
indices_well_classified = cellfun(@strcmp, materials(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, 'SVMs with one vs. all and ground truth data:\n');
fprintf(1, ' - Correctly classified: %d (%.2f %%)\n', num_correctly_classified, num_correctly_classified*100/num_images);
fprintf(1, ' - NOT-correctly classified: %d (%.2f %%)\n', num_images-num_correctly_classified, 100-(num_correctly_classified*100/num_images));
fprintf(1, '\n');

%keyboard;