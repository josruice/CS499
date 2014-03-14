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

% Data.
root_path = '../Dataset'; % Without last slash.

materials = {'Birch', 'Brick', 'Concrete', 'Corduroy', 'Denim', 'Elm', 'Feathers', 'Fur', 'Hair', 'KnitAran', 'KnitGuernsey', 'Leather', 'Marble', 'Scale', 'Silk', 'Slate', 'Stucco', 'Velour'};

file_names = {'01','02','03','04','05','06','07','08','09','10','11','12'};

img_format = 'png';

markup_file = '../Markups/Machine-Markup-(1.0).txt';

% Classifiers.
feature_method = 'SIFT'; % PHOW, SIFT or DSIFT.

% Number of clusters used in the K-means.
num_clusters = 200; 

% Support Vector Machine (SVM) solver.
solver = 'SDCA';


%%%%%%%%%%%%%%%%%%
%%%   Script   %%%
%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');

% Variables to improve code legibility.
num_materials = length(materials);
num_file_names = length(file_names);


%%% TRAINING %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'FIRST HALF', feature_method);

% Quantize the feature descriptors using k-means.
[features] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

% Read markup data.
[cell_properties] = read_markup(markup_file, num_materials, num_file_names);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features = permute(features, [2 1 3]);
features = reshape(features, [num_materials*num_file_names, num_clusters])';

%keyboard;

% Cell array formed by cell arrays each with 4 elements: 
% Scale string, feature string, SVM weight vector, SVM offset.
num_properties = length(cell_properties);
svms = cell(num_properties, 1);

% Variables used to test the accuracy.
min_accuracy = 1;
max_accuracy = 0;
mean_accuracy = 0;

% Use SVM to create the linear clasifiers for the properties.
for i = 1:num_properties,
    prop = cell_properties{i};
    prop_name = prop{1};
    labels = prop{2};

    % Get scale and feature of the property.
    scale_and_feature = strsplit(prop_name);
    scale = scale_and_feature(1);
    feature = scale_and_feature(1);

    % Build the classifier for this property.
    [W,B,~,scores] = vl_svmtrain(features, labels, 0.1, 'Solver', solver);

    % Store everything in the cell data structure.
    svms{i} = {scale, feature, W, B};

    % Testing accuracy in the training set.
    accuracy = sum(labels == sign(scores')) / length(labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

% Print the resulting accuracies.
mean_accuracy = (mean_accuracy / num_properties);
fprintf(1, 'TRAINING SET:\n');
fprintf(1, 'Mean accuracy: %.2f\n', mean_accuracy * 100);
fprintf(1, 'Min accuracy: %.2f\n', min_accuracy * 100);
fprintf(1, 'Max accuracy: %.2f\n', max_accuracy * 100);


%%% TESTING %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'SECOND HALF', feature_method);

% Quantize the feature descriptors using k-means.
[features] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

keyboard;