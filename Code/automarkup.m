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

num_properties_per_image = 6;

% Classifiers.
feature_method = 'SIFT'; % PHOW, SIFT or DSIFT.

% Number of clusters used in the K-means.
num_clusters = 300; 

% Support Vector Machine (SVM) solver.
solver = 'SDCA'; % SGD or SDCA.

% Lambda value of the SVM.
lambda = 0.05;


%%%%%%%%%%%%%%%%%%
%%%   Script   %%%
%%%%%%%%%%%%%%%%%%

% Load VLFeat library.
run('/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup');

% Variables to improve code legibility.
num_materials = length(materials);
num_file_names = length(file_names);
num_images = num_materials * num_file_names;

%%% TRAINING %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'FIRST HALF', feature_method);

% Quantize the feature descriptors using k-means.
[features_3d] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

% Read markup data. The structure of the output cell array will be a column
% vector of cell arrays with one row per different property, where each of the
% elements store the name of the property (scale-property_name) and a column
% vector with the images that have the property. 
[cell_real_properties] = read_markup(markup_file, num_materials, num_file_names);

% Permute and reshape the features to fit the Support Vector Machine (SVM)
% requirements: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_materials*num_file_names]);
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

% Print the resulting training accuracies.
mean_accuracy = (mean_accuracy / num_properties);
fprintf(1, 'TRAINING SET:\n');
fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);


%%% TESTING %%%

% Get the descriptors of the first half of each image in this set of images.
[descriptors, total_descriptors] = get_descriptors(root_path, materials, ...
    file_names, img_format, 'SECOND HALF', feature_method);

% Quantize the feature descriptors using k-means.
[features_3d] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters);

% Permute and reshape the features to classify all of them using matrix 
% operations: one column per example.
features_2d = permute(features_3d, [3 2 1]);
features_2d = reshape(features_2d, [num_clusters, num_materials*num_file_names]);
features_2d( find(features_2d) ) = 1; % Binary histograms.

% 3D matrix to store the binary properties vector of the images. The rows
% represent the materials, the columns the file names and the depth the 
% properties with value 0 (not present) or 1 (present).
est_properties = zeros(num_materials, num_file_names, num_properties);

% Use one vs. all multiclass classification.
for i = 1:num_properties,
    % Get real labels of the images for this property.
    labels = cell_real_properties{i}{2};

    % Use the SVM linear classifiers to classify the test data.
    % SVM cell structure: 'scale', 'feature', weight vector (W), bias (B).
    W = svms{i}{3};
    B = svms{i}{4};
    scores = (W' * features_2d) + B;

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't have the property).
    estimated_labels = sign(scores); % Returns -1, 0 or 1, depending on sign.
    estimated_labels( find(estimated_labels == 0) ) = -1;

    % Store the estimated properties of each image using -1 and 1.
    est_properties(:,:,i) = reshape(estimated_labels, [num_file_names, num_materials])'; 

    % Testing accuracy in the test set.
    accuracy = sum(labels == sign(estimated_labels')) / length(labels);
    mean_accuracy = mean_accuracy + accuracy;
    min_accuracy = min(min_accuracy, accuracy);
    max_accuracy = max(max_accuracy, accuracy);
end

% Print the resulting test accuracies.
mean_accuracy = (mean_accuracy / num_properties);
fprintf(1, 'TEST SET:\n');
fprintf(1, ' - Mean accuracy: %.2f\n', mean_accuracy * 100);
fprintf(1, ' - Min accuracy: %.2f\n', min_accuracy * 100);
fprintf(1, ' - Max accuracy: %.2f\n\n', max_accuracy * 100);

% Convert the estimated properties vectors to binary (-1 -> 0 and 1 -> 1).
est_properties( find(est_properties == -1) ) = 0;

keyboard;

% Print some statistics about the current estimated properties.
%  - Materials, images and properties (real and estimated).
fprintf(1, 'Number of materials: %d\n', num_materials);
fprintf(1, 'Number of images per material: %d\n', num_file_names);
fprintf(1, 'Number of properties per image: %d\n', num_properties_per_image);
fprintf(1, 'Number of different properties: %d\n', num_properties);
fprintf(1, '\n');

fprintf(1, 'Total number of images: %d\n', num_images);
total_num_properties = num_images * num_properties_per_image;
fprintf(1, 'Total number of properties: %d\n', total_num_properties);
total_num_est_properties = sum(est_properties(:));
fprintf(1, 'Total number of estimated properties: %d\n', total_num_est_properties);
total_size_est_properties = prod(size(est_properties));
fprintf(1, 'Total size of vectors of properties: %d\n', total_size_est_properties);
fprintf(1, '\n');

fprintf(1, 'Proportion between number of estimated properties and number of properties: %.3f\n', total_num_est_properties/total_num_properties);
fprintf(1, 'Proportion between number of estimated properties and total size of vectors of properties: %.3f\n', total_num_est_properties/total_size_est_properties);
fprintf(1, '\n');

%  - Check which images of different materials share the same vector of
%    properties.
fprintf(1, 'List of images that share the same vector of properties:\n');
same_vector_counter = 0;
same_prop_matrix = zeros(num_materials, num_file_names);
for i = 1:num_materials, for j = 1:num_file_names,
    for i2 = i+1:num_materials, for j2 = 1:num_file_names,
        if isequal(est_properties(i,j,:), est_properties(i2,j2,:))
            same_vector_counter = same_vector_counter+1;
            same_prop_matrix(i,j) = same_prop_matrix(i,j) + 1;
            same_prop_matrix(i2,j2) = same_prop_matrix(i2,j2) + 1;
            fprintf(1, '   %3d - %s %s is equal to %s %s\n', same_vector_counter, materials{i}, file_names{j}, materials{i2}, file_names{j2});
        end
    end; end
end; end

%  - Show some information about how the vector of properties are shared.
num_images_share = length(find(same_prop_matrix));
fprintf(1, '\n');
fprintf(1, 'Number of images that share their vector of properties with images from other classes: %d of a total of %d images (%.2f%%).\n', num_images_share, num_images, num_images_share*100/num_images);
fprintf(1, 'Number of images that don''t share their vector of properties with images from other classes: %d of a total of %d images (%.2f%%).\n', num_images-num_images_share, num_images, 100-(num_images_share*100/num_images));


%  - Get the materials that share the properties vector the most.
max_shared_material_times = max(sum(same_prop_matrix'));
max_shared_material_names = materials(find(sum(same_prop_matrix') == max_shared_material_times));
fprintf(1, 'The material(s) that share(s) the properties vector of their images the most: \n')
disp(sort(max_shared_material_names)')
fprintf(1, 'with a total of %d times.\n', max_shared_material_times);
fprintf(1, '\n');

%  - Get the materials that share the properties vector the less.
min_shared_material_times = min(sum(same_prop_matrix'));
min_shared_material_names = materials(find(sum(same_prop_matrix') == min_shared_material_times));
fprintf(1, 'The material(s) that share(s) the properties vector of their images the less: \n')
disp(sort(min_shared_material_names)')
fprintf(1, 'with a total of %d times.\n', min_shared_material_times);
fprintf(1, '\n');

%  - Get the average times that a material share their properties vector.
avg_shared_material_times = mean(sum(same_prop_matrix'));
fprintf(1, 'The average times that a material share their properties vector is %.2f.\n', avg_shared_material_times)
fprintf(1, '\n');

%  - Get the images that share the vector of properties the most.
max_shared_image_times = max(same_prop_matrix(:));
[I,J] = ind2sub([num_materials, num_file_names], find(same_prop_matrix == max(same_prop_matrix(:))));
max_shared_image_names = strcat(materials(I), '-', file_names(J));
fprintf(1, 'The image(s) that share(s) the properties vector the most: \n')
disp(sort(max_shared_image_names)');
fprintf(1, 'with a total of %d times.\n', max_shared_image_times);
fprintf(1, '\n');

%  - Get the images that share the vector of properties the less.
min_shared_image_times = min(same_prop_matrix(:));
[I,J] = ind2sub([num_materials, num_file_names], find(same_prop_matrix == min(same_prop_matrix(:))));
min_shared_image_names = strcat(materials(I), '-', file_names(J));
fprintf(1, 'The image(s) that share(s) the properties vector the less: \n')
disp(sort(min_shared_image_names)');
fprintf(1, 'with a total of %d times.\n', min_shared_image_times);
fprintf(1, '\n');

%  - Get the average times that an image shares its properties vector.
avg_shared_image_times = mean(same_prop_matrix(:));
fprintf(1, 'The average times that an image shares its properties vector is %.2f.\n', avg_shared_image_times)
fprintf(1, '\n');

keyboard;