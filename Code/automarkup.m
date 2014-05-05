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
NUM_CLUSTERS = 500;     % Min number of clusters obtained.
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
% The output matrix contains as many rows and columns as materials an images per 
% material and for each of them a binary vector with the properties.
[cell_real_properties, real_properties, num_properties] = read_markup(MARKUP_FILE, num_materials, num_file_names);


% --------------------------------------------------------------------------- %

disp('Execution data:');
fprintf(1,' - Descriptors: %s\n - Num clusters: %d (datatype %s, hierarchical %d, branching %d)\n - SVM solver: %s (%s loss, lambda %f)\n\n', FEATURE_METHOD, NUM_CLUSTERS, DATATYPE, HIERARCHICAL, BRANCHING_FACTOR, SOLVER, LOSS, LAMBDA);

% --------------------------------------------------------------------------- %

%%%
%%% Training SVMs for properties detection. 
%%%

% Get the descriptors of the first half of each image in this set of images.
disp('Training');
fprintf(1, ' - Descriptors: '); tic;
[training_descriptors, total_training_descriptors] = get_descriptors( ...
    ROOT_PATH, MATERIALS, FILE_NAMES, IMG_FORMAT,                     ...
    'FIRST HALF', FEATURE_METHOD, MAX_DESCRIPTORS_PER_IMAGE);
fprintf('%d descriptors. ', total_training_descriptors); toc;

% Quantize the feature descriptors using k-means.
fprintf(1, ' - Kmeans: '); tic;
[quantized_training_descriptors, clusters, real_num_clusters] =                ...
quantize_feature_vectors (                                                     ...
    training_descriptors, total_training_descriptors, NUM_CLUSTERS,            ...
    DATATYPE_PARAM,     DATATYPE,                                              ...
    HIERARCHICAL_PARAM, HIERARCHICAL,                                          ...
    BRANCHING_PARAM,    BRANCHING_FACTOR);
fprintf('%d clusters. ', real_num_clusters); toc;

% Use SVM to create the linear clasifiers for the properties.
% SVM cell structure: 'scale-feature', weight vector (W), bias (B).
fprintf(1, ' - SMVs: '); tic;
[~, properties_svms] =                                                        ...
SVM (quantized_training_descriptors, cell_real_properties,                    ...
    TRAINING_METHOD,                                                          ...
    LAMBDA_PARAM,  LAMBDA,                                                    ...
    SOLVER_PARAM,  SOLVER,                                                    ...
    LOSS_PARAM,    LOSS,                                                      ...
    VERBOSE_PARAM, VERBOSE);
toc;


% --------------------------------------------------------------------------- %

%%%
%%% Testing SVMs for properties detection.
%%%

% Get the descriptors of the second half of each image in this set of images.
disp('Testing: ');
fprintf(1, ' - Descriptors: '); tic;
[test_descriptors, total_test_descriptors] = get_descriptors(   ...
    ROOT_PATH, MATERIALS, FILE_NAMES, IMG_FORMAT,               ...
    'SECOND HALF', FEATURE_METHOD, MAX_DESCRIPTORS_PER_IMAGE);
fprintf('%d descriptors. ', total_test_descriptors); toc;

% Quantize the feature descriptors using k-means.
fprintf(1, ' - Kmeans: '); tic;
quantized_test_descriptors_3d = quantize_feature_vectors (               ...
    test_descriptors, total_test_descriptors, real_num_clusters,         ...
    CLUSTERS_PARAM,     clusters,                                        ...
    DATATYPE_PARAM,     DATATYPE,                                        ...
    HIERARCHICAL_PARAM, HIERARCHICAL,                                    ...
    BRANCHING_PARAM,    BRANCHING_FACTOR);
toc;

% Use the previously created SVMs with the training data to estimate the 
% vectors of properties of the test images.
fprintf(1, ' - SVMs: '); tic;
est_properties = SVM (quantized_test_descriptors_3d, cell_real_properties, ...
    TEST_METHOD,                                                           ...
    SMVS_PARAM,    properties_svms,                                        ...
    VERBOSE_PARAM, VERBOSE);
toc;


% --------------------------------------------------------------------------- %

%%%
%%% Naive Bayes for material detection.
%%%

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


% --------------------------------------------------------------------------- %

%%%
%%% Training SVMs for material labeling. 
%%%

% Build a cell array with one cell per material with its name and a binary vector
% with the images of that material.
cell_materials = cell(num_materials, 1);
for i = 1:num_materials,
    material_labels = zeros(num_images, 1);
    material_labels((i-1)*num_file_names+1 : i*num_file_names) = 1;
    cell_materials{i} = {MATERIALS{i}, material_labels};
end

% Use SVM to create the linear clasifiers for the materials.
% SVM cell structure: 'material_name', weight vector (W), bias (B).
[~, materials_svms] = SVM (real_properties, cell_materials, TRAINING_METHOD,  ...
    LAMBDA_PARAM,  LAMBDA,                                                    ...
    SOLVER_PARAM,  SOLVER,                                                    ...
    LOSS_PARAM,    LOSS,                                                      ...
    VERBOSE_PARAM, VERBOSE-1);

% --------------------------------------------------------------------------- %

%%%
%%% Testing SVMs (one vs. all) for material labeling. 
%%%

% Permute and reshape the class labels and the properties vectors to fit the
% SVMs requirements.
material_labels = repmat(MATERIALS, num_file_names, 1);
est_properties = permute(est_properties, [3 2 1]);
est_properties = reshape(est_properties, [num_properties, num_images]);
real_properties = permute(real_properties, [3 2 1]);
real_properties = reshape(real_properties, [num_properties, num_images]);

% Matrix with the weight vectors of the material svms as columns.
weights_materials_svms = extract_from_cell (materials_svms, 2);

% Column vector with the bias of the materials svms.
bias_materials_svms = extract_from_cell (materials_svms, 3);

%% Predicted data.
est_scores = (weights_materials_svms' * est_properties) + ...
              repmat(bias_materials_svms, [1, num_images]);
[~,I] = max(est_scores);
confusion = accumarray([ceil([1:num_images]./num_file_names); I]', 1) ./ num_file_names;
plot_confusion(confusion, 'SMVs Confusion - Predicted Data', MATERIALS); 
indices_well_classified = cellfun(@strcmp, MATERIALS(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, '\n[Predicted data] Correct with SVMs (one vs. all): %d out of %d (%.2f %%)\n', num_correctly_classified, num_images, num_correctly_classified*100/num_images);

%% Ground truth data.
real_scores = (weights_materials_svms' * real_properties) + ...
               repmat(bias_materials_svms, [1, num_images]);
[~,I] = max(real_scores);
confusion = accumarray([ceil([1:num_images]./num_file_names); I]', 1) ./ num_file_names;
plot_confusion(confusion, 'SMVs Confusion - Ground Truth', MATERIALS); 
indices_well_classified = cellfun(@strcmp, MATERIALS(I)', material_labels(:));
num_correctly_classified = sum(indices_well_classified);

fprintf(1, '[Ground truth data] Correct with SVMs (one vs. all): %d out of %d (%.2f %%)\n', num_correctly_classified, num_images, num_correctly_classified*100/num_images);

% --------------------------------------------------------------------------- %

