%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    CONTROL PANEL    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% General.
VERBOSE = 0; % Verbose level.

% VLFeat library path.
VLFEAT_LIBRARY_PATH = '/Users/Josevi/Libraries/vlfeat-0.9.18/toolbox/vl_setup';

%%% Data.
IMAGES_ROOT_PATH = '../Dataset'; % Without last slash.

CLASSES_CELL_ARRAY = {'Birch',    'Brick',        'Concrete',               ...
                      'Corduroy', 'Denim',        'Elm',                    ...
                      'Feathers', 'Fur',          'Hair',                   ...
                      'KnitAran', 'KnitGuernsey', 'Leather',                ...
                      'Marble',   'Scale',        'Silk',                   ...
                      'Slate',    'Stucco',       'Velour'};
NUMBER_OF_SAMPLES_PER_CLASS_ARRAY = [12, 12, 12,                            ...
                                     12, 12, 12,                            ...
                                     12, 12, 12,                            ...
                                     12, 12, 12,                            ...
                                     12, 12, 12,                            ...
                                     12, 12, 12];

TEST_PARTITIONS = reshape(1:216, 12, 18)';

%%% Descriptors.
FEATURE_EXTRACTOR = 'PHOW';        % One of PHOW, SIFT or DSIFT. 
MAX_DESCRIPTORS_PER_IMAGE = 2000;  % 0 means no maximum.

%%% K-means
NUM_CLUSTERS = 600;            % Min number of clusters obtained.
KMEANS_DATATYPE = 'single';    % Datatype of the matrix: single or uint8.
KMEANS_HIERARCHICAL = false;   % Hierarchical (only with uint8).
KMEANS_BRANCHING_FACTOR = 100;   % Branching factor (only with HIERARCHICAL).

%%% Support Vector Machines.
SOLVER = 'SDCA';    % Solver method: SGD or SDCA.
LOSS = 'Logistic';  % Loss function: Hinge, Hinge2, L1, L2 or LOGISTIC.
LAMBDA = 1e-03;     % Lambda value of the SVM.