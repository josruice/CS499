%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    CONTROL PANEL    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% General.
VERBOSE = 1; % Verbose level.

%%% Data.
IMAGES_ROOT_PATH = '../Dataset'; % Without last slash.

CLASSES_CELL_ARRAY = {'Birch', 'Brick', 'Concrete', 'Corduroy', 'Denim',     ...
                      'Elm', 'Feathers', 'Fur', 'Hair', 'KnitAran',          ...
                      'KnitGuernsey', 'Leather', 'Marble', 'Scale', 'Silk',  ...
                      'Slate', 'Stucco', 'Velour'};

SAMPLE_FILENAMES_CELL_ARRAY = {'01', '02', '03', '04', '05', '06', '07',   ...
                               '08', '09', '10', '11', '12'};

SAMPLE_IMAGES_FORMAT = 'png';

%%% Descriptors.
TRAINING_IMAGE_PART = 'LeftHalf';
TEST_IMAGE_PART = 'RightHalf';
FEATURE_EXTRACTOR = 'SIFT';          % One of PHOW, SIFT or DSIFT. 
MAX_DESCRIPTORS_PER_IMAGE = 1000;    % 0 means no maximum.

%%% K-means
NUM_CLUSTERS = 50;            % Min number of clusters obtained.
KMEANS_DATATYPE = 'single';    % Datatype of the matrix: single or uint8.
KMEANS_HIERARCHICAL = false;   % Hierarchical (only with uint8).
KMEANS_BRANCHING_FACTOR = 100;   % Branching factor (only with HIERARCHICAL).

%%% Support Vector Machines.
SOLVER = 'SDCA';    % Solver method: SGD or SDCA.
LOSS = 'Logistic';  % Loss function: Hinge, Hinge2, L1, L2 or LOGISTIC.
LAMBDA = 1e-01;     % Lambda value of the SVM.