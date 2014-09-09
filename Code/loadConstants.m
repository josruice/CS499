%%%%%%%%%%%%%%%%%%%%%
%%%    GENERAL    %%%
%%%%%%%%%%%%%%%%%%%%%

% Classification standard parameters.
FEATURES_PARAM = 'Features';
LABELS_PARAM = 'Labels';
NUMBER_OF_CLASSES_PARAM = 'NumberOfClasses';
MODALITY_PARAM = 'Modality';

% Classification modalities.
TRAINING_MODALITY = 'Training';
TESTING_MODALITY = 'Testing';
ALLOWED_MODALITIES = {TRAINING_MODALITY, TESTING_MODALITY};

% Standard out file descriptor.
STDOUT = 1;

% Matlab datatypes.
UINT8_DATATYPE = 'uint8';
SINGLE_DATATYPE = 'single';

% Verbose parameter name and default value.
VERBOSE_PARAM = 'Verbose';
DEFAULT_VERBOSE = 0;


%%%%%%%%%%%%%%%%%%%%
%%%    Markup    %%%
%%%%%%%%%%%%%%%%%%%%

% Markup file path.
MARKUP_FILE = '../Markups/Machine-Markup-(1.0).txt';

% Word index of the file name in every feature line of the markup file.
MARKUP_FILE_NAME_INDEX = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Descriptors    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter names.
FILE_PATHS_PARAM = 'FilePaths'; % Required.
IMAGE_PART_PARAM = 'ImagePart';
FEATURE_EXTRACTOR_PARAM = 'FeatureExtractor';
MAX_DESCRIPTORS_PER_IMAGE_PARAM = 'MaxDescriptorsPerImage';

% Image part possible values and warning in case of none of this used.
IMAGE_LEFT_HALF = 'LeftHalf';
IMAGE_RIGHT_HALF = 'RightHalf';
IMAGE_COMPLETE = 'Complete';

ALLOWED_IMAGE_PARTS = {IMAGE_LEFT_HALF,  ...
                       IMAGE_RIGHT_HALF, ...
                       IMAGE_COMPLETE};
DEFAULT_IMAGE_PART = IMAGE_COMPLETE;

% Feature extractor possible values, allowed values, default and warning.
FEATURE_EXTRACTOR_PHOW = 'PHOW';
FEATURE_EXTRACTOR_SIFT = 'SIFT';
FEATURE_EXTRACTOR_DSIFT = 'DSIFT';

ALLOWED_FEATURE_EXTRACTORS = {FEATURE_EXTRACTOR_PHOW, ...
                              FEATURE_EXTRACTOR_SIFT, ...
                              FEATURE_EXTRACTOR_DSIFT};
DEFAULT_FEATURE_EXTRACTOR = FEATURE_EXTRACTOR_SIFT;

% Max descriptors default and infinite value.
INFINITE_DESCRIPTORS_PER_IMAGE = 0;
DEFAULT_MAX_DESCRIPTORS_PER_IMAGE = INFINITE_DESCRIPTORS_PER_IMAGE;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Vector quantization  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter names.
VECTORS_CELL_ARRAY_PARAM = 'VectorsCellArray';  % Required.
NUM_CLUSTERS_PARAM = 'NumClusters';             % Required.
CLUSTERS_CENTERS_PARAM = 'ClustersCenters';
KMEANS_DATATYPE_PARAM = 'KmeansDatatype';
KMEANS_HIERARCHICAL_PARAM = 'KmeansHierarchical';
KMEANS_BRANCHING_FACTOR_PARAM = 'KmeansBranchingFactor';

% Default parameter values.
DEFAULT_CLUSTERS_CENTERS = [];
DEFAULT_KMEANS_DATATYPE = SINGLE_DATATYPE;
DEFAULT_KMEANS_HIERARCHICAL = false;
DEFAULT_KMEANS_BRANCHING_FACTOR = 10;

% Allowed datatype possible values.
ALLOWED_KMEANS_DATATYPES = {UINT8_DATATYPE, SINGLE_DATATYPE};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Support Vector Machines  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter names.
FEATURES_PARAM;     % Required.
LABELS_PARAM;       % Required.
MODALITY_PARAM;     % Required.
SVMS_PARAM = 'SVMs';
BINARY_FEATURES_PARAM = 'BinaryFeatures';
LAMBDA_PARAM = 'Lambda';
SOLVER_PARAM = 'Solver';
LOSS_PARAM = 'Loss';

% Default parameter values.
DEFAULT_SVMS = {};
DEFAULT_BINARY_FEATURES = false;
DEFAULT_LAMBDA = 0.01;
DEFAULT_SOLVER = 'SDCA';
DEFAULT_LOSS = 'Logistic';

% SVMs cell array indices.
SMVS_CELL_ARRAY_NUM_ELEMENTS = 3;
SMVS_CELL_ARRAY_CLASS_NAME_INDEX = 1;
SMVS_CELL_ARRAY_WEIGHT_VECTOR_INDEX = 2;
SMVS_CELL_ARRAY_BIAS_INDEX = 3;


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Naive Bayes    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter names.
FEATURES_PARAM;     % Required.
LABELS_PARAM;       % Required.
MODALITY_PARAM;     % Required.
BAYES_CLASSIFIER_PARAM = 'BayesClassifier';

% Default parameter values.
DEFAULT_BAYES_CLASSIFIER = {};

% Internal parameters.
BAYES_DISTRIBUTION_INTERNAL_PARAM = 'Distribution';
BAYES_MULTINOMIAL_DISTRIBUTION = 'mn';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Errors and warnings messages  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reading image error.
READING_FILENAME_ERROR = 'Error reading file %s.\n';

% Generic warning for invalid parameter value replaced by default value.
INVALID_PARAM_DEFAULT_VALUE_WARNING = ['Invalid value for parameter: %s. '  ...
                                       'Using default value: %s.\n'];

% Generic error for invalid parameter value.
INVALID_PARAM_ERROR = 'Invalid value for parameter: %s.\n';

