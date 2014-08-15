function descriptorsCellArray = computeDescriptors (filePathsCellArray, ...
                                                    varargin)
% computeDescriptors Compute the feature descriptors of a set of images.
%
%   descriptorsCellArray = computeDescriptors (filePathsCellArray) returns a 
%   column cell array where each cell contains a matrix with the descriptors by 
%   columns of the image with file path in the equivalent position of the 
%   filePathsCellArray cell array.
%
%   computeDescriptors() accepts the following options:
%
%   ImagePart:: Complete
%       Part of each image used while computing the feature descriptors. One of
%       Complete, FirstHorizontalHalf or SecondHorizontalHalf.
%   
%   FeatureExtractor:: SIFT
%       Feature extractor method used to get the descriptors. One of PHOW, SIFT
%       or DSIFT.
%      
%   MaxDescriptorsPerImage::
%       Maximum number of computed descriptors per image. By default, the 
%       maximum is infinite, equivalent to using value 0.

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(FILE_PATHS_PARAM, @(x) length(x)>0);
parser.addParamValue(IMAGE_PART_PARAM, DEFAULT_IMAGE_PART, @isstr);
parser.addParamValue(FEATURE_EXTRACTOR_PARAM, DEFAULT_FEATURE_EXTRACTOR, ...
                     @isstr);
parser.addParamValue(MAX_DESCRIPTORS_PER_IMAGE_PARAM, ...
                     DEFAULT_MAX_DESCRIPTORS_PER_IMAGE, @isnumeric);

% Parse input arguments.
parser.parse(filePathsCellArray, varargin{:});
inputs = parser.Results;

% Read the arguments.
filePathsCellArray = inputs.(FILE_PATHS_PARAM);
imagePart = inputs.(IMAGE_PART_PARAM); 
featureExtractor = inputs.(FEATURE_EXTRACTOR_PARAM);
maxDescriptorsPerImage = inputs.(MAX_DESCRIPTORS_PER_IMAGE_PARAM);

% Create a cell array to store the descriptor matrices.
nFiles = length(filePathsCellArray);
descriptorsCellArray = cell(nFiles, 1);

for iFile = 1:nFiles,
    % Load the image.
    try
        filePath = filePathsCellArray{iFile};
        image = imread(filePath);
    catch err
        % Error reading the image.
        fprintf(STDOUT, READING_FILENAME_ERROR, filePath);
        continue;
    end

    % Split the image if required.
    [h, w, d] = size(image); % Height, width and depth.
    half_w = floor(w/2);

    % Check if the image part given is allowed.
    imagePartAllowed = any(strcmpi(imagePart, ALLOWED_IMAGE_PARTS));
    if not(imagePartAllowed)
        % Use default and print a warning.
        imagePart = DEFAULT_IMAGE_PART;
        warning(WRONG_PARAM_DEFAULT_VALUE_WARNING, IMAGE_PART_PARAM,    ...
                DEFAULT_IMAGE_PART);
    end

    switch imagePart
    case IMAGE_COMPLETE
        % Do nothing.
    case IMAGE_LEFT_HALF
        image = image(:, 1:half_w, :);
    case IMAGE_RIGHT_HALF
        image = image(:, half_w+1:end, :);
    otherwise
        % Never supposed to happen.
    end

    % Convert the image to SINGLE (feature extractors requirements).
    image = im2single(image);

    % Check if the feature extractor selected is allowed.
    featureExtractorAllowed = any(strcmpi(featureExtractor, ...
                                         ALLOWED_FEATURE_EXTRACTORS));
    if not(featureExtractorAllowed)
        % Use default and print a warning.
        featureExtractorAllowed = DEFAULT_FEATURE_EXTRACTOR;
        warning(INVALID_PARAM_DEFAULT_VALUE_WARNING,  ...
                FEATURE_EXTRACTOR_PARAM, DEFAULT_FEATURE_EXTRACTOR);
    end

    % Get frames and descriptors using a feature extraction method.
    %  - The size of matrix D will be [128 num_descriptors].
    switch featureExtractor
    case FEATURE_EXTRACTOR_PHOW
        [~, D] = vl_phow(image);
    case FEATURE_EXTRACTOR_SIFT
        if ndims(image) == 3
            image = rgb2gray(image); % Gray scale. SIFT requirement.
        end
        [~, D] = vl_sift(image);
    case FEATURE_EXTRACTOR_DSIFT
        if ndims(image) == 3
            image = rgb2gray(image); % Gray scale. DSIFT requirement.
        end
        [~, D] = vl_dsift(image);
    otherwise
        % Never supposed to happend.
    end

    % Randomly sample the descriptors if they exceed the maximum allowed.
    nDescriptors = size(D, 2);
    if maxDescriptorsPerImage < nDescriptors && ...
       maxDescriptorsPerImage ~= INFINITE_DESCRIPTORS_PER_IMAGE
       D = D(:, randperm(nDescriptors, maxDescriptorsPerImage));
    end
       
    % Store the descriptors.
    descriptorsCellArray{iFile} = D;
end