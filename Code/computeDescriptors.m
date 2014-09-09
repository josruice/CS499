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
%
%   Verbose:: 0
%       Determines the level of verbosity of the execution.

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
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(filePathsCellArray, varargin{:});
inputs = parser.Results;

% Read the arguments.
filePathsCellArray = inputs.(FILE_PATHS_PARAM);
imagePart = inputs.(IMAGE_PART_PARAM); 
featureExtractor = inputs.(FEATURE_EXTRACTOR_PARAM);
maxDescriptorsPerImage = inputs.(MAX_DESCRIPTORS_PER_IMAGE_PARAM);
verbose = inputs.(VERBOSE_PARAM);

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
        [F, D] = vl_phow(image);
    case FEATURE_EXTRACTOR_SIFT
        if ndims(image) == 3
            image = rgb2gray(image); % Gray scale. SIFT requirement.
        end
        [F, D] = vl_sift(image);
    case FEATURE_EXTRACTOR_DSIFT
        if ndims(image) == 3
            image = rgb2gray(image); % Gray scale. DSIFT requirement.
        end
        [F, D] = vl_dsift(image);
    otherwise
        % Never supposed to happend.
    end

    % Randomly sample the descriptors if they exceed the maximum allowed.
    nDescriptors = size(D, 2);
    if maxDescriptorsPerImage < nDescriptors && ...
       maxDescriptorsPerImage ~= INFINITE_DESCRIPTORS_PER_IMAGE
       indices = randperm(nDescriptors, maxDescriptorsPerImage);
       D = D(:, indices);
       F = F(:, indices);
    end
    
    % Store the descriptors.
    descriptorsCellArray{iFile} = D;

    if verbose >= 3
        % Plot keypoints.
        fig1 = figure('Name', sprintf('%s keypoints', featureExtractor), ...
                      'Position', [100 200 500 500]);
        imshow(image, 'Border', 'tight', 'InitialMagnification', 150);
        hold on;
        f1 = vl_plotframe(F);
        f2 = vl_plotframe(F);
        set(f1, 'color', 'k', 'linewidth', 3);
        set(f2, 'color', 'y', 'linewidth', 2);

        % Plot descriptors.
        fig2 = figure('Name', sprintf('%s descriptors', featureExtractor), ...
                      'Position', [750 200 500 500]);
        imshow(image, 'Border', 'tight', 'InitialMagnification', 150);
        hold on;
        f3 = vl_plotframe(F);
        f4 = vl_plotframe(F);
        set(f3, 'color', 'k', 'linewidth', 3);
        set(f4, 'color', 'y', 'linewidth', 2);
        f5 = vl_plotsiftdescriptor(D, F);
        set(f5, 'color', 'g');
        
        pause;
        close(fig1);
        close(fig2);
    end
end