function featuresCellArray = readFeatures(filePath, nClasses, nSamplesPerClass)
% readFeatures Load the features from a file.
%   
%   featuresCellArray = readFeatures(filePath, nClasses, nSamplesPerClass) loads 
%   the features of a dataset with the nClasses number of classes and 
%   nSamplesPerClass samples per class from the specified file into the cell
%   array featuresCellArray.
% 
%   The structure of the output cell array will be a column vector of cell 
%   arrays with one row per different feature, where each of the cells store the
%   name of the feature and a binary matrix where the i,j entry represents if
%   jth sample of ith class has the feature.

% Load constants file.
loadConstants;

% Read the file by lines.
file = fopen(filePath);
linesCellArray = textscan(file, '%[^\n]');
linesCellArray = linesCellArray{1}; % Content is in the first cell.
fclose(file);

% Initalize a map where the keys will be the feature names and the values will 
% be arrays containing the global sample index (index inside the dataset) with
% these properties.
keys = {''};
values = {[]};
map = containers.Map(keys, values);

% Cover all the lines storing the features of each sample.
% Right now, classes are required to have the same number of samples.
iLine = 1;
for iClass = 1:nClasses,
    % Read the class name line (in case it is required).
    className = linesCellArray{iLine};
    iLine = iLine+1;

    for jClassSample = 1:nSamplesPerClass,
        % Compute the global index of this sample.
        iGlobal = (iClass-1)*nSamplesPerClass + jClassSample;

        % Split the data line by words.
        wordsCellArray = textscan(linesCellArray{iLine}, '%s');
        wordsCellArray = wordsCellArray{1}; % Content is in the first cell.

        % Read the content.
        file_name = wordsCellArray{MARKUP_FILE_NAME_INDEX};
        for kWord = MARKUP_FILE_NAME_INDEX+1:length(wordsCellArray),
            property = wordsCellArray{kWord};

            % Check if the feature has already been added to the map.
            if isKey(map, property)
                % It is already in the map. Update the value array.
                map(property) = [map(property), iGlobal];
            else
                % It is still not in the map. Add it.
                map(property) = iGlobal;
            end
        end
        
        iLine = iLine+1;
    end
end

% Delete the element used to give the datatypes to the map.
remove(map, '');

% At this point the map object is filled with features.
% It is time to create the labels vector. All the data will be stored in a cell
% array where each slot will contain the feature name and the labels matrix.
keys = map.keys();
nFeatures = length(map);
featuresCellArray = cell(nFeatures,1);

for iFeature = 1:nFeatures,
    % Feature name.
    name = keys{iFeature};

    % Feature labels. 
    %  - First all the labels are set to 0 (row -> sample, column -> class).
    labels = zeros(nSamplesPerClass, nClasses);

    %  - Then, those samples that have this property are set to 1.
    labels( map(name) ) = 1;

    %  - Traspose the labels matrix (row -> class, column -> sample).
    labels = labels';

    % Store the data of this property.
    featuresCellArray{iFeature} = {name, labels};
end