function featuresCellArray = readFeatures(filePath, samplesPerClassArray)
% readFeatures Load the features from a file.
%   
%   featuresCellArray = readFeatures(filePath, samplesPerClassArray) loads the
%   features of a dataset with samplesPerClassArray samples per class from the
%   specified file into the cell array featuresCellArray.
% 
%   The structure of the output cell array will be a column vector of cell
%   arrays with one row per different feature, where each of the cells store the
%   name of the feature and a linear array where each entry represents if that
%   sample has the feature.

% Load constants file.
loadConstants;

% Initalize a map where the keys will be the feature names and the values will 
% be arrays containing the global sample index (index inside the dataset) with
% these properties.
keys = {''};
values = {[]};
map = containers.Map(keys, values);

% Cover all the lines storing the features of each sample.
iGlobal = 1;    % Global index.
file = fopen(filePath);
line = fgetl(file);
while ischar(line),
    % Read the class name line (in case it is required).
    className = line;

    line = fgetl(file);
    while ischar(line) && not(isequal(line, '')),
        % Split the data line by words.
        wordsCellArray = textscan(line, '%s');
        wordsCellArray = wordsCellArray{1}; % Content is in the first cell.

        % Read the content.
        file_name = wordsCellArray{MARKUP_FILE_NAME_INDEX};
        for jWord = MARKUP_FILE_NAME_INDEX+1:length(wordsCellArray),
            property = wordsCellArray{jWord};

            % Check if the feature has already been added to the map.
            if isKey(map, property)
                % It is already in the map. Update the value array.
                map(property) = [map(property), iGlobal];
            else
                % It is still not in the map. Add it.
                map(property) = iGlobal;
            end
        end

        iGlobal = iGlobal+1;
        line = fgetl(file);
    end
    line = fgetl(file);
end
fclose(file);

% Delete the element used to give the datatypes to the map.
remove(map, '');

% At this point the map object is filled with features.
% It is time to create the labels vector. All the data will be stored in a cell
% array where each slot will contain the feature name and the labels array.
keys = map.keys();
nFeatures = length(map);
featuresCellArray = cell(nFeatures,1);

for iFeature = 1:nFeatures,
    % Feature name.
    name = keys{iFeature};

    % Feature labels. 
    %  - First all the labels are set to 0.
    labels = zeros(sum(samplesPerClassArray), 1);

    %  - Then, those samples that have this property are set to 1.
    labels( map(name) ) = 1;

    % Store the data of this property.
    featuresCellArray{iFeature} = {name, labels};
end