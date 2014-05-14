function filePathsCellArray = buildImagesPaths (rootPath, folderNames, ...
                                               fileNames, imagesFormat)
% buildImagesPaths Construct the file paths of a set of images.
%
%   filePathsCellArray = buildImagesPaths (rootPath, folderNames, fileNames, 
%                                         imagesFormat)
%   returns a cell matrix of strings where i,j cell represents the complete path 
%   to image jth image of ith folder.

nFolders = length(folderNames);
nFilesPerFolder = length(fileNames);

% Store the images paths by rows and columns with respect to the folders and 
% file names.
filePathsCellArray = cell(nFolders, nFilesPerFolder);

for iFolder = 1:nFolders,
    for jFileName = 1:nFilesPerFolder,
        % Build image path and store it in the cell array.
        filePath = sprintf('%s/%s/%s.%s', rootPath, folderNames{iFolder},  ...
                                          fileNames{jFileName}, imagesFormat);
        filePathsCellArray{iFolder, jFileName} = filePath; 
    end
end