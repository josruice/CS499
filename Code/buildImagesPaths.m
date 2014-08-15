function filePathsCellArray = buildImagesPaths (rootPath, folderNames,   ...
                                                nFilesPerFolderArray)
% buildImagesPaths Construct the file paths of a set of images.
%
%   filePathsCellArray = buildImagesPaths (rootPath, folderNames, 
%                                          filesPerFolderArray)
%   returns a cell matrix of strings where each cell represents the complete  
%   path to an image.

nFolders = length(folderNames);
filePathsCellArray = cell(sum(nFilesPerFolderArray), 1);
iGlobal = 1;
for iFolder = 1:nFolders,
    pathToFolder = strcat(rootPath, '/', folderNames{iFolder});
    listOfFiles = ls(pathToFolder);
    listOfFilesCellArray = strsplit(listOfFiles, '\s', 'DelimiterType', ...
                                    'RegularExpression');
    for jFileName = 1:nFilesPerFolderArray(iFolder),
        filePath = sprintf('%s/%s', pathToFolder, ...
                                    listOfFilesCellArray{jFileName});
        filePathsCellArray{iGlobal} = filePath; 
        iGlobal = iGlobal+1;
    end
end