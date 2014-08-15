function [quantizedVectorsCellArray, clusterCentersMatrix, nClusters] = ...
    quantizeVectors (vectorsCellArray, nClusters, varargin)
% quantizeVectors Quantize the given cell array of vectors.
%
%   quantizedVectorsCellArray = quantizeVectors (vectorsCellArray, nClusters)
%   returns a cell array where each cell contains a column vector that repre-
%   sents the quantization of the vectors of the equivalent one of the
%   vectorsCellArray cell array after computing the cluster centers with all the
%   vectors of vectorsCellArray.
%
%   [quantizedVectorsCellArray, clusterCentersMatrix, nClusters] = 
%       quantizeVectors (vectorsCellArray, nClusters)
%   returns the quantized vectors, a 2D matrix the clusters centers in the 
%   columns and the real number of clusters produced. This can change from the
%   inputed number of clusters in the case of hierarchical K-means.
%
%   quantizeVectors() accepts the following options:
%
%   ClustersCenters::
%       A 2D matrix with clusters centers in the columns as the one produced by
%       this function. If given, the vector quantization will be done using 
%       these clusters centers.
%
%   KmeansDatatype:: single
%       The datatype of the matrix used in K-means. One of single or uint8.
%       Datatype uint8 increases significantly the computation time, altough it
%       allows the use of hierarchical K-means. 
%       
%   KmeansHierarchical:: false
%       Use of hierarchical K-means.
%       
%   KmeansBranchingFactor:: 10
%       Only with hierarchical K-means. Branching factor of the hierarchical
%       K-means tree.

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(VECTORS_CELL_ARRAY_PARAM, @(x) length(x)>0);
parser.addRequired(NUM_CLUSTERS_PARAM, @(x) (x)>0);
parser.addParamValue(CLUSTERS_CENTERS_PARAM, DEFAULT_CLUSTERS_CENTERS,       ...
                     @(x) length(x)>0);
parser.addParamValue(KMEANS_DATATYPE_PARAM, DEFAULT_KMEANS_DATATYPE, @isstr);
parser.addParamValue(KMEANS_HIERARCHICAL_PARAM, DEFAULT_KMEANS_HIERARCHICAL, ...
                     @islogical);
parser.addParamValue(KMEANS_BRANCHING_FACTOR_PARAM,                          ...
                     DEFAULT_KMEANS_BRANCHING_FACTOR, @(x) (x)>1);

% Parse input arguments.
parser.parse(vectorsCellArray, nClusters, varargin{:});
inputs = parser.Results;

% Read the arguments.
vectorsCellArray = inputs.(VECTORS_CELL_ARRAY_PARAM);
nClusters = inputs.(NUM_CLUSTERS_PARAM);

clusterCentersMatrix = inputs.(CLUSTERS_CENTERS_PARAM);
datatype = inputs.(KMEANS_DATATYPE_PARAM);
shouldHierarchical = inputs.(KMEANS_HIERARCHICAL_PARAM);
branchingFactor = inputs.(KMEANS_BRANCHING_FACTOR_PARAM);

% Check if the K-means datatype given is allowed.
datatypeAllowed = any(strcmpi(datatype, ALLOWED_KMEANS_DATATYPES));
if not(datatypeAllowed)
    % Use default and print a warning.
    datatype = DEFAULT_KMEANS_DATATYPE;
    warning(INVALID_PARAM_DEFAULT_VALUE_WARNING, KMEANS_DATATYPE_PARAM, ...
            DEFAULT_KMEANS_DATATYPE);
end

% Convert the vectors cell array into a standard 2D matrix with the vectors in 
% the columns and the required datatype.
vectorsMatrix = cell2mat(cellfun(@(x) x', vectorsCellArray, ...
                        'UniformOutput', false))';
vectorsMatrix = eval([datatype '(vectorsMatrix)']);

% Keep track of the number of descriptors of each cell of the original vector
% cell array to reconstruct the quantized vectors later.
nDescriptorsVector = cellfun(@(x) size(x,2), vectorsCellArray);

% If no clusters centers have been given, build the clusters applying K-means 
% clustering to the vectors matrix obtained from the given vectors cell array
% and compute the center associated with each vector.
% Otherwise, use the given clusters to obtain the center associated with each
% vector.
switch datatype
case UINT8_DATATYPE
    if shouldHierarchical
        if isempty(clusterCentersMatrix)
            [clusterCentersMatrix, assignmentsArray] = vl_hikmeans(     ...
                vectorsMatrix, branchingFactor, nClusters);
        else
            assignmentsArray = vl_hikmeanspush(clusterCentersMatrix,    ...
                                               vectorsMatrix);
        end
        
        % Cover the assignments tree to convert it into a list.
        nClusters = branchingFactor;
        for i = 2:size(assignmentsArray,1),
            assignmentsArray(i,:) = ((assignmentsArray(i-1,:)-1) * ...
                                    branchingFactor) + assignmentsArray(i,:);
            nClusters = nClusters * branchingFactor;
        end

        indices = assignmentsArray(end,:);
    else
        if isempty(clusterCentersMatrix)
            [clusterCentersMatrix, indices] = vl_ikmeans(vectorsMatrix, num_clusterCentersMatrix);
        else
            indices = vl_ikmeanspush(vectorsMatrix, clusterCentersMatrix); 
        end
    end
case SINGLE_DATATYPE
    if isempty(clusterCentersMatrix)
        [clusterCentersMatrix, indices] = vl_kmeans(vectorsMatrix, nClusters);  
    else
        % Create a kd-tree with the clusters.
        kd_tree = vl_kdtreebuild(clusterCentersMatrix);

        % Obtain the nearest neighbour (cluster center, in this case) of each 
        % column (vector) of the vectors matrix. 
        indices = vl_kdtreequery(kd_tree, clusterCentersMatrix, vectorsMatrix);
    end
otherwise
    % Never supposed to happen.
end

% Build the quantized vectors and store the result in a 2D cell array where each
% cell contains the quantized vector of the equivalent one in the original vec-
% tors cell array.
nElements = length(vectorsCellArray);   
quantizedVectorsCellArray = cell(nElements, 1);
for i = 1:nElements,
    % For each cluster center, it is counted the number of vectors of the 
    % original cell array that belong to the same cluster. The result is a
    % quantized vector.
    quantized = accumarray(indices(1:nDescriptorsVector(i))', 1);
    quantizedVectorsCellArray{i} = zeros(nClusters, 1);
    quantizedVectorsCellArray{i}(1:length(quantized)) = quantized;

    % The indices vector can be chopped out in this way because the elements are
    % in order inside the vector.
    indices = indices(nDescriptorsVector(i)+1:end);
end