% TODO: Write proper documentation.

function [features, clusters, num_clusters] = quantize_feature_vectors (descriptors_in, total_descriptors_in, num_clusters_in, varargin)

    % Constants.
    UINT8_DATATYPE = 'uint8';
    SINGLE_DATATYPE = 'single';

    DEFAULT_DATATYPE = UINT8_DATATYPE;
    DEFAULT_HIERARCHICAL = false;
    DEFAULT_BRANCHING_FACTOR = 100;

    % Create arguments parser.
    parser = inputParser;

    % Add required and parametrized arguments.
    parser.addRequired('Descriptors', @(x) length(x)>0);
    parser.addRequired('TotalDescriptors', @(x) (x)>1);
    parser.addRequired('NumClusters', @(x) (x)>0);

    parser.addParamValue('Clusters', [], @(x) length(x)>0);
    parser.addParamValue('Datatype', DEFAULT_DATATYPE, @isstr);
    parser.addParamValue('Hierarchical', DEFAULT_HIERARCHICAL, @islogical);
    parser.addParamValue('BranchingFactor', DEFAULT_BRANCHING_FACTOR, @(x) (x)>1);

    % Parse input arguments.
    parser.parse(descriptors_in, total_descriptors_in, num_clusters_in, varargin{:});

    % Read the arguments.
    inputs = parser.Results;
    descriptors = inputs.Descriptors;
    total_descriptors = inputs.TotalDescriptors;
    num_clusters = inputs.NumClusters;

    clusters = inputs.Clusters;
    datatype = inputs.Datatype;
    hierarchical = inputs.Hierarchical;
    branching_factor = inputs.BranchingFactor;

    % Get the size of the descriptors cell.
    [num_rows num_columns] = size(descriptors);

    % If no clusters have been given, build the clusters applying k-means 
    % clustering to the descriptors matrix and obtain the center associated with
    % each descriptor.
    % Otherwise, use the given clusters to obtain the center associated with each
    % descriptor.
    if strcmp(datatype, UINT8_DATATYPE)
        % Convert the descriptors cell array into a standard 2D matrix with the
        % descriptors in the columns and the required datatype.
        [d_matrix, num_descriptors] = descriptors_cell_to_matrix(descriptors, total_descriptors, UINT8_DATATYPE);
        
        if hierarchical
            if isempty(clusters)
                [clusters, asgn] = vl_hikmeans(d_matrix, branching_factor, num_clusters);
            else
                asgn = vl_hikmeanspush(clusters, d_matrix);
            end
            
            % Cover the assignments tree to convert it into a list.
            num_clusters = branching_factor;
            for i = 2:size(asgn,1),
                asgn(i,:) = ( asgn(i-1,:)-1 )*branching_factor + asgn(i,:);
                num_clusters = num_clusters * branching_factor;
            end
            indices = asgn(end,:);
        else
            if isempty(clusters)
                [clusters, indices] = vl_ikmeans(d_matrix, num_clusters);
            else
                indices = vl_ikmeanspush(d_matrix, clusters); 
            end
        end
    else
        if not( strcmp(datatype, SINGLE_DATATYPE) )
            fprintf(1, 'Wrong parameters. Default execution.\n');
        end

        % Convert the descriptors cell array into a standard 2D matrix with the
        % descriptors in the columns and the required datatype.
        [d_matrix, num_descriptors] = descriptors_cell_to_matrix(descriptors, total_descriptors, SINGLE_DATATYPE);

        if isempty(clusters)
            [clusters, indices] = vl_kmeans(d_matrix, num_clusters);  
        else
            % Create a kd-tree with the clusters.
            kd_tree = vl_kdtreebuild(clusters);

            % Obtain the nearest neighbour (cluster center, in this case) of each 
            % column (descriptor) of the descriptors matrix. 
            [indices, ~] = vl_kdtreequery(kd_tree, clusters, d_matrix);
        end
    end
    % The output column vector indices contains the results sorted by folder 
    % and image, due to the way the input matrix have been built.       

    % Build the feature vectors of the images ans store the result in a 3D matrix
    % where the rows and columns are the images in the same positions as the 
    % input descriptors matrix and the third dimension are the feature values.
    features = zeros(num_rows, num_columns, num_clusters, 'single');
    for i = 1:num_rows,
        for j = 1:num_columns,
            % The closest neighbours of the feature descriptors are counted for
            % each image and the resulting array of size num clusters is the SVM
            % feature vector.
            feat_vector = accumarray(indices(1:num_descriptors{i,j})', 1);
            features(i, j, 1:size(feat_vector,1)) = feat_vector;

            % The indices vector can be chopped out in this way because the 
            % materials and images are in order inside the vector.
            indices = indices(num_descriptors{i,j}+1:end);
        end
    end
end