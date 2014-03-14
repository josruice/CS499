% TODO: Write proper documentation.

function [features] = quantize_feature_vectors (descriptors, total_descriptors, num_clusters)
    % Get the size of the descriptors cell.
    [num_rows num_columns] = size(descriptors);

    % Convert the descriptors cell array into a standard 2D single matrix with the
    % descriptors in the columns.
    [d_matrix, num_descriptors] = descriptors_cell_to_single_matrix(descriptors, total_descriptors);

    % Build the clusters applying k-means clustering to the descriptors matrix.
    [clusters, ~] = vl_kmeans(d_matrix, num_clusters);

    % Create a kd-tree with the clusters.
    kd_tree = vl_kdtreebuild(clusters);

    % Obtain the nearest neighbour (cluster center, in this case) of each column 
    % (descriptor) of the descriptors matrix. 
    [indices, dist] = vl_kdtreequery(kd_tree, clusters, d_matrix);
    % The output column vector indices contains the results sorted by folder and
    % image, due to the way the input matrix have been built. 

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

            % TODO: normalize feature values.

            % The indices vector can be chopped out in this way because the 
            % materials and images are in order inside the vector.
            indices = indices(num_descriptors{i,j}+1:end);
        end
    end
end