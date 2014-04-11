% Print statistics about the given estimated properties.

function print_estimated_properties_stats (materials, file_names, num_properties, num_properties_per_image, est_properties)
    % Variables to improve code legibility. 
    num_materials = length(materials);
    num_file_names = length(file_names);
    num_images = num_materials * num_file_names;

    % Materials, images and properties (real and estimated).
    fprintf(1, 'Number of materials: %d\n', num_materials);
    fprintf(1, 'Number of images per material: %d\n', num_file_names);
    fprintf(1, 'Number of properties per image: %d\n', num_properties_per_image);
    fprintf(1, 'Number of different properties: %d\n', num_properties);
    fprintf(1, '\n');

    fprintf(1, 'Total number of images: %d\n', num_images);
    total_num_properties = num_images * num_properties_per_image;
    fprintf(1, 'Total number of properties: %d\n', total_num_properties);
    total_num_est_properties = sum(est_properties(:));
    fprintf(1, 'Total number of estimated properties: %d\n', total_num_est_properties);
    total_size_est_properties = prod(size(est_properties));
    fprintf(1, 'Total size of vectors of properties: %d\n', total_size_est_properties);
    fprintf(1, '\n');

    fprintf(1, 'Proportion between number of estimated properties and number of properties: %.3f\n', total_num_est_properties/total_num_properties);
    fprintf(1, 'Proportion between number of estimated properties and total size of vectors of properties: %.3f\n', total_num_est_properties/total_size_est_properties);
    fprintf(1, '\n');

    % Check which images of different materials share the same vector of
    % properties.
    fprintf(1, 'List of images that share the same vector of properties:\n');
    same_vector_counter = 0;
    same_prop_matrix = zeros(num_materials, num_file_names);
    for i = 1:num_materials, for j = 1:num_file_names,
        for i2 = i+1:num_materials, for j2 = 1:num_file_names,
            if isequal(est_properties(i,j,:), est_properties(i2,j2,:))
                same_vector_counter = same_vector_counter+1;
                same_prop_matrix(i,j) = same_prop_matrix(i,j) + 1;
                same_prop_matrix(i2,j2) = same_prop_matrix(i2,j2) + 1;
                fprintf(1, '   %3d - %s %s is equal to %s %s\n', same_vector_counter, materials{i}, file_names{j}, materials{i2}, file_names{j2});
            end
        end; end
    end; end

    % Show some information about how the vector of properties are shared.
    num_images_share = length(find(same_prop_matrix));
    fprintf(1, '\n');
    fprintf(1, 'Number of images that share their vector of properties with images from other classes: %d of a total of %d images (%.2f%%).\n', num_images_share, num_images, num_images_share*100/num_images);
    fprintf(1, 'Number of images that don''t share their vector of properties with images from other classes: %d of a total of %d images (%.2f%%).\n', num_images-num_images_share, num_images, 100-(num_images_share*100/num_images));
    fprintf(1, '\n');


    % Get the ranking of the materials by number of vectors of properties shared.
    fprintf(1, 'Ranking of the materials by number of vectors of properties shared:\n')
    vector = sum(same_prop_matrix');
    [X, order] = sort(vector(:), 1, 'descend');
    disp(strcat(num2str(X), ' -', {' '}, materials(order)'));

    % Get the average times that a material share their properties vector.
    avg_shared_material_times = mean(sum(same_prop_matrix'));
    fprintf(1, 'The average times that a material share their properties vector is %.2f.\n', avg_shared_material_times)
    fprintf(1, '\n');

    % Get the ranking of the images by the number of times its vector of properties 
    % is shared.
    fprintf(1, 'Ranking of the images by the number of times its vector of properties is shared:\n')
    matrix = same_prop_matrix';
    [X,order] = sort(matrix(:), 1, 'descend');
    [J,I] = ind2sub(size(matrix), order);
    disp(strcat(num2str(X), ' -', {' '}, materials(I)', {' '}, file_names(J)'));

    % Get the average times that an image shares its properties vector.
    avg_shared_image_times = mean(same_prop_matrix(:));
    fprintf(1, 'The average times that an image shares its properties vector is %.2f.\n', avg_shared_image_times)
    fprintf(1, '\n');
end