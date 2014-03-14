% TODO: Write proper documentation.

function [cell_properties] = read_markup(file_path, num_materials, num_file_names)
    % Open the file with reading permissions and create a file descriptor.
    file = fopen(file_path);

    % Read the file by lines.
    cell_lines = textscan(file, '%[^\n]');
    cell_lines = cell_lines{1}; % All the content is in the first cell.

    % Close the file.
    fclose(file);

    % Initalize a map where the keys will be the properties in the form
    % <scale>-<feature> and the values will be arrays containing the global
    % image index (index inside the dataset) with these properties.
    keys = {''};
    values = {[]};
    map = containers.Map(keys, values);

    % Cover all the lines storing the properties of each material image.
    % Right now, materials are required to have the same number of images.
    i = 1;

    % For each material.
    for material_index = 1:num_materials,
        material_name = cell_lines{i};
        i = i+1;
        % For each image of the material.
        for file_name_index = 1:num_file_names,
            % Compute the global index of this image.
            global_index = (material_index-1)*num_file_names + file_name_index;

            % Split the data line by words.
            cell_words = textscan(cell_lines{i}, '%s');
            cell_words = cell_words{1}; % All the content is in the first cell.

            % Get the content.
            file_name = cell_words{1}; % First element is always the file name.
            for k = 2:length(cell_words),
                property = cell_words{k};

                % Check if the property has already been added to the map.
                if isKey(map, property)
                    % It is already in the map. Update the value array.
                    map(property) = [map(property), global_index];
                else
                    % It is still not in the map. Add it.
                    map(property) = global_index;
                end
            end
            
            % Update the line counter.
            i = i+1;
        end
    end

    % Delete the element used to give the datatypes to the map.
    remove(map,'');

    % At this point the map object is filled with data properties.
    % It is time to create the labels vector. All the data will be stored in a
    % cell matrix where each slot will contain the property name and the labels
    % vector.
    num_properties = length(map);
    cell_properties = cell(num_properties,1);
    keys = map.keys();
    for i = 1:num_properties,
        % Property name.
        name = keys{i};

        % Property labels. 
        %  - First all the labels are set to -1.
        labels = -ones(num_materials*num_file_names, 1);

        %  - Then, those material images that have this property are set to 1.
        labels( map(name) ) = 1;

        % Store the data of this property.
        cell_properties{i} = {name, labels};
    end
end