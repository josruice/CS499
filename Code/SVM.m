function [estimatedLabelsCellArray, svmsCellArray] = svm (featuresCellArray, realClassLabelsCellArray, modality, varargin)
% svm Trains or tests SVMs using the given sample features and labels.

% Load constants file.
loadConstants;

% Create arguments parser.
parser = inputParser;

% Add required and parametrized arguments.
parser.addRequired(FEATURES_PARAM, @(x) length(x)>0);
parser.addRequired(LABELS_PARAM, @(x) length(x)>0);
parser.addRequired(MODALITY_PARAM, @isstr);

parser.addParamValue(SVMS_PARAM, DEFAULT_SVMS, @(x) length(x)>0);
parser.addParamValue(BINARY_HISTOGRAMS_PARAM, DEFAULT_BINARY_HISTOGRAMS, ...
                     @islogical);
parser.addParamValue(LAMBDA_PARAM, DEFAULT_LAMBDA, @isnumeric);
parser.addParamValue(SOLVER_PARAM, DEFAULT_SOLVER, @isstr);
parser.addParamValue(LOSS_PARAM, DEFAULT_LOSS, @isstr);
parser.addParamValue(VERBOSE_PARAM, DEFAULT_VERBOSE, @isnumeric);

% Parse input arguments.
parser.parse(featuresCellArray, realClassLabelsCellArray, modality, varargin{:});
inputs = parser.Results;

% Read the arguments.
featuresCellArray = inputs.(FEATURES_PARAM);
realClassLabelsCellArray = inputs.(LABELS_PARAM); 
modality = inputs.(MODALITY_PARAM);

svmsCellArray = inputs.(SVMS_PARAM);
shouldBinaryHistograms = inputs.(BINARY_HISTOGRAMS_PARAM);
lambda = inputs.(LAMBDA_PARAM);
solver = inputs.(SOLVER_PARAM);
loss = inputs.(LOSS_PARAM);
verbose = inputs.(VERBOSE_PARAM);

% Check if the modality given is allowed.
modalityAllowed = any(strcmpi(modality, ALLOWED_MODALITIES));
if not(modalityAllowed)
    error(INVALID_PARAM_ERROR, MODALITY_PARAM);
end

% Variables to improve code legibility.
[nCategories, nSamplesPerCategory, nFeaturesPerSample] = ...
    size(featuresCellArray);
nSamples = nCategories * nSamplesPerCategory;
nClasses = length(realClassLabelsCellArray);

% Convert the samples features cell array into a standard 2D matrix with the 
% samples by columns.
featuresMatrix = cell2mat(cellfun(@(x) x', reshape(featuresCellArray',[],1), ...
                          'UniformOutput', false))';

% Use binary histograms if required.
if shouldBinaryHistograms
    featuresMatrix( find(featuresMatrix) ) = 1;
end

if not(strcmpi(modality, TESTING_MODALITY))
    % Cell structure: {'Class name', SVM weight vector (W), SVM bias (B)}.
    svmsCellArray = cell(nClasses, 1);
end

% Cell array to store the estimated labels of the samples. The i,j cell contains
% a column vectors where k element represents if jth sample of ith category
% belongs to kth class (1 if it belongs, 0 otherwise). 
estimatedLabelsCellArray = cell(nCategories, nSamplesPerCategory);

% Matrix with the same meaning as the above cell array.
estimatedLabelsMatrix = zeros(nCategories, nSamplesPerCategory, nClasses);

% Cell array used to store the name of a class, the accuracy obtained in the 
% classification process and its confusion matrix.
% Structure: {'className', accuracy, confusionMatrix}.
accuraciesCellArray = cell(nClasses, 1);

% If the modality is test, use the given SVMs to classify the specified samples
% feature vectors. Otherwise, use SVMs to create the linear classifiers.
for iClass = 1:nClasses,
    classCell = realClassLabelsCellArray{iClass};
    className = classCell{1};
    classRealBinaryLabels = reshape(classCell{2}', [], 1); % 0 or 1.

    % Convert binary labels to -1 or 1 (SVMs requirement).
    classRealLabels = classRealBinaryLabels;
    classRealLabels( find(classRealBinaryLabels == 0) ) = -1; 

    if strcmpi(modality, TESTING_MODALITY)
        % Use the SVM linear classifiers to classify the test data.
        % Cell structure: {'Class name', SVM weight vector (W), SVM bias (B)}.
        W = svmsCellArray{iClass}{SMVS_CELL_ARRAY_WEIGHT_VECTOR_INDEX};
        B = svmsCellArray{iClass}{SMVS_CELL_ARRAY_BIAS_INDEX};
        scores = (W' * featuresMatrix) + B;
    else
        % Build the classifier for this class.
        [W, B, ~, scores] = vl_svmtrain(featuresMatrix, classRealLabels,    ...
                                        lambda, SOLVER_PARAM, solver,       ...
                                                LOSS_PARAM, loss);

        % Store everything in the cell data structure.
        svmsCellArray{iClass} = cell(1, SMVS_CELL_ARRAY_NUM_ELEMENTS);
        svmsCellArray{iClass}{SMVS_CELL_ARRAY_CLASS_NAME_INDEX} = className;
        svmsCellArray{iClass}{SMVS_CELL_ARRAY_WEIGHT_VECTOR_INDEX} = W;
        svmsCellArray{iClass}{SMVS_CELL_ARRAY_BIAS_INDEX} = B;
    end

    % Elements with score 0 (on the line of the linear classifier) are
    % the same as negative (don't belong to the class).
    classEstimatedLabels = sign(scores);    % Returns -1, 0 or 1.
    classEstimatedBinaryLabels = classEstimatedLabels;
    classEstimatedBinaryLabels( find(classEstimatedLabels == -1) ) = 0;
    
    % Store the estimated labels of each sample using binary vectors.
    estimatedLabelsMatrix(:,:,iClass) = reshape(classEstimatedBinaryLabels, ...
                                        [nSamplesPerCategory, nCategories])'; 

    % Compute the confusion matrix and the accuracy. Store them in a cell array.
    confusionMatrix = accumarray([classRealBinaryLabels,                    ...
                                 classEstimatedBinaryLabels']+1, 1, [2 2])  ...
                      ./ nSamples;
    accuracy = sum(diag(confusionMatrix));
    accuraciesCellArray{iClass} = {className, accuracy, confusionMatrix};

    if verbose >= 3
        % Display confusion matrix for this class.
        fprintf(STDOUT, '\n%s confusion:\n', className);
        disp(confusionMatrix);
    end
end

% Convert the estimated labels matrix into a cell array with column vector 
% cells.
estimatedLabelsCellArray = mat2cell(estimatedLabelsMatrix, ...
                                    ones(nCategories,1),   ...
                                    ones(nSamplesPerCategory,1), nClasses);
estimatedLabelsCellArray = cellfun(@(x) reshape(x, [], 1),   ...
                                   estimatedLabelsCellArray, ...
                                   'UniformOutput', false);

if verbose >= 1
    % Print the resulting accuracies.
    accuraciesArray = extractFromCell(accuraciesCellArray, 2);
    fprintf(STDOUT, 'Mean accuracy: %.2f. ', mean(accuraciesArray) * 100);
    %fprintf(STDOUT, '%s:\n', modality);
    %fprintf(STDOUT, ' - Mean accuracy: %.2f\n', mean(accuraciesArray) * 100);
    %fprintf(STDOUT, ' - Min accuracy: %.2f\n', min(accuraciesArray) * 100);
    %fprintf(STDOUT, ' - Max accuracy: %.2f\n\n', max(accuraciesArray) * 100);

    if verbose >= 2
        % Plot chart of accuracies.
        classNameCellArray = cellfun(@(x) x{1}, accuraciesCellArray, ...
                                     'UniformOutput', false);
        confusionsMatrix = extractFromCell(accuraciesCellArray, 3);

        % Global accuracy.
        figureName = 'Properties detection global accuracy';
        xArray = accuraciesArray;
        xLabelsCellArray = classNameCellArray;
        color = 'y';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);

        % Sorted global accuracies.
        [~,I] = sortrows(extractFromCell(accuraciesCellArray, 2));
        sortedByAccuraciesCellArray = accuraciesCellArray(I);
        sortedClassNameCellArray = cellfun(@(x) x{1},                   ...
                                           sortedByAccuraciesCellArray, ...
                                           'UniformOutput', false);
        sortedAccuraciesArray = extractFromCell(sortedByAccuraciesCellArray, 2);

        figureName = 'Sorted properties detection global accuracy';
        xArray = sortedAccuraciesArray;
        xLabelsCellArray = sortedClassNameCellArray;
        color = 'y';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);

        % Recall.
        figureName = 'Properties detection recall';
        xArray = reshape(confusionsMatrix(2,2,:) ./ ...
                    (confusionsMatrix(2,1,:) + confusionsMatrix(2,2,:)), [], 1);
        xLabelsCellArray = classNameCellArray;
        color = 'b';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);

        % Sorted recall.
        [sortedRecallArray,I] = sortrows(cellfun(@(x) ...
                  x{3}(2,2,:)./(x{3}(2,1,:)+x{3}(2,2,:)), accuraciesCellArray));
        sortedRecallCellArray = accuraciesCellArray(I);
        sortedRecallClassNameCellArray = cellfun(@(x) x{1},                 ...
                                        sortedRecallCellArray,              ...
                                        'UniformOutput', false);

        figureName = 'Sorted properties detection recall';
        xArray = sortedRecallArray;
        xLabelsCellArray = sortedRecallClassNameCellArray;
        color = 'b';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);

        % Precision.
        figureName = 'Properties detection precision';
        xArray = reshape(confusionsMatrix(2,2,:) ./ ...
                    (confusionsMatrix(1,2,:) + confusionsMatrix(2,2,:)), [], 1);
        xLabelsCellArray = classNameCellArray;
        color = 'g';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);

        % Sorted precision.
        [sortedPrecisionArray,I] = sortrows(cellfun(@(x) ...
                  x{3}(2,2,:)./(x{3}(1,2,:)+x{3}(2,2,:)), accuraciesCellArray));
        sortedPrecisionCellArray = accuraciesCellArray(I);
        sortedPrecisionClassNameCellArray = cellfun(@(x) x{1},                 ...
                                        sortedPrecisionCellArray,              ...
                                        'UniformOutput', false);

        figureName = 'Sorted properties detection precision';
        xArray = sortedPrecisionArray;
        xLabelsCellArray = sortedPrecisionClassNameCellArray;
        color = 'g';
        plotAccuraciesBarGraph(figureName, xArray, xLabelsCellArray, color);
    end
end