function plotAccuraciesBarGraph (figureName, xArray, xLabelsCellArray, color)
figure('Name', figureName, 'Position', [200 200 1200 525]); 
bar(xArray, color);
axis([0, length(xArray)+1, 0, 1]);
set(0,'DefaultTextInterpreter','none');
set(gca, 'XTick', [1:length(xArray)], 'XTickLabel', xLabelsCellArray);
rotateXLabels(gca, 45);