function plot_confusion (confusion, figureName, labels)
figure('Name', figureName, 'Position', [200 200 600 525]); 
imagesc(confusion)
set(gca, 'YTick', [1:length(labels)], 'YTickLabel', labels);
colormap(hot)
colorbar
set(gca, 'XTick', [1:length(labels)], 'XTickLabel', labels);
rotateXLabelsImagesc(gca, 40);