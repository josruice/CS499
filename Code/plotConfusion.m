function plotConfusion (confusion, figureName, labels)
figure('Name', figureName, 'Position', [200 200 800 700]); 
imagesc(confusion)
set(gca, 'YTick', [1:length(labels)], 'YTickLabel', labels, 'FontSize', 14);
colormap(hot)
colorbar
set(gca, 'XTick', [1:length(labels)], 'XTickLabel', labels, 'FontSize', 14);
rotateXLabelsImagesc(gca, 40);