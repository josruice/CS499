% TODO: Write proper documentation.

function plot_confusion (confusion, figure_name, labels)
    figure('Name', figure_name, 'Position', [200 200 600 525]); 
    imagesc(confusion)
    set(gca, 'YTick', [1:length(labels)], 'YTickLabel', labels);
    colormap(hot)
    colorbar
    set(gca, 'XTick', [1:length(labels)], 'XTickLabel', labels);
    rotateXLabelsImagesc(gca, 40);
end