function visualize_model_comparison(matFile, ~)
%VISUALIZE_MODEL_COMPARISON Create predicted-vs-actual validation plots.
% This shows the top 3 models by validation RMSE, plus Random Forest if it
% is present and not already included.

if nargin < 1 || strlength(string(matFile)) == 0
    matFile = fullfile(fileparts(mfilename('fullpath')), 'matlab_visualization_data.mat');
end

data = load(matFile);

rng(42);

modelNames = normalizeNames(data.model_names);
metricModelNames = normalizeNames(data.metrics_model_names);
numVisualModels = min(3, numel(metricModelNames));
selectedMetricNames = metricModelNames(1:numVisualModels);

randomForestName = "Random Forest";
hasRandomForest = any(modelNames == randomForestName);
alreadyIncluded = any(selectedMetricNames == randomForestName);

if hasRandomForest && ~alreadyIncluded
    selectedMetricNames = [selectedMetricNames; randomForestName];
end

valIdx = sampleIndices(numel(data.y_val), 15000);

fprintf('Loaded %s\n', matFile);
fprintf('Split: %s\n', normalizeScalarString(data.split_label));
fprintf('Showing %d model plots.\n', numel(selectedMetricNames));

for idx = 1:numel(selectedMetricNames)
    modelName = selectedMetricNames(idx);
    predictionIdx = find(modelNames == modelName, 1);
    metricIdx = find(metricModelNames == modelName, 1);

    createPredictedVsActualFigure( ...
        data.y_val(valIdx), ...
        data.val_predictions(valIdx, predictionIdx), ...
        modelName, ...
        data.metrics_rmse(metricIdx), ...
        data.metrics_r2(metricIdx) ...
    );
end
end


function createPredictedVsActualFigure(yTrue, yPred, modelName, rmseValue, r2Value)
figure('Name', sprintf('%s Predicted vs Actual', modelName), 'Color', 'w', 'Position', [120 120 850 700]);

yTrue = yTrue(:);
yPred = yPred(:);
absError = abs(yPred - yTrue);
absError = absError(:);

scatter( ...
    yTrue, ...
    yPred, ...
    18, ...
    absError, ...
    'filled', ...
    'MarkerFaceAlpha', 0.65, ...
    'MarkerEdgeAlpha', 0.10 ...
);
hold on;

minVal = min([yTrue(:); yPred(:)]);
maxVal = max([yTrue(:); yPred(:)]);
plot([minVal maxVal], [minVal maxVal], 'k--', 'LineWidth', 1.5);

hold off;
grid on;
axis tight;
ax = gca;
set(ax, 'Color', 'w');
ax.XColor = 'k';
ax.YColor = 'k';
ax.GridColor = [0 0 0];
ax.GridAlpha = 0.18;
colormap(turbo);
cb = colorbar;
cb.Label.String = 'Absolute Error';
cb.Color = 'k';

title({
    sprintf('%s Predicted vs Actual LN\\_IC50', modelName), ...
    sprintf('Validation RMSE %.3f | R^2 %.3f', rmseValue, r2Value)
});
xlabel('Actual LN\_IC50');
ylabel('Predicted LN\_IC50');
titleHandle = get(gca, 'Title');
titleHandle.Color = 'k';
xlabelHandle = get(gca, 'XLabel');
xlabelHandle.Color = 'k';
ylabelHandle = get(gca, 'YLabel');
ylabelHandle.Color = 'k';
end


function idx = sampleIndices(n, maxPoints)
if n <= maxPoints
    idx = 1:n;
    return;
end

idx = randperm(n, maxPoints);
idx = sort(idx);
end


function names = normalizeNames(raw)
if iscell(raw)
    names = string(raw(:));
elseif ischar(raw)
    names = string(cellstr(raw));
else
    names = string(raw(:));
end

names = strip(names);
end


function value = normalizeScalarString(raw)
value = normalizeNames(raw);
value = value(1);
end
