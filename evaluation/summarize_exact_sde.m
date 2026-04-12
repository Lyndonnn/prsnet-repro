function [] = summarize_exact_sde(metrics_csv, summary_csv)
%SUMMARIZE_EXACT_SDE Summarize best-per-shape SDE from evaluate_predictions.
%
% Example:
%   summarize_exact_sde( ...
%       'D:\code\prsnet-repro\results\exp\test_latest\exact_sde_metrics.csv', ...
%       'D:\code\prsnet-repro\results\exp\test_latest\exact_sde_summary.csv')

    if nargin < 2 || isempty(summary_csv)
        [folder, name, ~] = fileparts(metrics_csv);
        summary_csv = fullfile(folder, [name, '_summary.csv']);
    end

    T = readtable(metrics_csv, 'TextType', 'string');
    methods = unique(T.method);
    shape_ids = unique(T.shape_id);

    out_method = strings(0, 1);
    out_num_shapes = zeros(0, 1);
    out_mean = zeros(0, 1);
    out_median = zeros(0, 1);
    out_min = zeros(0, 1);
    out_max = zeros(0, 1);

    best_by_method = containers.Map();
    for m = 1:length(methods)
        method = methods(m);
        best_values = [];
        for s = 1:length(shape_ids)
            rows = T(T.method == method & T.shape_id == shape_ids(s), :);
            if isempty(rows)
                continue;
            end
            best_values(end + 1) = min(rows.sde_exact); %#ok<AGROW>
        end
        best_by_method(char(method)) = best_values;
        out_method(end + 1, 1) = method; %#ok<AGROW>
        out_num_shapes(end + 1, 1) = length(best_values); %#ok<AGROW>
        out_mean(end + 1, 1) = mean(best_values); %#ok<AGROW>
        out_median(end + 1, 1) = median(best_values); %#ok<AGROW>
        out_min(end + 1, 1) = min(best_values); %#ok<AGROW>
        out_max(end + 1, 1) = max(best_values); %#ok<AGROW>
    end

    S = table(out_method, out_num_shapes, out_mean, out_median, out_min, out_max, ...
        'VariableNames', {'method', 'num_shapes', 'mean_best_sde_exact', ...
                          'median_best_sde_exact', 'min_best_sde_exact', 'max_best_sde_exact'});
    writetable(S, summary_csv);
    disp(S);

    if isKey(best_by_method, 'prsnet') && isKey(best_by_method, 'pca')
        prs_wins = 0;
        pca_wins = 0;
        comparable = 0;
        for s = 1:length(shape_ids)
            prs_rows = T(T.method == "prsnet" & T.shape_id == shape_ids(s), :);
            pca_rows = T(T.method == "pca" & T.shape_id == shape_ids(s), :);
            if isempty(prs_rows) || isempty(pca_rows)
                continue;
            end
            comparable = comparable + 1;
            if min(prs_rows.sde_exact) < min(pca_rows.sde_exact)
                prs_wins = prs_wins + 1;
            else
                pca_wins = pca_wins + 1;
            end
        end
        fprintf('PRS-Net wins: %d\n', prs_wins);
        fprintf('PCA wins: %d\n', pca_wins);
        fprintf('Comparable shapes: %d\n', comparable);
    end

    fprintf('[summarize_exact_sde] wrote %s\n', summary_csv);
end
