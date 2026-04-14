function [] = evaluate_official_benchmark(benchmark_txt, gt_planes_path, prediction_dir, obj_dir, output_csv, summary_csv, sde_threshold, duplicate_angle_threshold, include_pca)
%EVALUATE_OFFICIAL_BENCHMARK Evaluate PRS-Net predictions on the official 1000 IDs.
%
% This script uses evaluation_old/gt_planes.mat and evaluation_old/objs.
% It reports:
%   - gt: SDE of the provided ground-truth planes, a sanity lower bound
%   - prsnet_raw: all plane* outputs from test.py
%   - prsnet_filtered: PRS-Net outputs after duplicate and SDE-threshold filtering
%   - pca: three PCA baseline planes from the sampled mesh points
%
% Example from repo root on Windows MATLAB:
%   addpath(genpath('D:\tools\gptoolbox'))
%   addpath(fullfile(pwd, 'evaluation_old'))
%   addpath(fullfile(pwd, 'evaluation'))
%   evaluate_official_benchmark( ...
%       fullfile(pwd, '1000.txt'), ...
%       fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
%       fullfile(pwd, 'results', 'my_exp', 'test_200'), ...
%       fullfile(pwd, 'evaluation_old', 'objs'), ...
%       fullfile(pwd, 'results', 'my_exp', 'test_200', 'official_metrics.csv'), ...
%       fullfile(pwd, 'results', 'my_exp', 'test_200', 'official_summary.csv'), ...
%       0.0004, ...
%       pi / 6, ...
%       true)

    this_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(this_dir);
    addpath(this_dir);
    addpath(fullfile(repo_root, 'evaluation_old'));
    addpath(fullfile(repo_root, 'preprocess'));

    if nargin < 1 || isempty(benchmark_txt)
        benchmark_txt = fullfile(repo_root, '1000.txt');
    end
    if nargin < 2 || isempty(gt_planes_path)
        gt_planes_path = fullfile(repo_root, 'evaluation_old', 'gt_planes.mat');
    end
    if nargin < 3 || isempty(prediction_dir)
        error('prediction_dir is required. Run test.py on datasets/shapenet_official_eval1000 first.');
    end
    if nargin < 4 || isempty(obj_dir)
        obj_dir = fullfile(repo_root, 'evaluation_old', 'objs');
    end
    if nargin < 5 || isempty(output_csv)
        output_csv = fullfile(prediction_dir, 'official_metrics.csv');
    end
    if nargin < 6 || isempty(summary_csv)
        summary_csv = fullfile(prediction_dir, 'official_summary.csv');
    end
    if nargin < 7 || isempty(sde_threshold)
        sde_threshold = 0.0004;
    end
    if nargin < 8 || isempty(duplicate_angle_threshold)
        duplicate_angle_threshold = pi / 6;
    end
    if nargin < 9 || isempty(include_pca)
        include_pca = true;
    end

    if exist('point_mesh_squared_distance', 'file') ~= 3 && exist('point_mesh_squared_distance', 'file') ~= 2
        error('point_mesh_squared_distance is not on the MATLAB path.');
    end
    if exist('uniform_sampling', 'file') ~= 2
        error('uniform_sampling.m is not on the MATLAB path. Add evaluation_old/.');
    end

    ids = readBenchmarkIds(benchmark_txt);
    data = load(gt_planes_path);
    gt_cells = data.gt_planes;
    if size(gt_cells, 1) < length(ids)
        error('gt_planes has %d rows but benchmark has %d IDs.', size(gt_cells, 1), length(ids));
    end

    output_parent = fileparts(output_csv);
    if ~isempty(output_parent) && ~exist(output_parent, 'dir')
        mkdir(output_parent);
    end
    summary_parent = fileparts(summary_csv);
    if ~isempty(summary_parent) && ~exist(summary_parent, 'dir')
        mkdir(summary_parent);
    end

    metrics_fid = fopen(output_csv, 'w');
    if metrics_fid < 0
        error('Could not open metrics CSV: %s', output_csv);
    end
    cleanup = onCleanup(@() fclose(metrics_fid));
    fprintf(metrics_fid, 'shape_index,shape_id,method,plane_id,kept,sde_exact,best_gt_plane_dist,matched_at_0p2,a,b,c,d,prediction_mat,obj_path\n');

    methods = {'gt', 'prsnet_raw', 'prsnet_filtered', 'pca'};
    method_planes = struct();
    method_sdes = struct();
    for m = 1:length(methods)
        method_planes.(methods{m}) = cell(length(ids), 1);
        method_sdes.(methods{m}) = cell(length(ids), 1);
    end
    gt_by_shape = cell(length(ids), 1);

    rng(1);
    fprintf('[evaluate_official_benchmark] benchmark=%s\n', benchmark_txt);
    fprintf('[evaluate_official_benchmark] predictions=%s\n', prediction_dir);
    fprintf('[evaluate_official_benchmark] obj_dir=%s\n', obj_dir);

    for i = 1:length(ids)
        shape_id = ids{i};
        obj_path = fullfile(obj_dir, [shape_id, '.obj']);
        if ~exist(obj_path, 'file')
            warning('[evaluate_official_benchmark] Missing OBJ: %s', obj_path);
            continue;
        end

        [vertices, faces] = loadObjMesh(obj_path);
        sample = uniform_sampling(vertices, faces, 1000)';
        gt_planes = loadGtPlanes(gt_cells(i, :));
        gt_by_shape{i} = gt_planes;

        gt_sdes = evaluatePlaneSet(gt_planes, vertices, faces, sample);
        method_planes.gt{i} = gt_planes;
        method_sdes.gt{i} = gt_sdes;
        writePlaneRows(metrics_fid, i, shape_id, 'gt', gt_planes, gt_sdes, true(size(gt_sdes)), gt_planes, '', obj_path);

        pred_path = fullfile(prediction_dir, [shape_id, '.mat']);
        if exist(pred_path, 'file')
            [raw_planes, raw_names] = loadPredictionPlanes(pred_path);
            raw_sdes = evaluatePlaneSet(raw_planes, vertices, faces, sample);
            method_planes.prsnet_raw{i} = raw_planes;
            method_sdes.prsnet_raw{i} = raw_sdes;
            writePlaneRows(metrics_fid, i, shape_id, 'prsnet_raw', raw_planes, raw_sdes, true(size(raw_sdes)), gt_planes, pred_path, obj_path, raw_names);

            keep = filterPredictionPlanes(raw_planes, raw_sdes, sde_threshold, duplicate_angle_threshold);
            filtered_planes = raw_planes(keep, :);
            filtered_sdes = raw_sdes(keep);
            filtered_names = raw_names(keep);
            method_planes.prsnet_filtered{i} = filtered_planes;
            method_sdes.prsnet_filtered{i} = filtered_sdes;
            writePlaneRows(metrics_fid, i, shape_id, 'prsnet_filtered', filtered_planes, filtered_sdes, true(size(filtered_sdes)), gt_planes, pred_path, obj_path, filtered_names);
        else
            warning('[evaluate_official_benchmark] Missing prediction: %s', pred_path);
        end

        if include_pca
            pca_planes = pcaBaselinePlanes(sample);
            pca_sdes = evaluatePlaneSet(pca_planes, vertices, faces, sample);
            method_planes.pca{i} = pca_planes;
            method_sdes.pca{i} = pca_sdes;
            writePlaneRows(metrics_fid, i, shape_id, 'pca', pca_planes, pca_sdes, true(size(pca_sdes)), gt_planes, '', obj_path);
        end

        if mod(i, 25) == 0 || i == length(ids)
            fprintf('[evaluate_official_benchmark] processed %d / %d\n', i, length(ids));
        end
    end

    writeSummary(summary_csv, methods, method_planes, method_sdes, gt_by_shape, length(ids));
    fprintf('[evaluate_official_benchmark] wrote %s\n', output_csv);
    fprintf('[evaluate_official_benchmark] wrote %s\n', summary_csv);
end

function ids = readBenchmarkIds(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Could not open benchmark txt: %s', path);
    end
    data = textscan(fid, '%s %s');
    fclose(fid);
    ids = data{1};
end

function [vertices, faces] = loadObjMesh(obj_path)
    if exist('readobjfromfile', 'file') == 3 || exist('readobjfromfile', 'file') == 2
        [vertices, faces] = readobjfromfile(obj_path);
        faces = double(faces) + 1;
    elseif exist('readOBJ', 'file') == 2
        [vertices, faces, ~] = readOBJ(obj_path);
        faces = double(faces);
        if min(faces(:)) == 0
            faces = faces + 1;
        end
    else
        error('Neither readobjfromfile nor readOBJ is on the MATLAB path.');
    end
    vertices = double(vertices);
end

function planes = loadGtPlanes(gt_row)
    planes = zeros(0, 4);
    for j = 1:size(gt_row, 2)
        plane = normalizePlane(gt_row{j});
        if ~isempty(plane)
            planes(end + 1, :) = plane; %#ok<AGROW>
        end
    end
end

function [planes, names] = loadPredictionPlanes(pred_path)
    pred = load(pred_path);
    fields = fieldnames(pred);
    names = {};
    planes = zeros(0, 4);
    for i = 1:length(fields)
        name = fields{i};
        if length(name) >= 6 && strcmp(name(1:5), 'plane')
            plane = normalizePlane(pred.(name));
            if ~isempty(plane)
                names{end + 1, 1} = name; %#ok<AGROW>
                planes(end + 1, :) = plane; %#ok<AGROW>
            end
        end
    end
    ids = zeros(length(names), 1);
    for i = 1:length(names)
        ids(i) = sscanf(names{i}, 'plane%d');
    end
    [~, order] = sort(ids);
    names = names(order);
    planes = planes(order, :);
end

function plane = normalizePlane(raw)
    plane = [];
    if isempty(raw)
        return;
    end
    raw = double(raw(:))';
    if length(raw) ~= 4 || ~all(isfinite(raw))
        return;
    end
    n = norm(raw(1:3));
    if n < 1e-12
        return;
    end
    plane = raw / n;
end

function sdes = evaluatePlaneSet(planes, vertices, faces, sample)
    sdes = zeros(size(planes, 1), 1);
    for i = 1:size(planes, 1)
        sdes(i) = planeSde(planes(i, :), vertices, faces, sample);
    end
end

function sde = planeSde(plane, vertices, faces, sample)
    hpoints = [sample, ones(size(sample, 1), 1)];
    lambda = hpoints * plane';
    reflected = hpoints(:, 1:3) - 2 .* lambda .* plane(1:3);
    distances = point_mesh_squared_distance(reflected, vertices, double(faces));
    sde = sum(distances) / size(sample, 1);
end

function planes = pcaBaselinePlanes(sample)
    center = mean(sample, 1);
    centered = sample - center;
    [~, ~, v] = svd(centered, 'econ');
    planes = zeros(3, 4);
    for i = 1:3
        normal = v(:, i)';
        normal = normal / norm(normal);
        d = -dot(normal, center);
        planes(i, :) = [normal, d];
    end
end

function keep = filterPredictionPlanes(planes, sdes, sde_threshold, duplicate_angle_threshold)
    keep = false(size(sdes));
    if isempty(sdes)
        return;
    end
    [~, order] = sort(sdes);
    for oi = 1:length(order)
        idx = order(oi);
        if sdes(idx) > sde_threshold
            continue;
        end
        duplicate = false;
        kept_ids = find(keep);
        for kj = 1:length(kept_ids)
            kidx = kept_ids(kj);
            cos_angle = abs(dot(planes(idx, 1:3), planes(kidx, 1:3)));
            cos_angle = min(max(cos_angle, -1), 1);
            angle = acos(cos_angle);
            if angle < duplicate_angle_threshold
                duplicate = true;
                break;
            end
        end
        if ~duplicate
            keep(idx) = true;
        end
    end
end

function writePlaneRows(fid, shape_index, shape_id, method, planes, sdes, kept, gt_planes, pred_path, obj_path, names)
    if nargin < 11 || isempty(names)
        names = cell(size(planes, 1), 1);
        for i = 1:size(planes, 1)
            names{i} = sprintf('%s%d', method, i - 1);
        end
    end
    for i = 1:size(planes, 1)
        dist = bestPlaneDistance(planes(i, :), gt_planes);
        matched = dist <= 0.2;
        fprintf(fid, '%d,%s,%s,%s,%d,%.12g,%.12g,%d,%.12g,%.12g,%.12g,%.12g,%s,%s\n', ...
            shape_index, shape_id, method, names{i}, kept(i), sdes(i), dist, matched, ...
            planes(i, 1), planes(i, 2), planes(i, 3), planes(i, 4), pred_path, obj_path);
    end
end

function dist = bestPlaneDistance(plane, gt_planes)
    if isempty(gt_planes)
        dist = Inf;
        return;
    end
    dists = zeros(size(gt_planes, 1), 1);
    for i = 1:size(gt_planes, 1)
        dists(i) = planeDistance(plane, gt_planes(i, :));
    end
    dist = min(dists);
end

function dist = planeDistance(a, b)
    dist = min(norm(a - b), norm(a + b));
end

function writeSummary(summary_csv, methods, method_planes, method_sdes, gt_by_shape, total_shapes)
    fid = fopen(summary_csv, 'w');
    if fid < 0
        error('Could not open summary CSV: %s', summary_csv);
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'method,total_shapes,shapes_with_predictions,total_predictions,mean_best_sde,median_best_sde,min_best_sde,max_best_sde,fscore_avg_0_0p2,fscore_at_0p05,fscore_at_0p1,fscore_at_0p2\n');

    thresholds = linspace(0, 0.2, 201);
    for m = 1:length(methods)
        method = methods{m};
        best_values = [];
        total_predictions = 0;
        shapes_with_predictions = 0;
        for i = 1:total_shapes
            sdes = method_sdes.(method){i};
            planes = method_planes.(method){i};
            if ~isempty(planes)
                total_predictions = total_predictions + size(planes, 1);
                shapes_with_predictions = shapes_with_predictions + 1;
            end
            if ~isempty(sdes)
                best_values(end + 1) = min(sdes); %#ok<AGROW>
            end
        end

        if isempty(best_values)
            mean_best = NaN;
            median_best = NaN;
            min_best = NaN;
            max_best = NaN;
        else
            mean_best = mean(best_values);
            median_best = median(best_values);
            min_best = min(best_values);
            max_best = max(best_values);
        end

        f_avg = averageFscore(method_planes.(method), gt_by_shape, thresholds);
        f_005 = datasetFscore(method_planes.(method), gt_by_shape, 0.05);
        f_01 = datasetFscore(method_planes.(method), gt_by_shape, 0.1);
        f_02 = datasetFscore(method_planes.(method), gt_by_shape, 0.2);

        fprintf(fid, '%s,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
            method, total_shapes, shapes_with_predictions, total_predictions, ...
            mean_best, median_best, min_best, max_best, f_avg, f_005, f_01, f_02);
    end
end

function f_avg = averageFscore(pred_by_shape, gt_by_shape, thresholds)
    values = zeros(length(thresholds), 1);
    for i = 1:length(thresholds)
        values(i) = datasetFscore(pred_by_shape, gt_by_shape, thresholds(i));
    end
    f_avg = mean(values);
end

function f = datasetFscore(pred_by_shape, gt_by_shape, threshold)
    tp = 0;
    fp = 0;
    fn = 0;
    for i = 1:length(gt_by_shape)
        pred = pred_by_shape{i};
        gt = gt_by_shape{i};
        [shape_tp, shape_fp, shape_fn] = matchCounts(pred, gt, threshold);
        tp = tp + shape_tp;
        fp = fp + shape_fp;
        fn = fn + shape_fn;
    end
    precision = tp / max(tp + fp, 1);
    recall = tp / max(tp + fn, 1);
    if precision + recall == 0
        f = 0;
    else
        f = 2 * precision * recall / (precision + recall);
    end
end

function [tp, fp, fn] = matchCounts(pred, gt, threshold)
    if isempty(pred)
        tp = 0;
        fp = 0;
        fn = size(gt, 1);
        return;
    end
    if isempty(gt)
        tp = 0;
        fp = size(pred, 1);
        fn = 0;
        return;
    end

    pairs = zeros(0, 3);
    for i = 1:size(pred, 1)
        for j = 1:size(gt, 1)
            dist = planeDistance(pred(i, :), gt(j, :));
            if dist <= threshold
                pairs(end + 1, :) = [dist, i, j]; %#ok<AGROW>
            end
        end
    end
    if isempty(pairs)
        tp = 0;
        fp = size(pred, 1);
        fn = size(gt, 1);
        return;
    end

    [~, order] = sort(pairs(:, 1));
    pairs = pairs(order, :);
    used_pred = false(size(pred, 1), 1);
    used_gt = false(size(gt, 1), 1);
    tp = 0;
    for k = 1:size(pairs, 1)
        pi = pairs(k, 2);
        gi = pairs(k, 3);
        if ~used_pred(pi) && ~used_gt(gi)
            used_pred(pi) = true;
            used_gt(gi) = true;
            tp = tp + 1;
        end
    end
    fp = size(pred, 1) - tp;
    fn = size(gt, 1) - tp;
end
