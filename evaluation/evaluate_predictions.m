function [] = evaluate_predictions(test_dir, prediction_dir, output_csv, max_files, include_pca)
%EVALUATE_PREDICTIONS Exact point-to-mesh SDE for PRS-Net predictions.
%
% This evaluates prediction .mat files written by test.py. It uses the same
% point_mesh_squared_distance MEX dependency used by the official MATLAB
% preprocessing/evaluation code.
%
% Example:
%   evaluate_predictions( ...
%       'D:\code\prsnet-repro\datasets\shapenet\test', ...
%       'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest', ...
%       'D:\code\prsnet-repro\results\airplane_1000train_201test_bs32\test_latest\exact_sde_metrics.csv', ...
%       Inf, true)

    if nargin < 4 || isempty(max_files)
        max_files = Inf;
    end
    if nargin < 5 || isempty(include_pca)
        include_pca = true;
    end

    this_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(this_dir);
    addpath(this_dir);
    addpath(fullfile(repo_root, 'preprocess'));

    if exist('point_mesh_squared_distance', 'file') ~= 3 && exist('point_mesh_squared_distance', 'file') ~= 2
        error(['point_mesh_squared_distance is not on the MATLAB path. ', ...
               'Run from the repo, or add preprocess/ or the unzipped evaluation.zip directory to the path.']);
    end

    prediction_files = dir(fullfile(prediction_dir, '*.mat'));
    [~, order] = sort({prediction_files.name});
    prediction_files = prediction_files(order);
    if isfinite(max_files)
        prediction_files = prediction_files(1:min(length(prediction_files), max_files));
    end
    if isempty(prediction_files)
        error('No prediction .mat files found in %s', prediction_dir);
    end

    out_fid = fopen(output_csv, 'w');
    if out_fid < 0
        error('Could not open output CSV: %s', output_csv);
    end
    cleanup = onCleanup(@() fclose(out_fid));
    fprintf(out_fid, 'shape_id,method,plane_id,sde_exact,a,b,c,d,source_mat,prediction_mat\n');

    fprintf('[evaluate_predictions] test_dir=%s\n', test_dir);
    fprintf('[evaluate_predictions] prediction_dir=%s\n', prediction_dir);
    fprintf('[evaluate_predictions] files=%d\n', length(prediction_files));

    for i = 1:length(prediction_files)
        pred_path = fullfile(prediction_files(i).folder, prediction_files(i).name);
        [~, shape_id, ~] = fileparts(prediction_files(i).name);
        source_path = fullfile(test_dir, prediction_files(i).name);
        if ~exist(source_path, 'file')
            source_path = pred_path;
            fprintf('[evaluate_predictions] source missing, using prediction mesh: %s\n', pred_path);
        end

        source = load(source_path);
        pred = load(pred_path);
        [vertices, faces, points] = load_shape_data(source, pred);

        plane_names = fieldnames(pred);
        plane_names = plane_names(startsWith(plane_names, 'plane'));
        plane_names = sort_plane_names(plane_names);
        for j = 1:length(plane_names)
            plane = normalize_plane(pred.(plane_names{j}));
            sde = plane_sde(plane, vertices, faces, points);
            write_row(out_fid, shape_id, 'prsnet', plane_names{j}, sde, plane, source_path, pred_path);
        end

        if include_pca
            pca_planes = pca_baseline_planes(points);
            for j = 1:size(pca_planes, 1)
                plane = pca_planes(j, :);
                sde = plane_sde(plane, vertices, faces, points);
                write_row(out_fid, shape_id, 'pca', sprintf('pca%d', j - 1), sde, plane, source_path, pred_path);
            end
        end

        if mod(i, 25) == 0 || i == length(prediction_files)
            fprintf('[evaluate_predictions] processed %d / %d\n', i, length(prediction_files));
        end
    end

    fprintf('[evaluate_predictions] wrote %s\n', output_csv);
end

function [vertices, faces, points] = load_shape_data(source, pred)
    if isfield(source, 'vertices')
        vertices = double(source.vertices);
    else
        vertices = double(pred.vertices);
    end
    if isfield(source, 'faces')
        faces = double(source.faces);
    else
        faces = double(pred.faces);
    end
    if min(faces(:)) == 0
        faces = faces + 1;
    end

    if isfield(source, 'surfaceSamples')
        points = double(source.surfaceSamples);
    elseif isfield(pred, 'sample')
        points = double(pred.sample);
    elseif isfield(pred, 'surfaceSamples')
        points = double(pred.surfaceSamples);
    else
        error('Missing surfaceSamples/sample in source and prediction mat.');
    end
    if size(points, 1) == 3
        points = points';
    end
    if size(points, 2) ~= 3
        error('surface samples should be N x 3 or 3 x N, got %s', mat2str(size(points)));
    end
end

function names = sort_plane_names(names)
    ids = zeros(length(names), 1);
    for i = 1:length(names)
        ids(i) = sscanf(names{i}, 'plane%d');
    end
    [~, order] = sort(ids);
    names = names(order);
end

function plane = normalize_plane(plane)
    plane = double(plane(:))';
    if length(plane) ~= 4
        error('Plane should have 4 coefficients, got %d', length(plane));
    end
    n = norm(plane(1:3));
    if n < 1e-12
        error('Plane normal is near zero');
    end
    plane = plane / n;
end

function sde = plane_sde(plane, vertices, faces, points)
    if size(points, 2) ~= 4
        hpoints = [points, ones(size(points, 1), 1)];
    else
        hpoints = points;
    end
    lambda = hpoints * plane';
    reflected = hpoints(:, 1:3) - 2 .* lambda .* plane(1:3);
    distances = point_mesh_squared_distance(reflected(:, 1:3), vertices, double(faces));
    sde = sum(distances) / size(points, 1);
end

function planes = pca_baseline_planes(points)
    center = mean(points, 1);
    centered = points - center;
    [~, ~, v] = svd(centered, 'econ');
    planes = zeros(3, 4);
    for i = 1:3
        normal = v(:, i)';
        normal = normal / norm(normal);
        d = -dot(normal, center);
        planes(i, :) = [normal, d];
    end
end

function write_row(fid, shape_id, method, plane_id, sde, plane, source_path, pred_path)
    fprintf(fid, '%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s\n', ...
        shape_id, method, plane_id, sde, plane(1), plane(2), plane(3), plane(4), source_path, pred_path);
end
