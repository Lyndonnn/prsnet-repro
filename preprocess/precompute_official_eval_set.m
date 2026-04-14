function [] = precompute_official_eval_set(shapenet_dir, output_dir, benchmark_txt, gt_planes_path, obj_output_dir, max_shapes)
%PRECOMPUTE_OFFICIAL_EVAL_SET Build the PRS-Net 1000-shape benchmark input set.
%
% This creates identity-oriented test .mat files for PRS-Net inference and
% copies the matching ShapeNet OBJ files into evaluation_old/objs.
%
% Unlike precomputeShapeData.m, this script does not apply random rotations.
% The official gt_planes.mat planes are in the OBJ coordinate system, so the
% benchmark inference inputs must stay in that same coordinate system unless
% predictions are explicitly transformed back before evaluation.
%
% Example from repo root on Windows MATLAB:
%   addpath(genpath('D:\tools\gptoolbox'))
%   addpath(fullfile(pwd, 'preprocess'))
%   precompute_official_eval_set( ...
%       'E:\ShapeNetCore.v2\ShapeNetCore.v2', ...
%       fullfile(pwd, 'datasets', 'shapenet_official_eval1000'), ...
%       fullfile(pwd, '1000.txt'), ...
%       fullfile(pwd, 'evaluation_old', 'gt_planes.mat'), ...
%       fullfile(pwd, 'evaluation_old', 'objs'), ...
%       Inf)

    this_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(this_dir);
    addpath(this_dir);

    if nargin < 1 || isempty(shapenet_dir)
        shapenet_dir = fullfile(this_dir, 'shapenet');
    end
    shapenet_dir = normalizeShapeNetRoot(shapenet_dir);

    if nargin < 2 || isempty(output_dir)
        output_dir = fullfile(repo_root, 'datasets', 'shapenet_official_eval1000');
    end
    if nargin < 3 || isempty(benchmark_txt)
        benchmark_txt = fullfile(repo_root, '1000.txt');
    end
    if nargin < 4 || isempty(gt_planes_path)
        gt_planes_path = fullfile(repo_root, 'evaluation_old', 'gt_planes.mat');
    end
    if nargin < 5 || isempty(obj_output_dir)
        obj_output_dir = fullfile(repo_root, 'evaluation_old', 'objs');
    end
    if nargin < 6 || isempty(max_shapes)
        max_shapes = Inf;
    end

    if exist('readOBJ', 'file') ~= 2
        error('readOBJ is not on the MATLAB path. Add gptoolbox with addpath(genpath(...)).');
    end
    if exist('meshlpsampling', 'file') ~= 3 && exist('meshlpsampling', 'file') ~= 2
        error('meshlpsampling is not on the MATLAB path.');
    end
    if exist('point_mesh_squared_distance', 'file') ~= 3 && exist('point_mesh_squared_distance', 'file') ~= 2
        error('point_mesh_squared_distance is not on the MATLAB path.');
    end

    ensureDir(output_dir);
    ensureDir(fullfile(output_dir, 'test'));
    ensureDir(obj_output_dir);

    ids = readBenchmarkIds(benchmark_txt);
    if isfinite(max_shapes)
        ids = ids(1:min(length(ids), max_shapes));
    end

    gt = [];
    if exist(gt_planes_path, 'file')
        data = load(gt_planes_path);
        gt = data.gt_planes;
    else
        warning('gt_planes.mat not found at %s. Duplicate IDs will use first filesystem match.', gt_planes_path);
    end

    gridSize = 32;
    numSamples = 1000;
    stepRange = -0.5 + 1 / (2 * gridSize):1 / gridSize:0.5 - 1 / (2 * gridSize);
    [Xp, Yp, Zp] = ndgrid(stepRange, stepRange, stepRange);
    queryPoints = [Xp(:), Yp(:), Zp(:)];
    axisangle = [1, 0, 0, 0];

    manifest_path = fullfile(output_dir, 'official_eval_manifest.csv');
    mfid = fopen(manifest_path, 'w');
    if mfid < 0
        error('Could not open manifest for writing: %s', manifest_path);
    end
    cleanup = onCleanup(@() fclose(mfid));
    fprintf(mfid, 'shape_index,shape_id,status,category,candidate_count,gt_sde,source_obj,dataset_mat,obj_copy\n');

    fprintf('[precompute_official_eval_set] ShapeNet root: %s\n', shapenet_dir);
    fprintf('[precompute_official_eval_set] Output dataset: %s\n', output_dir);
    fprintf('[precompute_official_eval_set] OBJ output: %s\n', obj_output_dir);
    fprintf('[precompute_official_eval_set] Shapes: %d\n', length(ids));

    rng(1);
    for i = 1:length(ids)
        shape_id = ids{i};
        candidates = findCandidateObjects(shapenet_dir, shape_id);
        if isempty(candidates)
            warning('[precompute_official_eval_set] Missing ShapeNet OBJ for %s', shape_id);
            fprintf(mfid, '%d,%s,missing,,0,NaN,,,\n', i, shape_id);
            continue;
        end

        [selected, score] = chooseCandidate(candidates, gt, i, numSamples);
        [vertices, faces, surfaceSamples, ok] = loadModelData(selected.path, numSamples);
        if ~ok
            warning('[precompute_official_eval_set] Could not load selected OBJ for %s: %s', shape_id, selected.path);
            fprintf(mfid, '%d,%s,load_failed,%s,%d,NaN,%s,,\n', ...
                i, shape_id, selected.category, length(candidates), selected.path);
            continue;
        end

        FV = struct();
        FV.faces = faces;
        FV.vertices = gridSize * (vertices + 0.5) + 0.5;
        Volume = polygon2voxel(FV, gridSize, 'none', false);

        [~, ~, closestPoints] = point_mesh_squared_distance(queryPoints, vertices, faces);
        closestPoints = reshape(closestPoints, [size(Xp), 3]);

        dataset_mat = fullfile(output_dir, 'test', [shape_id, '.mat']);
        save(dataset_mat, 'Volume', 'surfaceSamples', 'vertices', 'faces', 'axisangle', 'closestPoints');

        obj_copy = fullfile(obj_output_dir, [shape_id, '.obj']);
        copyfile(selected.path, obj_copy, 'f');

        fprintf(mfid, '%d,%s,ok,%s,%d,%.12g,%s,%s,%s\n', ...
            i, shape_id, selected.category, length(candidates), score, selected.path, dataset_mat, obj_copy);

        if mod(i, 25) == 0 || i == length(ids)
            fprintf('[precompute_official_eval_set] processed %d / %d\n', i, length(ids));
        end
    end

    fprintf('[precompute_official_eval_set] wrote %s\n', manifest_path);
end

function root = normalizeShapeNetRoot(root)
    if countCategoryDirs(root) > 0
        return;
    end
    nested = fullfile(root, 'ShapeNetCore.v2');
    if exist(nested, 'dir') && countCategoryDirs(nested) > 0
        root = nested;
    end
end

function count = countCategoryDirs(root)
    count = 0;
    if ~exist(root, 'dir')
        return;
    end
    entries = dir(root);
    for i = 1:length(entries)
        name = entries(i).name;
        if entries(i).isdir && ~isempty(regexp(name, '^\d{8}$', 'once'))
            count = count + 1;
        end
    end
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

function candidates = findCandidateObjects(shapenet_dir, shape_id)
    entries = dir(shapenet_dir);
    candidates = struct('category', {}, 'path', {});
    for i = 1:length(entries)
        category = entries(i).name;
        if ~entries(i).isdir || isempty(regexp(category, '^\d{8}$', 'once'))
            continue;
        end
        obj_path = fullfile(shapenet_dir, category, shape_id, 'models', 'model_normalized.obj');
        if exist(obj_path, 'file')
            candidates(end + 1).category = category; %#ok<AGROW>
            candidates(end).path = obj_path;
        end
    end
end

function [selected, score] = chooseCandidate(candidates, gt, shape_index, numSamples)
    selected = candidates(1);
    score = NaN;
    if length(candidates) == 1 || isempty(gt)
        return;
    end

    best_score = Inf;
    best_index = 1;
    for i = 1:length(candidates)
        candidate_score = scoreCandidate(candidates(i).path, gt(shape_index, :), numSamples);
        if candidate_score < best_score
            best_score = candidate_score;
            best_index = i;
        end
    end
    selected = candidates(best_index);
    score = best_score;
end

function score = scoreCandidate(obj_path, gt_row, numSamples)
    score = Inf;
    [vertices, faces, surfaceSamples, ok] = loadModelData(obj_path, numSamples);
    if ~ok
        return;
    end
    points = surfaceSamples';
    values = [];
    for j = 1:size(gt_row, 2)
        plane = normalizePlane(gt_row{j});
        if isempty(plane)
            continue;
        end
        values(end + 1) = planeSde(plane, vertices, faces, points); %#ok<AGROW>
    end
    if ~isempty(values)
        score = mean(values);
    end
end

function [vertices, faces, surfaceSamples, ok] = loadModelData(obj_path, numSamples)
    ok = false;
    vertices = [];
    faces = [];
    surfaceSamples = [];
    try
        [vertices, faces, ~] = readOBJ(obj_path);
        [surfaceSamples, ~] = meshlpsampling(obj_path, numSamples);
    catch ME
        warning('[precompute_official_eval_set] Could not load %s: %s', obj_path, ME.message);
        return;
    end
    vertices = double(vertices);
    faces = double(faces);
    if min(faces(:)) == 0
        faces = faces + 1;
    end
    if isempty(vertices) || isempty(faces) || isempty(surfaceSamples)
        return;
    end
    ok = true;
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

function sde = planeSde(plane, vertices, faces, points)
    hpoints = [points, ones(size(points, 1), 1)];
    lambda = hpoints * plane';
    reflected = hpoints(:, 1:3) - 2 .* lambda .* plane(1:3);
    distances = point_mesh_squared_distance(reflected, vertices, double(faces));
    sde = sum(distances) / size(points, 1);
end

function ensureDir(path)
    if ~exist(path, 'dir')
        mkdir(path);
    end
end
