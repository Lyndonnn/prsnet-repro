function [] = precomputeShapeData(shapenet_dir, savepath, split_dir, categories, train_limit, test_limit, num_aug_per_model)
    % Official defaults are preserved for a full preprocessing run from this
    % directory. Extra arguments make tiny smoke subsets possible.
    if nargin < 1 || isempty(shapenet_dir)
        shapenet_dir = './shapenet/';
    end
    if nargin < 2 || isempty(savepath)
        savepath = '../datasets/shapenet/';
    end
    if nargin < 3 || isempty(split_dir)
        split_dir = './data_split/';
    end
    if nargin < 4 || isempty(categories)
        cates = dir(shapenet_dir);
        category_names = {};
        for i = 1:length(cates)
            if cates(i).isdir && ~strcmp(cates(i).name, '.') && ~strcmp(cates(i).name, '..')
                category_names{end + 1} = cates(i).name; %#ok<AGROW>
            end
        end
    else
        category_names = normalizeCategories(categories);
    end
    if nargin < 5 || isempty(train_limit)
        train_limit = Inf;
    end
    if nargin < 6 || isempty(test_limit)
        test_limit = Inf;
    end
    if nargin < 7 || isempty(num_aug_per_model)
        num_aug_per_model = 0;
    end

    ensureDir(savepath);
    ensureDir(fullfile(savepath, 'train'));
    ensureDir(fullfile(savepath, 'test'));

    gridSize = 32;
    numSamples = 1000;
    targetnum = 4000;
    stepRange = -0.5 + 1 / (2 * gridSize):1 / gridSize:0.5 - 1 / (2 * gridSize);
    [Xp, Yp, Zp] = ndgrid(stepRange, stepRange, stepRange);
    queryPoints = [Xp(:), Yp(:), Zp(:)];

    fprintf('[precomputeShapeData] ShapeNet root: %s\n', shapenet_dir);
    fprintf('[precomputeShapeData] Output root: %s\n', savepath);
    fprintf('[precomputeShapeData] Split dir: %s\n', split_dir);
    fprintf('[precomputeShapeData] Categories: %s\n', strjoin(category_names, ', '));

    for i = 1:length(category_names)
        cate = category_names{i};
        fprintf('[precomputeShapeData] Category %s\n', cate);

        train_split = fullfile(split_dir, [cate, '_train.txt']);
        if exist(train_split, 'file')
            train_ids = readIdList(train_split);
            train_ids = limitIds(train_ids, train_limit);
            if isempty(train_ids)
                warning('[precomputeShapeData] No train ids selected for category %s', cate);
            else
                if num_aug_per_model > 0
                    num_per_model = num_aug_per_model;
                else
                    num_per_model = ceil(targetnum / length(train_ids));
                end
                fprintf('[precomputeShapeData] Train ids: %d, augmentations per id: %d\n', length(train_ids), num_per_model);
                for j = 1:size(train_ids, 1)
                    modelfile = fullfile(shapenet_dir, cate, train_ids{j}, 'models', 'model_normalized.obj');
                    [vertices, faces, surfaceSamples, ok] = loadModelData(modelfile, numSamples);
                    if ~ok
                        continue;
                    end
                    for k = 1:num_per_model
                        out_file = fullfile(savepath, 'train', [train_ids{j}, '_a', num2str(k), '.mat']);
                        preprocessLoadedModel(vertices, faces, surfaceSamples, modelfile, out_file, gridSize, queryPoints, Xp, Yp, Zp);
                    end
                end
            end
        else
            warning('[precomputeShapeData] Missing train split: %s', train_split);
        end

        test_split = fullfile(split_dir, [cate, '_test.txt']);
        if exist(test_split, 'file')
            test_ids = readIdList(test_split);
            test_ids = limitIds(test_ids, test_limit);
            fprintf('[precomputeShapeData] Test ids: %d\n', length(test_ids));
            for j = 1:size(test_ids, 1)
                modelfile = fullfile(shapenet_dir, cate, test_ids{j}, 'models', 'model_normalized.obj');
                [vertices, faces, surfaceSamples, ok] = loadModelData(modelfile, numSamples);
                if ~ok
                    continue;
                end
                out_file = fullfile(savepath, 'test', [test_ids{j}, '.mat']);
                preprocessLoadedModel(vertices, faces, surfaceSamples, modelfile, out_file, gridSize, queryPoints, Xp, Yp, Zp);
            end
        else
            warning('[precomputeShapeData] Missing test split: %s', test_split);
        end
    end
end

function [vertices, faces, surfaceSamples, ok] = loadModelData(modelfile, numSamples)
    ok = false;
    vertices = [];
    faces = [];
    surfaceSamples = [];
    try
        [vertices, faces, ~] = readOBJ(modelfile);
        [surfaceSamples, ~] = meshlpsampling(modelfile, numSamples);
    catch ME
        warning('[precomputeShapeData] Skipping unreadable mesh %s: %s', modelfile, ME.message);
        return;
    end
    if isempty(vertices) || isempty(faces)
        warning('[precomputeShapeData] Skipping empty mesh %s', modelfile);
        return;
    end
    ok = true;
end

function preprocessLoadedModel(vertices, faces, surfaceSamples, modelfile, out_file, gridSize, queryPoints, Xp, Yp, Zp)
    fprintf('[precomputeShapeData]   %s -> %s\n', modelfile, out_file);
    axis = rand(1, 3);
    axis = axis / norm(axis);
    angle = rand(1) * 2 * pi;
    axisangle = [axis, angle];
    R = axisAngleToRotm(axisangle);
    v = R * vertices';

    FV = struct();
    FV.faces = faces;
    FV.vertices = (gridSize) * (v' + 0.5) + 0.5;
    Volume = polygon2voxel(FV, gridSize, 'none', false);
    surfaceSamples = R * surfaceSamples;
    [~, ~, closestPoints] = point_mesh_squared_distance(queryPoints, v', faces);
    closestPointsGrid = reshape(closestPoints, [size(Xp), 3]);
    savefunc(out_file, Volume, surfaceSamples, v', faces, axisangle, closestPointsGrid);
end

function categories = normalizeCategories(categories)
    if isa(categories, 'string')
        categories = cellstr(categories);
    end
    if ischar(categories)
        categories = strsplit(categories, ',');
    end
    for i = 1:length(categories)
        categories{i} = strtrim(categories{i});
    end
end

function ids = readIdList(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('[precomputeShapeData] Could not open split file: %s', path);
    end
    ids = textscan(fid, '%s', 'delimiter', '\n');
    ids = ids{1};
    fclose(fid);
end

function ids = limitIds(ids, limit)
    if isfinite(limit)
        ids = ids(1:min(length(ids), limit));
    end
end

function ensureDir(path)
    if ~exist(path, 'dir')
        mkdir(path);
    end
end

function R = axisAngleToRotm(axisangle)
    axis = axisangle(1:3);
    axis = axis / norm(axis);
    angle = axisangle(4);
    x = axis(1);
    y = axis(2);
    z = axis(3);
    K = [0, -z, y; z, 0, -x; -y, x, 0];
    R = eye(3) * cos(angle) + (1 - cos(angle)) * (axis(:) * axis(:)') + sin(angle) * K;
end

function savefunc(tsdfFile, Volume, surfaceSamples, vertices, faces, axisangle, closestPoints)
    save(tsdfFile, 'Volume', 'surfaceSamples', 'vertices', 'faces', 'axisangle', 'closestPoints');
end
