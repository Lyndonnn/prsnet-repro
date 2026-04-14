function [] = run_official_eval_preprocess(shapenet_dir, max_shapes)
%RUN_OFFICIAL_EVAL_PREPROCESS Build the official PRS-Net 1000-shape test set.
%
% Run this from the repository root in MATLAB:
%
%   addpath(genpath('D:\tools\gptoolbox'))
%   run_official_eval_preprocess('E:\ShapeNetCore.v2\ShapeNetCore.v2')
%
% Or set environment variables before starting MATLAB:
%
%   SHAPENET_DIR=E:\ShapeNetCore.v2\ShapeNetCore.v2
%   GPT_TOOLBOX_DIR=D:\tools\gptoolbox
%
% Then run:
%
%   run_official_eval_preprocess

    repo_root = fileparts(mfilename('fullpath'));
    preprocess_dir = fullfile(repo_root, 'preprocess');
    addpath(preprocess_dir);

    gptoolbox_dir = getenv('GPT_TOOLBOX_DIR');
    if ~isempty(gptoolbox_dir) && exist(gptoolbox_dir, 'dir')
        addpath(genpath(gptoolbox_dir));
    end

    if nargin < 1 || isempty(shapenet_dir)
        shapenet_dir = getenv('SHAPENET_DIR');
    end
    if isempty(shapenet_dir)
        shapenet_dir = fullfile(preprocess_dir, 'shapenet');
    end

    if nargin < 2 || isempty(max_shapes)
        max_shapes = Inf;
    end

    precompute_official_eval_set( ...
        shapenet_dir, ...
        fullfile(repo_root, 'datasets', 'shapenet_official_eval1000'), ...
        fullfile(repo_root, '1000.txt'), ...
        fullfile(repo_root, 'evaluation_old', 'gt_planes.mat'), ...
        fullfile(repo_root, 'evaluation_old', 'objs'), ...
        max_shapes);
end
