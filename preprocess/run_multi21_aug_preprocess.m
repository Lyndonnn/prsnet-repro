% Multi-category augmented preprocessing helper.
%
% This scans ShapeNetCore.v2 category folders in sorted order, takes the
% first max_categories categories that also have official split files, then
% preprocesses a small augmented dataset.
%
% Default size:
%   train: up to 21 categories * 50 ids * 4 augmentations = 4200 .mat files
%   test:  up to 21 categories * 10 ids = 210 .mat files
%
% Output:
%   ../datasets/shapenet_multi21_50train_10test_aug4/train/*.mat
%   ../datasets/shapenet_multi21_50train_10test_aug4/test/*.mat

shapenet_dir = 'E:\ShapeNetCore.v2';
output_dir = '..\datasets\shapenet_multi21_50train_10test_aug4';
split_dir = '.\data_split';

max_categories = 21;
train_limit_per_category = 50;
test_limit_per_category = 10;
num_aug_per_model = 4;

cd(fileparts(mfilename('fullpath')));

entries = dir(shapenet_dir);
names = {};
for i = 1:length(entries)
    name = entries(i).name;
    if entries(i).isdir && ~strcmp(name, '.') && ~strcmp(name, '..')
        train_split = fullfile(split_dir, [name, '_train.txt']);
        test_split = fullfile(split_dir, [name, '_test.txt']);
        if exist(train_split, 'file') && exist(test_split, 'file')
            names{end + 1} = name; %#ok<AGROW>
        else
            fprintf('[run_multi21_aug_preprocess] skip %s because split files are missing\n', name);
        end
    end
end

names = sort(names);
categories = names(1:min(max_categories, length(names)));

fprintf('[run_multi21_aug_preprocess] ShapeNet root: %s\n', shapenet_dir);
fprintf('[run_multi21_aug_preprocess] Output root: %s\n', output_dir);
fprintf('[run_multi21_aug_preprocess] Categories (%d): %s\n', length(categories), strjoin(categories, ', '));
fprintf('[run_multi21_aug_preprocess] train_limit/category=%d test_limit/category=%d aug/train_id=%d\n', ...
    train_limit_per_category, test_limit_per_category, num_aug_per_model);

precomputeShapeData( ...
    shapenet_dir, ...
    output_dir, ...
    split_dir, ...
    categories, ...
    train_limit_per_category, ...
    test_limit_per_category, ...
    num_aug_per_model)
