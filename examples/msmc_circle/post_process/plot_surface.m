clc; clear; close all;

% Plot the final surface-averaged quantities from the bundled MSMC case.
script_dir = fileparts(mfilename('fullpath'));
case_dir = fileparts(script_dir);
surf_file = fullfile(case_dir, 'output', 'circle_msmc.surf.1000.dat');
geom_file = fullfile(case_dir, 'data.circle');
output_file = fullfile(case_dir, 'output', 'circle_msmc_surface.png');

if ~isfile(surf_file)
    error('Surface dump not found: %s\nRun the MSMC example first.', surf_file);
end

surf = read_surface_dump(surf_file);
geom = read_surface_geometry(geom_file);

angle = nan(size(surf.id));
for i = 1:numel(surf.id)
    index = find(geom.id == surf.id(i), 1);
    if isempty(index)
        error('Surface ID %d is missing from %s.', surf.id(i), geom_file);
    end
    angle(i) = geom.angle(index);
end

[angle, order] = sort(angle);
values = [surf.press, surf.shx, surf.shy, surf.etot];
values = values(order, :);
labels = {'Pressure', 'x shear force', 'y shear force', 'Total energy flux'};

fig = figure('Color', 'w', 'Position', [100, 100, 1050, 720]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:4
    ax = nexttile(layout);
    plot(ax, angle, values(:, i), 'o-', 'LineWidth', 1.5, ...
         'MarkerSize', 4, 'Color', [0.00, 0.38, 0.72]);
    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [0, 180]);
    xticks(ax, 0:30:180);
    xlabel(ax, 'Surface angle (deg)');
    ylabel(ax, labels{i});
    set(ax, 'FontSize', 11, 'LineWidth', 1.0);
end
title(layout, 'SPARTA-MSMC: surface-averaged quantities');

exportgraphics(fig, output_file, 'Resolution', 300);
fprintf('Saved %s\n', output_file);

function data = read_surface_dump(filename)
    fid = fopen(filename, 'r');
    if fid < 0
        error('Cannot open surface dump: %s', filename);
    end
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    raw = textscan(fid, '%f', 'HeaderLines', 9);
    raw = raw{1};
    ncol = 5;
    if mod(numel(raw), ncol) ~= 0
        error('Unexpected number of values in surface dump: %s', filename);
    end
    array = reshape(raw, ncol, [])';
    data.id = array(:, 1);
    data.press = array(:, 2);
    data.shx = array(:, 3);
    data.shy = array(:, 4);
    data.etot = array(:, 5);
end

function geom = read_surface_geometry(filename)
    lines = readlines(filename);
    points_start = find(strtrim(lines) == "Points", 1);
    lines_start = find(strtrim(lines) == "Lines", 1);
    if isempty(points_start) || isempty(lines_start)
        error('Unexpected surface geometry format: %s', filename);
    end

    points = zeros(0, 3);
    segments = zeros(0, 3);
    for i = points_start + 1 : lines_start - 1
        values = sscanf(lines(i), '%f');
        if numel(values) == 3
            points(end + 1, :) = values'; %#ok<AGROW>
        end
    end
    for i = lines_start + 1 : numel(lines)
        values = sscanf(lines(i), '%f');
        if numel(values) == 3
            segments(end + 1, :) = values'; %#ok<AGROW>
        end
    end

    geom.id = segments(:, 1);
    geom.angle = zeros(size(geom.id));
    for i = 1:size(segments, 1)
        p1 = points(points(:, 1) == segments(i, 2), 2:3);
        p2 = points(points(:, 1) == segments(i, 3), 2:3);
        midpoint = 0.5 * (p1 + p2);
        geom.angle(i) = 180.0 - atan2d(midpoint(2), midpoint(1));
    end
end
