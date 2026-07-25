clc; clear; close all;

% Plot the final cell-averaged temperature field from the bundled MSMC case.
script_dir = fileparts(mfilename('fullpath'));
case_dir = fileparts(script_dir);
input_file = fullfile(case_dir, 'output', 'circle_msmc.grid.1000.dat');
output_file = fullfile(case_dir, 'output', 'circle_msmc_temperature.png');

if ~isfile(input_file)
    error('Grid dump not found: %s\nRun the MSMC example first.', input_file);
end

fid = fopen(input_file, 'r');
if fid < 0
    error('Cannot open grid dump: %s', input_file);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
raw = textscan(fid, '%f', 'HeaderLines', 9);
raw = raw{1};

ncol = 9;
if mod(numel(raw), ncol) ~= 0
    error('Unexpected number of values in grid dump: %s', input_file);
end

data = reshape(raw, ncol, [])';
xc = data(:, 2);
yc = data(:, 3);
volume = data(:, 4);
temperature = data(:, 8);

mask = volume > 0.0;
xc = xc(mask);
yc = yc(mask);
volume = volume(mask);
temperature = temperature(mask);

cell_width = sqrt(volume);
xpatch = [xc - 0.5 * cell_width, xc + 0.5 * cell_width, ...
          xc + 0.5 * cell_width, xc - 0.5 * cell_width]';
ypatch = [yc - 0.5 * cell_width, yc - 0.5 * cell_width, ...
          yc + 0.5 * cell_width, yc + 0.5 * cell_width]';

fig = figure('Color', 'w', 'Position', [100, 100, 1000, 520]);
patch('XData', xpatch, 'YData', ypatch, 'CData', temperature, ...
      'FaceColor', 'flat', 'EdgeColor', 'none');
hold on;

theta = linspace(0.0, pi, 400);
xsurf = 0.13 + 0.04 * cos(theta);
ysurf = 0.04 * sin(theta);
fill([xsurf, 0.17, 0.09], [ysurf, 0.0, 0.0], 'w', ...
     'EdgeColor', 'k', 'LineWidth', 1.5);

axis equal tight;
xlim([0.0, 0.30]);
ylim([0.0, 0.15]);
box on;
set(gca, 'Layer', 'top', 'FontSize', 13, 'LineWidth', 1.0);
xlabel('x (m)');
ylabel('y (m)');
title('SPARTA-MSMC: translational temperature');
cb = colorbar;
cb.Label.String = 'Temperature (K)';
colormap(turbo(256));

exportgraphics(fig, output_file, 'Resolution', 300);
fprintf('Saved %s\n', output_file);
