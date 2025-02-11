function[Stats, Data2] = DataAnalysesPyESPER_NNs(Variable, Method)

% Load the data.
% Data column order is 1: order, 2-17: Python estimates from 16 equations
% (in order), 18-33: MATLAB estimates from 16 equations, 34-36: latitude, 
% longitude, depth
filename = horzcat('PyESPER_', Variable, '_', Method, '.csv');
filename2 = horzcat('PyESPER_', Variable, '_', Method, '2.csv');
data = csvread(filename, 1);
data2 = csvread(filename2, 1);
data = vertcat(data, data2);
data(isnan(data)) = 0; % Preprocessing nans to 0

% Create a structure and assign aliases for names

% Extract Py and M data and coordinates
py_indices = 2:17;
m_indices = 18:33;
coord_indices = 34:36;
meas_index = 37;
G2_indices = 38:44;
keep_index = 45;

for i = 1:16
    py_data = data(data(:, py_indices(i)) ~= 0, ...
        [py_indices(i), coord_indices, meas_index, keep_index]);
    m_data = data(data(:, m_indices(i)) ~= 0, ...
        [m_indices(i), coord_indices, meas_index, keep_index]);
    meas_data = data(data(:, py_indices(i)) ~= 0 & ...
        data(:, meas_index) ~= -9999 & data(:, meas_index) ~= 0, ...
        [py_indices(i), m_indices(i), coord_indices, meas_index, keep_index]);
    py_data_open = data(data(:, py_indices(i)) ~= 0 & ...
        data(:, keep_index) ~= 0, ...
        [py_indices(i), coord_indices, meas_index, keep_index]);
    m_data_open = data(data(:, m_indices(i)) ~= 0 & ...
        data(:, keep_index) ~= 0, ...
        [m_indices(i), coord_indices, meas_index, keep_index]);
    meas_data_open = data(data(:, py_indices(i)) ~= 0 & ...
        data(:, meas_index) ~= -9999 & data(:, meas_index) ~= 0 & ...
        data(:, keep_index) ~= 0, ...
        [py_indices(i), m_indices(i), coord_indices, meas_index, keep_index]);
    % Assign to Py and M structures dynamically
    Py.(['Py_', num2str(i)]) = py_data;
    Py.(['Py_Open_', num2str(i)]) = py_data_open;
    M.(['M_', num2str(i)]) = m_data;
    M.(['M_Open_', num2str(i)]) = m_data_open;
    Meas.(['Meas_', num2str(i)]) = meas_data;
    Meas.(['Meas_Open_', num2str(i)]) = meas_data_open;

   % Calculate differences and assign to d structure dynamically
    d.(['d_', num2str(i)]) = Py.(['Py_', num2str(i)]) - M.(['M_', num2str(i)]);
    d.(['d_Open_', num2str(i)]) = Py.(['Py_Open_', num2str(i)]) - M.(['M_Open_', num2str(i)]);
    dPyM.(['dPyM_', num2str(i)]) = Meas.(['Meas_', num2str(i)])(:, 1) - ...
        Meas.(['Meas_', num2str(i)])(:, 6);
    dPyM.(['dPyM_Open_', num2str(i)]) = Meas.(['Meas_Open_', num2str(i)])(:, 1) - ...
        Meas.(['Meas_Open_', num2str(i)])(:, 6);
    dMM.(['dMM_', num2str(i)]) = Meas.(['Meas_', num2str(i)])(:, 2) - ...
        Meas.(['Meas_', num2str(i)])(:, 6);
    dMM.(['dMM_Open_', num2str(i)]) = Meas.(['Meas_Open_', num2str(i)])(:, 2) - ...
        Meas.(['Meas_Open_', num2str(i)])(:, 6);
end

Data.Py = Py;
Data.M = M;
Data.d = d;
Data.Meas = Meas;
Data.dPyM = dPyM;
Data.dMM = dMM;
% Initialize Stats matrix to store statistics
IndividStats = zeros(40, 16);

% Define data variables
dataSets = {'d.d_', 'M.M_', 'Py.Py_', 'dPyM.dPyM_', 'dMM.dMM_', ...
    'd.d_Open_', 'M.M_Open_', 'Py.Py_Open_', 'dPyM.dPyM_Open_', 'dMM.dMM_Open_'};
statFunctions = {@mean, @std, @max, @min};

% Loop through each dataset and populate Stats matrix with statistics
for i = 1:4
    for j = 1:16
        for k = 1:10
            % Evaluate the appropriate function on the dataset
            func = statFunctions{i};
            evalStr = [func2str(func), '(', dataSets{k}, num2str(j), '(:, 1))'];
            IndividStats((k-1)*4+i, j) = eval(evalStr); % Compute and assign the value
        end
    end
end
Stats2.IndividStats = IndividStats;
% Compile data
d_compiled = vertcat(d.d_1, d.d_2, d.d_3, d.d_4, d.d_5, d.d_6, d.d_7, ...
    d.d_8, d.d_9, d.d_10, d.d_11, d.d_12, d.d_13, d.d_14, d.d_15, d.d_16);
d_Open_compiled = vertcat(d.d_Open_1, d.d_Open_2, d.d_Open_3, d.d_Open_4,...
    d.d_Open_5, d.d_Open_6, d.d_Open_7, d.d_Open_8, d.d_Open_9, d.d_Open_10,...
    d.d_Open_11, d.d_Open_12, d.d_Open_13, d.d_Open_14, d.d_Open_15, d.d_Open_16);
M_compiled = vertcat(M.M_1, M.M_2, M.M_3, M.M_4, M.M_5, M.M_6, M.M_7, ...
    M.M_8, M.M_9, M.M_10, M.M_11, M.M_12, M.M_13, M.M_14, M.M_15, M.M_16);
M_Open_compiled = vertcat(M.M_Open_1, M.M_Open_2, M.M_Open_3, M.M_Open_4,...
    M.M_Open_5, M.M_Open_6, M.M_Open_7, M.M_Open_8, M.M_Open_9, M.M_Open_10,...
    M.M_Open_11, M.M_Open_12, M.M_Open_13, M.M_Open_14, M.M_Open_15, M.M_Open_16);
Py_compiled = vertcat(Py.Py_1, Py.Py_2, Py.Py_3, Py.Py_4, Py.Py_5, ...
    Py.Py_6, Py.Py_7, Py.Py_8, Py.Py_9, Py.Py_10, Py.Py_11, Py.Py_12, ...
    Py.Py_13, Py.Py_14, Py.Py_15, Py.Py_16);
Py_Open_compiled = vertcat(Py.Py_Open_1, Py.Py_Open_2, Py.Py_Open_3, ...
    Py.Py_Open_4, Py.Py_Open_5, Py.Py_Open_6, Py.Py_Open_7, Py.Py_Open_8, ...
    Py.Py_Open_9, Py.Py_Open_10, Py.Py_Open_11, Py.Py_Open_12, ...
    Py.Py_Open_13, Py.Py_Open_14, Py.Py_Open_15, Py.Py_Open_16);
dMM_compiled = vertcat(dMM.dMM_1, dMM.dMM_2, dMM.dMM_3, dMM.dMM_4, ...
    dMM.dMM_5, dMM.dMM_6, dMM.dMM_7, dMM.dMM_8, dMM.dMM_9, dMM.dMM_10, ...
    dMM.dMM_11, dMM.dMM_12, dMM.dMM_13, dMM.dMM_14, dMM.dMM_15, ...
    dMM.dMM_16);
dMM_Open_compiled = vertcat(dMM.dMM_Open_1, dMM.dMM_Open_2, dMM.dMM_Open_3, ...
    dMM.dMM_Open_4, dMM.dMM_Open_5, dMM.dMM_Open_6, dMM.dMM_Open_7, ...
    dMM.dMM_Open_8, dMM.dMM_Open_9, dMM.dMM_Open_10, dMM.dMM_Open_11, ...
    dMM.dMM_Open_12, dMM.dMM_Open_13, dMM.dMM_Open_14, dMM.dMM_Open_15, ...
    dMM.dMM_Open_16);
dPyM_compiled = vertcat(dPyM.dPyM_1, dPyM.dPyM_2, dPyM.dPyM_3, dPyM.dPyM_4, ...
    dPyM.dPyM_5, dPyM.dPyM_6, dPyM.dPyM_7, dPyM.dPyM_8, dPyM.dPyM_9, ...
    dPyM.dPyM_10, dPyM.dPyM_11, dPyM.dPyM_12, dPyM.dPyM_13, dPyM.dPyM_14, ...
    dPyM.dPyM_15, dPyM.dPyM_16);
dPyM_Open_compiled = vertcat(dPyM.dPyM_Open_1, dPyM.dPyM_Open_2, ...
    dPyM.dPyM_Open_3, dPyM.dPyM_Open_4, dPyM.dPyM_Open_5, dPyM.dPyM_Open_6, ...
    dPyM.dPyM_Open_7, dPyM.dPyM_Open_8, dPyM.dPyM_Open_9, dPyM.dPyM_Open_10, ...
    dPyM.dPyM_Open_11, dPyM.dPyM_Open_12, dPyM.dPyM_Open_13, dPyM.dPyM_Open_14, ...
    dPyM.dPyM_Open_15, dPyM.dPyM_Open_16);
Meas_compiled = vertcat(Meas.Meas_1, Meas.Meas_2, Meas.Meas_3, Meas.Meas_4,...
    Meas.Meas_5, Meas.Meas_6, Meas.Meas_7, Meas.Meas_8, Meas.Meas_9, ...
    Meas.Meas_10, Meas.Meas_11, Meas.Meas_12, Meas.Meas_13, Meas.Meas_14, ...
    Meas.Meas_15, Meas.Meas_16);
Meas_Open_compiled = vertcat(Meas.Meas_Open_1, Meas.Meas_Open_2, ...
    Meas.Meas_Open_3, Meas.Meas_Open_4, Meas.Meas_Open_5, Meas.Meas_Open_6, ...
    Meas.Meas_Open_7, Meas.Meas_Open_8, Meas.Meas_Open_9, Meas.Meas_Open_10, ...
    Meas.Meas_Open_11, Meas.Meas_Open_12, Meas.Meas_Open_13, Meas.Meas_Open_14, ...
    Meas.Meas_Open_15, Meas.Meas_Open_16);
Data2.dPyM = d_compiled;
Data2.dPyMo = d_Open_compiled;
Data2.MAT = M_compiled;
Data2.MATo = M_Open_compiled;
Stats.dcomp.mean = mean(d_compiled(:, 1));
Stats.dcomp.std = std(d_compiled(:, 1));
Stats.dcomp.max = max(d_compiled(:, 1));
Stats.dcomp.min = min(d_compiled(:, 1));
Stats.dOcomp.mean = mean(d_Open_compiled(:, 1));
Stats.dOcomp.std = std(d_Open_compiled(:, 1));
Stats.dOcomp.max = max(d_Open_compiled(:, 1));
Stats.dOcomp.min = min(d_Open_compiled(:, 1));
Stats.Pycomp.mean = mean(Py_compiled(:, 1));
Stats.Pycomp.std = std(Py_compiled(:, 1));
Stats.Pycomp.max = max(Py_compiled(:, 1));
Stats.Pycomp.min = min(Py_compiled(:, 1));
Stats.PyOcomp.mean = mean(Py_Open_compiled(:, 1));
Stats.PyOcomp.std = std(Py_Open_compiled(:, 1));
Stats.PyOcomp.max = max(Py_Open_compiled(:, 1));
Stats.PyOcomp.min = min(Py_Open_compiled(:, 1));
Stats.Mcomp.mean = mean(M_compiled(:, 1));
Stats.Mcomp.std = std(M_compiled(:, 1));
Stats.Mcomp.max = max(M_compiled(:, 1));
Stats.Mcomp.min = min(M_compiled(:, 1));
Stats.MOcomp.mean = mean(M_Open_compiled(:, 1));
Stats.MOcomp.std = std(M_Open_compiled(:, 1));
Stats.MOcomp.max = max(M_Open_compiled(:, 1));
Stats.MOcomp.min = min(M_Open_compiled(:, 1));
Stats.dMMcomp.mean = mean(dMM_compiled(:, 1));
Stats.dMMcomp.std = std(dMM_compiled(:, 1));
Stats.dMMcomp.max = max(dMM_compiled(:, 1));
Stats.dMMcomp.min = min(dMM_compiled(:, 1));
Stats.dMMOcomp.mean = mean(dMM_Open_compiled(:, 1));
Stats.dMMOcomp.std = std(dMM_Open_compiled(:, 1));
Stats.dMMOcomp.max = max(dMM_Open_compiled(:, 1));
Stats.dMMOcomp.min = min(dMM_Open_compiled(:, 1));
Stats.dPyMcomp.mean = mean(dPyM_compiled(:, 1));
Stats.dPyMcomp.std = std(dPyM_compiled(:, 1));
Stats.dPyMcomp.max = max(dPyM_compiled(:, 1));
Stats.dPyMcomp.min = min(dPyM_compiled(:, 1));
Stats.dPyMOcomp.mean = mean(dPyM_Open_compiled(:, 1));
Stats.dPyMOcomp.std = std(dPyM_Open_compiled(:, 1));
Stats.dPyMOcomp.max = max(dPyM_Open_compiled(:, 1));
Stats.dPyMOcomp.min = min(dPyM_Open_compiled(:, 1));
Stats.Meascomp.mean = mean(Meas_compiled(:, 6));
Stats.Meascomp.std = std(Meas_compiled(:, 6));
Stats.Meascomp.max = max(Meas_compiled(:, 6));
Stats.Meascomp.min = min(Meas_compiled(:, 6));
Stats.MeasOcomp.mean = mean(Meas_Open_compiled(:, 1));
Stats.MeasOcomp.std = std(Meas_Open_compiled(:, 1));
Stats.MeasOcomp.max = max(Meas_Open_compiled(:, 1));
Stats.MeasOcomp.min = min(Meas_Open_compiled(:, 1));
Stats.RMSE.MPy = rmse(M_compiled(:, 1), Py_compiled(:, 1));
Stats.RMSE.MPyO = rmse(M_Open_compiled(:, 1), Py_Open_compiled(:, 1));
Stats.RMSE.MMeas = rmse(Meas_compiled(:, 6), Meas_compiled(:, 2));
Stats.RMSE.MMeasO = rmse(Meas_Open_compiled(:, 6), Meas_Open_compiled(:, 2));
Stats.RMSE.PyMeas = rmse(Meas_compiled(:, 6), Meas_compiled(:, 1));
Stats.RMSE.PyMeasO = rmse(Meas_Open_compiled(:, 6), Meas_Open_compiled(:, 1));
Stats.RMSD.MPy = nanmean(d_compiled.^2).^(1/2);
Stats.WeightedRMSE.MPy = (rmse(M_compiled(:, 1), Py_compiled(:, 1)))...
    /(mean(M_compiled(:, 1)));
Stats.WeightedRMSE.MPyO = (rmse(M_Open_compiled(:, 1), Py_Open_compiled(:, 1)))...
    /(mean(M_Open_compiled(:, 1)));
Stats.WeightedRMSE.MMeas = (rmse(Meas_compiled(:, 6), Meas_compiled(:, 2)))...
    /(mean(Meas_compiled(:, 6)));
Stats.WeightedRMSE.MMeasO = (rmse(Meas_Open_compiled(:, 6), Meas_Open_compiled(:, 2)))...
    /(mean(Meas_Open_compiled(:, 6)));
Stats.WeightedRMSE.PyMeas = (rmse(Meas_compiled(:, 6), Meas_compiled(:, 1)))...
    /(mean(Meas_compiled(:, 6)));
Stats.WeightedRMSE.PyMeasO = (rmse(Meas_Open_compiled(:, 6), Meas_Open_compiled(:, 1)))...
    /(mean(Meas_Open_compiled(:, 6)));
Stats.Meascomp.Meas_compiled = Meas_compiled;
Stats.MeasOcomp.MeasO_compiled = Meas_Open_compiled;
Stats.Meascomp.Meas_compiled(:, 14) = Stats.Meascomp.Meas_compiled(:, 2) - Stats.Meascomp.Meas_compiled(:, 6);
Stats.MeasOcomp.MeasO_compiled(:, 14) = Stats.MeasOcomp.MeasO_compiled(:, 2) - Stats.MeasOcomp.MeasO_compiled(:, 6);

% Checking mean and total differences
Stats2.Tot_compiled = horzcat(Py_compiled(:, 1), M_compiled(:, 1), ...
   d_compiled(:, 1), M_compiled(:, 2:4), ...
   (d_compiled(:, 1)./M_compiled(:, 1))*100);
Stats2.TotO_compiled = horzcat(Py_Open_compiled(:, 1), M_Open_compiled(:, 1), ...
   d_Open_compiled(:, 1), M_Open_compiled(:, 2:4), ...
   (d_Open_compiled(:, 1)./M_Open_compiled(:, 1))*100);


end