function[Stats, dPyMs, dPyMsopen, Python_full, MATLAB_full, Python_openfull, MATLAB_openfull] = DataAnalysesPyESPERLIRs(Variable, Method)

GLODAP = load("GLODAPv2.2022_Merged_Master_File.mat");
n = size(GLODAP.G2longitude(:, 1));
Python_Estimates = zeros(n(:, 1), 21); 

for eq = 1:16 
    varnames.(['eq_' num2str(eq)]) = [Variable num2str(eq)];
    filename = sprintf('PyESPER_%s_%s.mat', varnames.(['eq_' num2str(eq)]), Method);
    data = load(filename);
    pe = data.([Variable num2str(eq)]);
    Python_Estimate.(['eq_' num2str(eq)]) = pe.';
    Python_Estimates(:, eq) = Python_Estimate.(['eq_' num2str(eq)])(:, 1);
end

Python_Estimates(:, 17) = GLODAP.G2longitude(:,:);
Python_Estimates(:, 18) = GLODAP.G2latitude(:,:);
Python_Estimates(:, 19) = GLODAP.G2depth(:,:);

% Load indicator and measured value files
indfilename = horzcat('G2_', Variable, '_inds.mat');
Ind_data = load(indfilename);
Python_Estimates(:, 20) = Ind_data.KeepInds(:,:);
Python_Estimates(:, 21) = Ind_data.MeasValues(:,:);
% Reading the values for open ocean only
Python_open = Python_Estimates(Python_Estimates(:, 20) == 1, :);

% Load MATLAB Estimates
matfilename = horzcat('MATESPER_', Variable, '_LIR.mat');
matdata = load(matfilename);
MATLAB_Estimates = matdata.Estimates(:, :);
MATLAB_Estimates(:, 17) = GLODAP.G2longitude(:,:);
MATLAB_Estimates(:, 18) = GLODAP.G2latitude(:,:);
MATLAB_Estimates(:, 19) = GLODAP.G2depth(:,:);
MATLAB_Estimates(:, 20) = Ind_data.KeepInds(:,:);
MATLAB_Estimates(:, 21) = Ind_data.MeasValues(:,:);
MATLAB_open = MATLAB_Estimates(MATLAB_Estimates(:, 20) == 1, :);

uncname = horzcat('matdata.Uncertainties.', Variable);
MATLAB_Uncertainties = eval(uncname);
MATLAB_Uncertainties = MATLAB_Uncertainties(:, :);
MATLAB_Uncertaintiesopen = MATLAB_Uncertainties(MATLAB_Estimates(:, 20) == 1, :);

% Concatenate Python and MATLAB arrays
Python_full = vertcat(Python_Estimates(:, [1,17:21]), ...
    Python_Estimates(:, [2,17:21]), Python_Estimates(:, [3,17:21]), ...
    Python_Estimates(:, [4,17:21]), Python_Estimates(:, [5,17:21]), ...
    Python_Estimates(:, [6,17:21]), Python_Estimates(:, [7,17:21]), ...
    Python_Estimates(:, [8,17:21]), Python_Estimates(:, [9,17:21]), ...
    Python_Estimates(:, [10,17:21]), Python_Estimates(:, [11,17:21]), ...
    Python_Estimates(:, [12,17:21]), Python_Estimates(:, [13,17:21]), ...
    Python_Estimates(:, [14,17:21]), Python_Estimates(:, [15,17:21]), ...
    Python_Estimates(:, [16,17:21]));

MATLAB_full = vertcat(MATLAB_Estimates(:, 1), MATLAB_Estimates(:, 2), ...
    MATLAB_Estimates(:, 3), MATLAB_Estimates(:, 4), MATLAB_Estimates(:, 5), ...
    MATLAB_Estimates(:, 6), MATLAB_Estimates(:, 7), MATLAB_Estimates(:, 8), ...
    MATLAB_Estimates(:, 9), MATLAB_Estimates(:, 10), ...
    MATLAB_Estimates(:, 11), MATLAB_Estimates(:, 12), ...
    MATLAB_Estimates(:, 13), MATLAB_Estimates(:, 14), ...
    MATLAB_Estimates(:, 15), MATLAB_Estimates(:, 16));

MATLAB_Usfull = vertcat(MATLAB_Uncertainties(:, 1), MATLAB_Uncertainties(:, 2), ...
    MATLAB_Uncertainties(:, 3), MATLAB_Uncertainties(:, 4), MATLAB_Uncertainties(:, 5), ...
    MATLAB_Uncertainties(:, 6), MATLAB_Uncertainties(:, 7), MATLAB_Uncertainties(:, 8), ...
    MATLAB_Uncertainties(:, 9), MATLAB_Uncertainties(:, 10), ...
    MATLAB_Uncertainties(:, 11), MATLAB_Uncertainties(:, 12), ...
    MATLAB_Uncertainties(:, 13), MATLAB_Uncertainties(:, 14), ...
    MATLAB_Uncertainties(:, 15), MATLAB_Uncertainties(:, 16));

MATLAB_full = horzcat(MATLAB_full(:, 1), Python_full(:, 2:6), ...
    MATLAB_Usfull(:, 1));

Python_openfull = vertcat(Python_open(:, [1,17:21]), ...
    Python_open(:, [2,17:21]), Python_open(:, [3,17:21]), ...
    Python_open(:, [4,17:21]), Python_open(:, [5,17:21]), ...
    Python_open(:, [6,17:21]), Python_open(:, [7,17:21]), ...
    Python_open(:, [8,17:21]), Python_open(:, [9,17:21]), ...
    Python_open(:, [10,17:21]), Python_open(:, [11,17:21]), ...
    Python_open(:, [12,17:21]), Python_open(:, [13,17:21]), ...
    Python_open(:, [14,17:21]), Python_open(:, [15,17:21]), ...
    Python_open(:, [16,17:21]));

MATLAB_openfull = vertcat(MATLAB_open(:, 1), MATLAB_open(:, 2), ...
    MATLAB_open(:, 3), MATLAB_open(:, 4), MATLAB_open(:, 5), ...
    MATLAB_open(:, 6), MATLAB_open(:, 7), MATLAB_open(:, 8), ...
    MATLAB_open(:, 9), MATLAB_open(:, 10), MATLAB_open(:, 11), ...
    MATLAB_open(:, 12), MATLAB_open(:, 13), MATLAB_open(:, 14), ...
    MATLAB_open(:, 15), MATLAB_open(:, 16));

MATLAB_usopenfull = vertcat(MATLAB_Uncertaintiesopen(:, 1), MATLAB_Uncertaintiesopen(:, 2), ...
    MATLAB_Uncertaintiesopen(:, 3), MATLAB_Uncertaintiesopen(:, 4), MATLAB_Uncertaintiesopen(:, 5), ...
    MATLAB_Uncertaintiesopen(:, 6), MATLAB_Uncertaintiesopen(:, 7), MATLAB_Uncertaintiesopen(:, 8), ...
    MATLAB_Uncertaintiesopen(:, 9), MATLAB_Uncertaintiesopen(:, 10), MATLAB_Uncertaintiesopen(:, 11), ...
    MATLAB_Uncertaintiesopen(:, 12), MATLAB_Uncertaintiesopen(:, 13), MATLAB_Uncertaintiesopen(:, 14), ...
    MATLAB_Uncertaintiesopen(:, 15), MATLAB_Uncertaintiesopen(:, 16));

MATLAB_openfull = horzcat(MATLAB_openfull(:, 1), Python_openfull(:, 2:6), ...
    MATLAB_usopenfull(:, 1));

% Stats and check for mismatches
alleqs = size(MATLAB_full(:, 1));
alleqsopen = size(MATLAB_openfull(:, 1));
mismatch = zeros(alleqs(:, 1), 1);

for row=1:alleqs
    if isnan(MATLAB_full(row, 1)) && isnan(Python_full(row, 1))
        mismatch(row, 1) = false;
    elseif ~isnan(MATLAB_full(row, 1)) && ~isnan(Python_full(row, 1))
        mismatch(row, 1) = false;
    else
        mismatch(row, 1) = true;
        disp(row)
    end
end

dPyMs = zeros(alleqs(:, 1), 8);
dPyMs(:, 1) = Python_full(:, 1) - MATLAB_full(:, 1); % Python - MATLAB
dPyMs(:, 2) = Python_full(:, 1) - Python_full(:, 6);
dPyMs(:, 3) = MATLAB_full(:, 1) - MATLAB_full(:, 6);
dPyMs(:, 4) = Python_full(:, 2);
dPyMs(:, 5) = Python_full(:, 3);
dPyMs(:, 6) = Python_full(:, 4);
dPyMs(:, 7) = Python_full(:, 5);
dPyMs(:, 8) = Python_full(:, 6);

Stats.ds.meanPyM = mean(rmmissing(dPyMs(:, 1)));
Stats.ds.stdPyM = std(rmmissing(dPyMs(:, 1)));
Stats.ds.maxPyM = max(rmmissing(dPyMs(:, 1)));
Stats.ds.minPyM = min(rmmissing(dPyMs(:, 1)));
Stats.ds.meanPyMeas = mean(rmmissing(dPyMs(:, 2)));
Stats.ds.stdPyMeas = std(rmmissing(dPyMs(:, 2)));
Stats.ds.maxPyMeas = max(rmmissing(dPyMs(:, 2)));
Stats.ds.minPyMeas = min(rmmissing(dPyMs(:, 2)));
Stats.ds.meanMMeas = mean(rmmissing(dPyMs(:, 3)));
Stats.ds.stdMMeas = std(rmmissing(dPyMs(:, 3)));
Stats.ds.maxMMeas = max(rmmissing(dPyMs(:, 3)));
Stats.ds.minMMeas = min(rmmissing(dPyMs(:, 3)));
   
Stats.Pythons.mean = mean(rmmissing(Python_full(:, 1)));
Stats.Pythons.std = std(rmmissing(Python_full(:, 1)));
Stats.Pythons.max = max(rmmissing(Python_full(:, 1)));
Stats.Pythons.min = min(rmmissing(Python_full(:, 1)));

Stats.MATLABs.mean = mean(rmmissing(MATLAB_full(:, 1)));
Stats.MATLABs.std = std(rmmissing(MATLAB_full(:, 1)));
Stats.MATLABs.max = max(rmmissing(MATLAB_full(:, 1)));
Stats.MATLABs.min = min(rmmissing(MATLAB_full(:, 1)));

Stats.Uncertainties.mean = mean(rmmissing(MATLAB_Uncertainties));
Stats.Uncertainties.std = std(rmmissing(MATLAB_Uncertainties));
Stats.Uncertainties.max = max(rmmissing(MATLAB_Uncertainties));
Stats.Uncertainties.min = min(rmmissing(MATLAB_Uncertainties));

Stats.WeightedRMSEPyM = (rmse(rmmissing(MATLAB_full(:, 1)), ...
   rmmissing(Python_full(:, 1))))/(mean(rmmissing(MATLAB_full(:, 1))));
Python_estforrmse = Python_full(~isnan(Python_full(:, 6)), :);
Python_pyestforrmse = Python_estforrmse(~isnan(Python_estforrmse(:, 1)), :);
Stats.WeightedRMSEPyMeas = (rmse(Python_pyestforrmse(:, 1), ...
    Python_pyestforrmse(:, 6))/(mean(Python_pyestforrmse(:, 6))));
MATLAB_estforrmse = MATLAB_full(~isnan(MATLAB_full(:, 6)), :);
MATLAB_mestforrmse = MATLAB_estforrmse(~isnan(MATLAB_estforrmse(:, 1)), :);
Stats.WeightedRMSEMMeas = (rmse(MATLAB_mestforrmse(:, 1), ...
    MATLAB_mestforrmse(:, 6))/(mean(MATLAB_mestforrmse(:, 6))));

dPyMsopen = zeros(alleqsopen(:, 1), 8);
dPyMsopen(:, 1) = Python_openfull(:, 1) - MATLAB_openfull(:, 1); % Python - MATLAB
dPyMsopen(:, 2) = Python_openfull(:, 1) - Python_openfull(:, 6);
dPyMsopen(:, 3) = MATLAB_openfull(:, 1) - MATLAB_openfull(:, 6);
dPyMsopen(:, 4) = Python_openfull(:, 2);
dPyMsopen(:, 5) = Python_openfull(:, 3);
dPyMsopen(:, 6) = Python_openfull(:, 4);
dPyMsopen(:, 7) = Python_openfull(:, 5);
dPyMsopen(:, 8) = Python_openfull(:, 6);

Stats.dsopen.meanPyM = mean(rmmissing(dPyMsopen(:, 1)));
Stats.dsopen.stdPyM = std(rmmissing(dPyMsopen(:, 1)));
Stats.dsopen.maxPyM = max(rmmissing(dPyMsopen(:, 1)));
Stats.dsopen.minPyM = min(rmmissing(dPyMsopen(:, 1)));
Stats.dsopen.meanPyMeas = mean(rmmissing(dPyMsopen(:, 2)));
Stats.dsopen.stdPyMeas = std(rmmissing(dPyMsopen(:, 2)));
Stats.dsopen.maxPyMeas = max(rmmissing(dPyMsopen(:, 2)));
Stats.dsopen.minPyMeas = min(rmmissing(dPyMsopen(:, 2)));
Stats.dsopen.meanMMeas = mean(rmmissing(dPyMsopen(:, 3)));
Stats.dsopen.stdMMeas = std(rmmissing(dPyMsopen(:, 3)));
Stats.dsopen.maxMMeas = max(rmmissing(dPyMsopen(:, 3)));
Stats.dsopen.minMMeas = min(rmmissing(dPyMsopen(:, 3)));
   
Stats.Pythonsopen.mean = mean(rmmissing(Python_openfull(:, 1)));
Stats.Pythonsopen.std = std(rmmissing(Python_openfull(:, 1)));
Stats.Pythonsopen.max = max(rmmissing(Python_openfull(:, 1)));
Stats.Pythonsopen.min = min(rmmissing(Python_openfull(:, 1)));

Stats.MATLABsopen.mean = mean(rmmissing(MATLAB_openfull(:, 1)));
Stats.MATLABsopen.std = std(rmmissing(MATLAB_openfull(:, 1)));
Stats.MATLABsopen.max = max(rmmissing(MATLAB_openfull(:, 1)));
Stats.MATLABsopen.min = min(rmmissing(MATLAB_openfull(:, 1)));

Stats.Uncertaintiesopen.mean = mean(rmmissing(MATLAB_Uncertaintiesopen));
Stats.Uncertaintiesopen.std = std(rmmissing(MATLAB_Uncertaintiesopen));
Stats.Uncertaintiesopen.max = max(rmmissing(MATLAB_Uncertaintiesopen));
Stats.Uncertaintiesopen.min = min(rmmissing(MATLAB_Uncertaintiesopen));

Stats.WeightedRMSEPyMopen = (rmse(rmmissing(MATLAB_openfull(:, 1)), ...
   rmmissing(Python_openfull(:, 1))))/(mean(rmmissing(MATLAB_openfull(:, 1))));
Python_estforrmseopen = Python_openfull(~isnan(Python_openfull(:, 6)), :);
Python_pyestforrmseopen = Python_estforrmseopen(~isnan(Python_estforrmseopen(:, 1)), :);
Stats.WeightedRMSEPyMeasopen = (rmse(Python_pyestforrmseopen(:, 1), ...
    Python_pyestforrmseopen(:, 6))/(mean(Python_pyestforrmseopen(:, 6))));
MATLAB_estforrmseopen = MATLAB_openfull(~isnan(MATLAB_openfull(:, 6)), :);
MATLAB_mestforrmseopen = MATLAB_estforrmseopen(~isnan(MATLAB_estforrmseopen(:, 1)), :);
Stats.WeightedRMSEMMeasopen = (rmse(MATLAB_mestforrmseopen(:, 1), ...
    MATLAB_mestforrmseopen(:, 6))/(mean(MATLAB_mestforrmseopen(:, 6))));

nonansfull = rmmissing(dPyMs(:, 1));
Together = horzcat(dPyMs(:, :), MATLAB_full(:, 7));
Overthresh_full = Together(abs(Together(:, 1))>(2.*Together(:, 9)), :);
Stats.ns.overthresh = size(Overthresh_full(:, 1));
Stats.ns.whole = size(Together(:, 1));
Stats.ns.underthresh = size(Together(:, 1)) - size(Overthresh_full(:, 1));
nonansfullopen = rmmissing(dPyMsopen(:, 1));
Togetheropen = horzcat(dPyMsopen(:, :), MATLAB_openfull(:, 7));
Overthresh_fullopen = Togetheropen(abs(Togetheropen(:, 1))>(2.*Togetheropen(:, 9)), :);
Stats.ns.overthresho = size(Overthresh_fullopen(:, 1));
Stats.ns.open = size(Togetheropen(:, 1));
Stats.ns.underthresho = size(Togetheropen(:, 1)) - size(Overthresh_fullopen(:, 1));

end