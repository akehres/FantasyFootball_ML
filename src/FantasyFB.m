% Austin Kehres
% ML Model Fantasy Football
% Updated for 2026 Prediction (VBD Box Plot & File Output)

clear
clc
close all

%% Declarations
file_list = ["stats_player_reg_2019.csv", "stats_player_reg_2020.csv", ...
             "stats_player_reg_2021.csv", "stats_player_reg_2022.csv", ...
             "stats_player_reg_2023.csv", "stats_player_reg_2024.csv", ...
             "stats_player_reg_2025.csv"];

num_teams = 12; % Number of fantasy teams
roster_settings = struct('QB', 1, 'RB', 2, 'WR', 3, 'TE', 1);
num_runs = 10; % Runs per phase

cols_to_keep = {'player_id', 'player_display_name', 'position', 'season', ...
                'recent_team', 'age', 'games', 'fantasy_points_ppr', ...
                'passing_yards', 'passing_tds', 'attempts', 'completions', ...
                'passing_epa', 'passing_first_downs', 'passing_air_yards', ...
                'passing_cpoe', 'sacks_suffered', ...
                'rushing_yards', 'rushing_tds', 'carries', 'rushing_epa', ...
                'rushing_first_downs', 'rushing_fumbles_lost', ...
                'receptions', 'receiving_yards', 'receiving_tds', 'targets', ...
                'receiving_epa', 'target_share', 'wopr', 'air_yards_share', ...
                'receiving_first_downs', 'receiving_yards_after_catch', 'racr'}; 

fprintf('Importing History (2019-2025)...\n');

%% Helper function
function T = load_clean_data(file_list, cols)
    T = table();
    for i = 1:length(file_list)
        filename = file_list(i);
        if isfile(filename)
            opts = detectImportOptions(filename);
            opts.VariableNamingRule = 'preserve';
            tempT = readtable(filename, opts);
            tempT.Properties.VariableNames = lower(tempT.Properties.VariableNames);
            existing_cols = intersect(tempT.Properties.VariableNames, cols);
            tempT = tempT(:, existing_cols);
            
            if ismember('age', tempT.Properties.VariableNames)
                tempT.age(isnan(tempT.age)) = 25;
            else
                tempT.age = repmat(25, height(tempT), 1);
            end
            
            numeric_cols = {'fantasy_points_ppr', 'wopr', 'target_share', ...
                            'passing_epa', 'rushing_epa', 'receiving_epa', ...
                            'passing_cpoe', 'racr', 'sacks_suffered'};
            for k = 1:length(numeric_cols)
                c = numeric_cols{k};
                if ismember(c, tempT.Properties.VariableNames)
                    tempT.(c)(isnan(tempT.(c))) = 0;
                end
            end
            tempT = tempT(ismember(tempT.position, {'QB', 'RB', 'WR', 'TE'}), :);
            T = [T; tempT];
        end
    end
end
AllHistory = load_clean_data(file_list, cols_to_keep);

% Grouping
TeamPoints = groupsummary(AllHistory, {'season', 'recent_team'}, 'sum', 'fantasy_points_ppr');
TeamPoints.off_rank = zeros(height(TeamPoints), 1);
seasons = unique(TeamPoints.season);
for s = 1:length(seasons)
    yr = seasons(s);
    idx = TeamPoints.season == yr;
    [~, sortIdx] = sort(TeamPoints.sum_fantasy_points_ppr(idx), 'descend');
    ranks = zeros(length(sortIdx), 1);
    ranks(sortIdx) = 1:length(sortIdx);
    TeamPoints.off_rank(find(idx)) = ranks;
end
AllHistory = innerjoin(AllHistory, TeamPoints(:, {'season', 'recent_team', 'off_rank'}), ...
    'Keys', {'season', 'recent_team'});

AllHistory.NextSeason = AllHistory.season + 1;
FutureData = AllHistory(:, {'player_id', 'season', 'fantasy_points_ppr'});
FutureData.Properties.VariableNames{'season'} = 'JoinSeason';
FutureData.Properties.VariableNames{'fantasy_points_ppr'} = 'TargetNextYearPoints';

PairedData = innerjoin(AllHistory, FutureData, ...
    'LeftKeys', {'player_id', 'NextSeason'}, ...
    'RightKeys', {'player_id', 'JoinSeason'});

% Feature Selector
function feats = get_pos_features(pos)
    common = {'fantasy_points_ppr', 'games', 'off_rank'}; 
    switch pos
        case 'QB'
            feats = [common, {'passing_yards', 'passing_tds', 'attempts', ...
                              'passing_epa', 'passing_cpoe', 'passing_first_downs', ...
                              'passing_air_yards', 'sacks_suffered', ...
                              'rushing_yards', 'rushing_tds', 'carries', ...
                              'rushing_epa', 'rushing_first_downs'}];
        case 'RB'
            feats = [common, {'rushing_yards', 'rushing_tds', 'carries', ...
                              'rushing_epa', 'rushing_first_downs', 'rushing_fumbles_lost', ...
                              'receptions', 'receiving_yards', 'receiving_tds', ...
                              'targets', 'receiving_epa', 'target_share', 'receiving_first_downs'}];
        case {'WR', 'TE'}
            feats = [common, {'receptions', 'receiving_yards', 'receiving_tds', ...
                              'targets', 'receiving_epa', 'target_share', 'wopr', ...
                              'air_yards_share', 'receiving_first_downs', ...
                              'receiving_yards_after_catch', 'racr'}];
    end
end

positions = {'QB', 'RB', 'WR', 'TE'};

%% Testing against 2025
fprintf('\n--- PHASE 1: VALIDATION (Backtesting 2025) ---\n');
fprintf('Training on 2019-2024 data only. Hiding 2025 results from model.\n');
Validation2025 = AllHistory(AllHistory.season == 2024, ...
    {'player_id', 'player_display_name', 'position', 'fantasy_points_ppr'});
Validation2025.PointsSum = zeros(height(Validation2025), 1);

for run = 1:num_runs
    TrainIdx = (PairedData.NextSeason < 2025);
    CurrentTrain = PairedData(TrainIdx, :);
    
    for p = 1:length(positions)
        pos = positions{p};
        PosTrain = CurrentTrain(strcmp(CurrentTrain.position, pos), :);
        features = get_pos_features(pos);
        valid_feats = intersect(features, PosTrain.Properties.VariableNames);
        
        rng(run); 
        model = fitrensemble(PosTrain, 'TargetNextYearPoints', ...
            'Method', 'Bag', 'NumLearningCycles', 150, ... 
            'Learners', templateTree('MinLeafSize', 5), ...
            'PredictorNames', valid_feats);
        CurrentInput = AllHistory(AllHistory.season == 2024 & strcmp(AllHistory.position, pos), :);
        
        if height(CurrentInput) > 0
            preds = predict(model, CurrentInput);
            TempPreds = table(CurrentInput.player_id, preds, 'VariableNames', {'player_id', 'Pts'});
            [Lia, Locb] = ismember(TempPreds.player_id, Validation2025.player_id);
            if any(Lia)
                Validation2025.PointsSum(Locb(Lia)) = Validation2025.PointsSum(Locb(Lia)) + TempPreds.Pts(Lia);
            end
        end
    end
end

Validation2025.Predicted2025 = Validation2025.PointsSum / num_runs;
Actual2025 = AllHistory(AllHistory.season == 2025, {'player_id', 'fantasy_points_ppr'});
Actual2025.Properties.VariableNames{'fantasy_points_ppr'} = 'ActualPoints';

Results2025 = innerjoin(Validation2025, Actual2025, 'Keys', 'player_id');
err = Results2025.Predicted2025 - Results2025.ActualPoints;
mae = mean(abs(err));
rmse = sqrt(mean(err.^2));
r2 = 1 - sum(err.^2) / sum((Results2025.ActualPoints - mean(Results2025.ActualPoints)).^2);

fprintf('\n2025 VALIDATION RESULTS:\n');
fprintf('  MAE:  %.2f points\n', mae);
fprintf('  RMSE: %.2f points\n', rmse);
fprintf('  R^2:  %.2f \n', r2);

%% Predicting
fprintf('\n--- PHASE 2: FORECASTING 2026 ---\n');
fprintf('Retraining on ALL history (2019-2025) to predict 2026.\n');

MasterForecast = AllHistory(AllHistory.season == 2025, ...
    {'player_id', 'player_display_name', 'position', 'age', 'off_rank'});
MasterForecast.PointsSum = zeros(height(MasterForecast), 1);

for run = 1:num_runs
    fprintf('Run %d/%d... ', run, num_runs);
    
    for p = 1:length(positions)
        pos = positions{p};
        PosTrain = PairedData(strcmp(PairedData.position, pos), :);
        features = get_pos_features(pos);
        valid_feats = intersect(features, PosTrain.Properties.VariableNames);
        rng(run + 1000);
        model = fitrensemble(PosTrain, 'TargetNextYearPoints', ...
            'Method', 'Bag', 'NumLearningCycles', 200, ... 
            'Learners', templateTree('MinLeafSize', 5), ...
            'PredictorNames', valid_feats);
        CurrentInput = AllHistory(AllHistory.season == 2025 & strcmp(AllHistory.position, pos), :);
        
        if height(CurrentInput) > 0
            preds = predict(model, CurrentInput);
            TempPreds = table(CurrentInput.player_id, preds, 'VariableNames', {'player_id', 'Pts'});
            [Lia, Locb] = ismember(TempPreds.player_id, MasterForecast.player_id);
            if any(Lia)
                MasterForecast.PointsSum(Locb(Lia)) = MasterForecast.PointsSum(Locb(Lia)) + TempPreds.Pts(Lia);
            end
        end
    end
    fprintf('Done.\n');
end

% Stats
MasterForecast.Predicted2026 = MasterForecast.PointsSum / num_runs;

positions_list = unique(string(MasterForecast.position)); 
baseline_map = dictionary(positions_list, zeros(length(positions_list),1));

for i = 1:length(positions_list)
    p = positions_list(i);
    subTable = MasterForecast(string(MasterForecast.position) == p, :);
    subTable = sortrows(subTable, 'Predicted2026', 'descend');
    
    limit = min(num_teams * roster_settings.(p), height(subTable));
    if limit > 0
        baseline_map(p) = subTable.Predicted2026(limit);
    end
end

MasterForecast.VBD = zeros(height(MasterForecast), 1);
for i = 1:height(MasterForecast)
    if iscell(MasterForecast.position)
        p = string(MasterForecast.position{i});
    else
        p = string(MasterForecast.position(i));
    end
    MasterForecast.VBD(i) = MasterForecast.Predicted2026(i) - baseline_map(p);
end

DraftBoard = sortrows(MasterForecast, 'VBD', 'descend');

%% Outputs
output_filename = '2026_Fantasy_Draft_Board.csv';
writetable(DraftBoard(:, {'player_display_name', 'position', 'off_rank', 'Predicted2026', 'VBD'}), output_filename);
fprintf('\nFull Draft Board saved to file: %s\n', output_filename);

figure;

subplot(2,1,1);
gscatter(Results2025.ActualPoints, Results2025.Predicted2025, Results2025.position, 'rbkm', '.', 15);
hold on; plot([0 450], [0 450], 'k--', 'LineWidth', 1); hold off;
title(['Model Accuracy Check (2025 Test): R^2 = ' num2str(r2, '%.2f')]);
xlabel('Actual 2025 Points'); ylabel('Predicted 2025 Points');
legend('Location', 'best'); grid on;

subplot(2,1,2);
boxchart(categorical(DraftBoard.position), DraftBoard.VBD);
title('2026 VBD Projections');
ylabel('Projected VBD');
grid on;