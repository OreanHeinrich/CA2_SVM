% Task 2: classify train/test sets and report Table 1 accuracies.

clear;
load train.mat;
load test.mat;

if ~isfile('task1_models.mat')
    task1_compute_models;
end
load task1_models.mat;

Xtr = double(train_data);
ytr = double(train_label(:));
Xte = double(test_data);
yte = double(test_label(:));

results = struct();

% Hard-margin linear
[results.hardLinearTrain, results.hardLinearTest] = evaluate_one(models.hard_linear, Xtr, ytr, Xte, yte);

% Hard-margin polynomial p = 2..5
results.hardPolyTrain = nan(numel(pHard), 1);
results.hardPolyTest = nan(numel(pHard), 1);
for i = 1:numel(pHard)
    [results.hardPolyTrain(i), results.hardPolyTest(i)] = evaluate_one(models.hard_poly{i}, Xtr, ytr, Xte, yte);
end

% Soft-margin polynomial p = 1..5, C in [0.1,0.6,1.1,2.1]
results.softPolyTrain = nan(numel(pSoft), numel(CSoft));
results.softPolyTest = nan(numel(pSoft), numel(CSoft));
for i = 1:numel(pSoft)
    for j = 1:numel(CSoft)
        [results.softPolyTrain(i, j), results.softPolyTest(i, j)] = evaluate_one(models.soft_poly{i, j}, Xtr, ytr, Xte, yte);
    end
end

save('task2_results.mat', 'results', 'pHard', 'pSoft', 'CSoft');

fprintf('\n=== Table 1 results (accuracy) ===\n');
fprintf('Hard linear: train=%.4f, test=%.4f\n', results.hardLinearTrain, results.hardLinearTest);

fprintf('\nHard polynomial (p=2..5)\n');
disp(table(pHard(:), results.hardPolyTrain, results.hardPolyTest, ...
    'VariableNames', {'p', 'TrainAcc', 'TestAcc'}));

fprintf('\nSoft polynomial (rows p=1..5, cols C=[0.1 0.6 1.1 2.1])\n');
fprintf('Train accuracy matrix:\n');
disp(results.softPolyTrain);
fprintf('Test accuracy matrix:\n');
disp(results.softPolyTest);

function [accTr, accTe] = evaluate_one(model, Xtr, ytr, Xte, yte)
if isempty(model)
    accTr = NaN;
    accTe = NaN;
    return;
end
predTr = svm_predict_labels(model, Xtr);
predTe = svm_predict_labels(model, Xte);
accTr = mean(predTr == ytr);
accTe = mean(predTe == yte);
end
