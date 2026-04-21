% Task 1: compute discriminant functions g(.) for required SVMs.

clear;
load train.mat;  % train_data is features x samples (57 x N), labels are N x 1

X = double(train_data);
y = double(train_label(:));
if size(X, 1) ~= 57
    error('Expected train_data to be 57xN (features x samples).');
end

pHard = 2:5;
pSoft = 1:5;
CSoft = [0.1, 0.6, 1.1, 2.1];

models = struct();

% (i) Hard-margin with linear kernel
try
    models.hard_linear = svm_train_dual(X, y, 'linear', 1, []);
catch ME
    models.hard_linear = [];
    warning('Hard linear failed: %s', ME.message);
end

% (ii) Hard-margin with polynomial kernel
models.hard_poly = cell(numel(pHard), 1);
for i = 1:numel(pHard)
    try
        models.hard_poly{i} = svm_train_dual(X, y, 'poly', pHard(i), []);
    catch ME
        models.hard_poly{i} = [];
        warning('Hard poly p=%d failed: %s', pHard(i), ME.message);
    end
end

% (iii) Soft-margin with polynomial kernel
models.soft_poly = cell(numel(pSoft), numel(CSoft));
for i = 1:numel(pSoft)
    for j = 1:numel(CSoft)
        try
            models.soft_poly{i, j} = svm_train_dual(X, y, 'poly', pSoft(i), CSoft(j));
        catch ME
            models.soft_poly{i, j} = [];
            warning('Soft poly p=%d, C=%.1f failed: %s', pSoft(i), CSoft(j), ME.message);
        end
    end
end

save('task1_models.mat', 'models', 'pHard', 'pSoft', 'CSoft');
disp('Saved Task 1 models to task1_models.mat');
