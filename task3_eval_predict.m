% Task 3: design one SVM and output eval_predicted for eval_data.
% This file must output a variable named eval_predicted (N x 1).

if ~exist('eval_data', 'var')
    if isfile('eval.mat')
        load eval.mat;
        if ~exist('eval_data', 'var')
            error('eval.mat exists but does not contain variable eval_data.');
        end
    else
        error('eval_data is not in workspace and eval.mat is not found.');
    end
end

if ~exist('train_data', 'var') || ~exist('train_label', 'var')
    load train.mat;
end

Xtr = double(train_data);
ytr = double(train_label(:));
Xev = double(eval_data);

% Designed model: soft-margin polynomial SVM
% (chosen for good performance and robust feasibility)
p = 3;
C = 2.1;
model = svm_train_dual(Xtr, ytr, 'poly', p, C);

eval_predicted = svm_predict_labels(model, Xev);
