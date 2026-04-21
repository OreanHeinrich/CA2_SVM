# CA2_SVM

MATLAB implementation for EE5904/ME5404 Project 1 (SVM spam classification).

## Files
- `task1_compute_models.m`: trains SVMs required in Task 1 and saves `task1_models.mat`.
- `task2_evaluate_table1.m`: evaluates Task 1 SVMs on `train.mat` and `test.mat`, prints Table 1 accuracies, and saves `task2_results.mat`.
- `task3_eval_predict.m`: trains a designed SVM and outputs `eval_predicted` from `eval_data`.
- `svm_train_dual.m`: shared dual-QP SVM training function (uses `quadprog`).
- `svm_predict_labels.m`: shared SVM prediction function.

## Run order
1. In MATLAB, set current folder to this repository root.
2. Run Task 1:
   ```matlab
   task1_compute_models
   ```
3. Run Task 2:
   ```matlab
   task2_evaluate_table1
   ```
4. For Task 3 assessment (when `eval_data` is already in workspace), run:
   ```matlab
   task3_eval_predict
   ```
   The script outputs `eval_predicted` (`N x 1`, values in `{-1, +1}`).
