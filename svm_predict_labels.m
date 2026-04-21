function [pred, g] = svm_predict_labels(model, Xq)
%SVM_PREDICT_LABELS Predict labels {-1,+1} using trained dual SVM model.

Xq = double(Xq);

switch lower(model.kernelType)
    case 'linear'
        K = model.X' * Xq;
    case 'poly'
        K = (model.X' * Xq + 1) .^ model.p;
    otherwise
        error('Unsupported kernelType: %s', model.kernelType);
end

g = (model.alpha .* model.y)' * K + model.b;
pred = ones(1, size(Xq, 2));
pred(g < 0) = -1;
pred = pred(:);
end
