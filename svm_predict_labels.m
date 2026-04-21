function [pred, g] = svm_predict_labels(model, Xq)
%SVM_PREDICT_LABELS Predict labels {-1,+1} using trained dual SVM model.

Xq = double(Xq);
if isfield(model, 'normMu') && isfield(model, 'normSigma')
    Xq = bsxfun(@rdivide, bsxfun(@minus, Xq, model.normMu), model.normSigma);
end

switch lower(model.kernelType)
    case 'linear'
        K = model.X' * Xq;
    case 'poly'
        dotProd = model.X' * Xq;
        if isfield(model, 'kernelScale') && isfinite(model.kernelScale) && model.kernelScale > 0
            dotProd = dotProd / model.kernelScale;
        end
        K = (dotProd + 1) .^ model.p;
    otherwise
        error('Unsupported kernelType: %s', model.kernelType);
end

g = (model.alpha .* model.y)' * K + model.b;
pred = ones(size(Xq, 2), 1);
pred(g(:) < 0) = -1;
end
