function model = svm_train_dual(X, y, kernelType, p, C)
%SVM_TRAIN_DUAL Train binary SVM from dual QP using quadprog.
%   X: d x n features (column-wise samples)
%   y: n x 1 labels in {-1,+1}
%   kernelType: 'linear' or 'poly'
%   p: polynomial order (ignored for linear)
%   C: [] for hard margin, scalar > 0 for soft margin

if nargin < 5
    C = [];
end

X = double(X);
y = double(y(:));
ALPHA_TOLERANCE = 1e-6;
[~, nSamples] = size(X);

if ~all(ismember(unique(y), [-1, 1]))
    error('Labels must be -1 or +1.');
end

switch lower(kernelType)
    case 'linear'
        K = X' * X;
        kernelParam = 1;
    case 'poly'
        K = (X' * X + 1) .^ p;
        kernelParam = p;
    otherwise
        error('Unsupported kernelType: %s', kernelType);
end

H = (y * y') .* K;
H = (H + H') / 2 + 1e-10 * eye(nSamples);
f = -ones(nSamples, 1);
Aeq = y';
beq = 0;
lb = zeros(nSamples, 1);

if isempty(C)
    ub = [];
else
    ub = C * ones(nSamples, 1);
end

opts = optimoptions('quadprog', 'Display', 'off');
[alpha, ~, exitflag] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);

if exitflag <= 0 || isempty(alpha)
    error(['QP failed (exitflag=%d). Check kernel/data compatibility, ', ...
        'hard-margin feasibility, or numerical conditioning.'], exitflag);
end

svMask = alpha > ALPHA_TOLERANCE;
if ~any(svMask)
    error('No support vectors found.');
end

if isempty(C)
    bMask = svMask;
else
    bMask = alpha > ALPHA_TOLERANCE & alpha < (C - ALPHA_TOLERANCE);
    if ~any(bMask)
        bMask = svMask;
    end
end

idx = find(bMask);
Kbias = K(:, idx);
decNoBias = (alpha .* y)' * Kbias;
b = mean(y(idx)' - decNoBias);

model = struct();
model.X = X;
model.y = y;
model.alpha = alpha;
model.b = b;
model.kernelType = lower(kernelType);
model.p = kernelParam;
model.C = C;
model.svMask = svMask;
end
