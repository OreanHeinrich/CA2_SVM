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
ALPHA_TOLERANCE = 1e-6;  % treat smaller alpha values as numerical zero
[~, nSamples] = size(X);

if ~all(ismember(unique(y), [-1, 1]))
    error('Labels must be -1 or +1.');
end

% Feature standardization (per feature) to improve numerical conditioning.
normMu = mean(X, 2);
normSigma = std(X, 0, 2);
normSigma(normSigma < 1e-12) = 1;
Xn = bsxfun(@rdivide, bsxfun(@minus, X, normMu), normSigma);

switch lower(kernelType)
    case 'linear'
        K = Xn' * Xn;
        kernelParam = 1;
        kernelScale = 1;
    case 'poly'
        dotProd = Xn' * Xn;
        dotDiagMean = mean(diag(dotProd));
        if ~isfinite(dotDiagMean) || dotDiagMean <= 0
            warning('Polynomial kernel scale fallback triggered; using scale=1.');
            dotDiagMean = 1;
        end
        kernelScale = dotDiagMean;
        K = (dotProd / kernelScale + 1) .^ p;
        kernelParam = p;
    otherwise
        error('Unsupported kernelType: %s', kernelType);
end

% Build Hessian and enforce symmetry.
Hbase = (y * y') .* K;
Hbase = (Hbase + Hbase') / 2;
f = -ones(nSamples, 1);
Aeq = y';
beq = 0;
lb = zeros(nSamples, 1);

if isempty(C)
    ub = [];
else
    ub = C * ones(nSamples, 1);
end

opts = optimoptions('quadprog', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point-convex', ...
    'MaxIterations', 2000, ...
    'ConstraintTolerance', 1e-8, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12);

diagMean = mean(diag(Hbase));
if ~isfinite(diagMean) || diagMean <= 0
    warning('Hessian diagonal scale fallback triggered; using scale=1.');
    diagMean = 1;
end
ridgeCandidates = diagMean * [1e-10, 1e-8, 1e-6, 1e-4];
ridgeCandidates = max(ridgeCandidates, 1e-12);  % keep minimum ridge to avoid near-singular H

alpha = [];
exitflag = -1;
selectedRidge = NaN;
for r = 1:numel(ridgeCandidates)
    H = Hbase + ridgeCandidates(r) * eye(nSamples);
    [alphaTry, ~, exitflagTry] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], opts);
    if exitflagTry > 0 && ~isempty(alphaTry)
        alpha = alphaTry;
        exitflag = exitflagTry;
        selectedRidge = ridgeCandidates(r);
        break;
    end
    exitflag = exitflagTry;
end

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
model.X = Xn;
model.y = y;
model.alpha = alpha;
model.b = b;
model.kernelType = lower(kernelType);
model.p = kernelParam;
model.C = C;
model.svMask = svMask;
model.normMu = normMu;
model.normSigma = normSigma;
model.kernelScale = kernelScale;
model.ridgeUsed = selectedRidge;  % records successful ridge level used by quadprog
end
