% Recover the second moments, B3.
function B = regB3(y, X, sigma2)
  % Load data
  [N, d] = size( X );
  avgBetas = X\y; % y_i = x_i^T \beta
  y_ = y.^3 - 3 * sigma2 * (X * avgBetas);

  D = d * (d+1) * (d+2) / 6;

  % Construct the tensor form of X
  X_ = zeros( N, D );
  for n = 1:N
    Xn = X(n,:);
    X_(n,:) = cvec( tensor([d,d,d], 'unit', Xn ) );
  end

  b = X_\y_;
  B = uncvec(b,d);
end

