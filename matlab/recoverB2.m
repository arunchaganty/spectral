% Recover the second moments, B2.
function [B2] = recoverB2( y, X, sigma2, lambda )
  [N, d] = size( X );
  y = y.^2 - sigma2;

  % Construct the tensor form of X
  X_ = zeros( N, d.^2 );
  for n = 1:N
    Xn = X(n,:);
    X_(n,:) = vec(kron(Xn, Xn));
  end

  cvx_begin sdp;
    variables B2(d, d);
    variables W1(d, d);
    variables W2(d, d);
    variables t;

    minimize ( lambda * t + 0.5/N * norm( y - X_ * vec(B2) ) );
    subject to 
      0.5 * trace(W1) + 0.5 * trace(W2) <= t;
      [W1, B2; B2', W2] == semidefinite(d + d);
  cvx_end;

end

