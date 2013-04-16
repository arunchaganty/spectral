% Recover the second moments, B2.
function [B2] = recoverB2( y, X, sigma2, lambda )
  [N, d] = size( X );
  y = y.^2 - sigma2;

  cvx_begin sdp quiet;
    variables B2(d, d);
    variables W1(d, d);
    variables W2(d, d);
    variables t;

    minimize ( lambda * t + 0.5/N * norm( y - diag( X * B2 * X' ) ) );
    subject to 
      B2 == semidefinite( d );
      0.5 * trace(W1) + 0.5 * trace(W2) <= t;
      [W1, B2; B2', W2] == semidefinite(2*d);
      sigma2 >= 0;
  cvx_end;
end

