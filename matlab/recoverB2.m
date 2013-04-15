% Recover the second moments, B2.
function [B2, sigma2] = recoverB2( y, X, sigma2 )
  [N, d] = size( X );
  y2 = y.^2;

  cvx_begin quiet;
    variables B2(d, d);
    variables W1(d, d);
    variables W2(d, d);
%    variables sigma2;
    variables t;

    minimize ( 1e-3 * t + 0.5 * norm( y2 - diag( X * B2 * X' ) - sigma2 ) );
    subject to 
      B2 == semidefinite( d );
      0.5 * trace(W1) + 0.5 * trace(W2) <= t;
      [W1, B2; B2', W2] == semidefinite(2*d);
      sigma2 >= 0;
  cvx_end;
end

