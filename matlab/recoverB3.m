% Recover the second moments, B2.
function B3 = recoverB3( y, X, sigma2, lambda )
  [N, d] = size( X );
  y3 = y.^3;

  cvx_begin quiet;
    variables B3(d, d, d);

    minimize ( lambda * trace(mode_unfold(B3,0)) + 0.5 * norm( y2 - diag( X * B3 * X' ) - sigma2 ) );
  cvx_end;
end

