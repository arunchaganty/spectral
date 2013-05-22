% Recover the second moments, B3.
function B = recoverB3( y, X, sigma2, lambda )
  [N, d] = size( X );
  avgBetas = X\y; % y_i = x_i^T \beta
  y = y.^3 - 3 * sigma2 * (X * avgBetas);

  % Construct the tensor form of X
  X_ = zeros( N, d.^3 );
  for n = 1:N
    Xn = X(n,:);
    X_(n,:) = vec( tensor([d,d,d], 'unit', Xn ) );
  end
  X = X_;

  cvx_begin sdp;
    variables B(d, d, d);
    variables W11(d, d);
    variables W12(d.^2, d.^2);
    variables W21(d, d);
    variables W22(d.^2, d.^2);
    variables W31(d, d);
    variables W32(d.^2, d.^2);

    variables t1;
    variables t2;
    variables t3;

    %minimize ( 0.5/N * norm( y - X_ * vec(B) ) );
    %subject to
    %  B == shiftdim( B, 1 );
    %  B == shiftdim( B, 2 );

    minimize ( lambda * (t1 + t2 + t3)/3 + 0.5/N * norm( y - X_ * vec(B) ) );
    subject to
      0.5 * trace(W11) + 0.5 * trace(W12) <= t1;
      [W11, mode_unfold(B,1); mode_unfold(B,1)', W12] == semidefinite(d.^2 + d);
      0.5 * trace(W21) + 0.5 * trace(W22) <= t2;
      [W21, mode_unfold(B,2); mode_unfold(B,2)', W22] == semidefinite(d.^2 + d);
      0.5 * trace(W31) + 0.5 * trace(W32) <= t3;
      [W31, mode_unfold(B,3); mode_unfold(B,3)', W32] == semidefinite(d.^2 + d);
  cvx_end;
end

