% Recover the second moments, B2.
function B = recoverB2a( y, X, sigma2, lambda )
  [N, d] = size( X );

  % Construct the variables of the problem.
  y = y.^2 - sigma2;
  % Construct the covariance matrix
  S = zeros( d.^2, d.^2 );
  for n = 1:N
    Xn = X(n,:) * X(n,:)'; 
    x = vec(Xn); % Construct the 'x' matrix.
    S = S + (x*x' - S)/(n+1); % Increment the covariance
  end
  % Do SVD to get the actual parts
  [V,Sigma,~] = svd(S);
  V = V * sqrt(Sigma);

  % Get the U matrix
  U = zeros( d, d );

  for n = 1:N
    Xn = X(n,:) * X(n,:)'; 
    yn = y(n);
    U = (yn * Xn - U)/(n+1);
  end

  % Construct
  cvx_begin sdp;
    variables B(d, d);
    variables W1(d, d);
    variables W2(d, d);
    variables s;
    variables t;

    minimize ( s + lambda * t );
    subject to 
      B == semidefinite(d);
      [eye(d.^2) , V * vec(B); (V * vec(B))', s + 2 * trace(U * B)] == semidefinite( d.^2 + 1 );
      0.5 * trace(W1) + 0.5 * trace(W2) <= t;
      [W1, B; B', W2] == semidefinite(2*d);
  cvx_end;
end

