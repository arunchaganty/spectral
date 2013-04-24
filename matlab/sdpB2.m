% Recover the second moments, B2.
function sdpB2(path)
  % Load data
  y = load( strcat(path,'/y.txt'), '-ascii' );
  X = load( strcat(path,'/X.txt'), '-ascii' );
  sigma2 = load( strcat(path,'/sigma2.txt'), '-ascii' );
  lambda = load( strcat(path,'/lambda2.txt'), '-ascii' );
  
  [N, d] = size( X );

  y = y.^2 - sigma2;
  % Construct the tensor form of X
  X_ = zeros( N, d.^2 );
  for n = 1:N
    Xn = X(n,:);
    X_(n,:) = vec(kron(Xn, Xn));
  end
  X = X_;

  cvx_begin sdp;
    variables B2(d, d);
    variables W1(d, d);
    variables W2(d, d);
    variables t;

    minimize ( lambda * t + 0.5/N * norm( y - X * vec(B2) ) );
    subject to 
      B2 == semidefinite( d );
      0.5 * trace(W1) + 0.5 * trace(W2) <= t;
      [W1, B2; B2', W2] == semidefinite(d + d);
  cvx_end;

  save( strcat(path,'/B2.txt'), '-ascii', 'B2' );
end


