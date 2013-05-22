% Recover the second moments, B2.
function sdpB2(path)
  % Load data
  y = load( strcat(path,'/y.txt'), '-ascii' );
  X = load( strcat(path,'/X.txt'), '-ascii' );
  sigma2 = load( strcat(path,'/sigma2.txt'), '-ascii' );
  lambda = load( strcat(path,'/lambda2.txt'), '-ascii' );

  B2 = recoverB2(y, X, sigma2, lambda);

  save( strcat(path,'/B2.txt'), '-ascii', 'B2' );
end


