% Recover the second moments, B3.
function sdpB3(path)
  % Load data
  y = load( strcat(path,'/y.txt'), '-ascii' );
  X = load( strcat(path,'/X.txt'), '-ascii' );
  sigma2 = load( strcat(path,'/sigma2.txt'), '-ascii' );
  lambda = load( strcat(path,'/lambda3.txt'), '-ascii' );

  B = recoverB3(y, X, sigma2, lambda);

  % Save as a vector because matlab cant write tensors to file.
  B = vec(B);

  save( strcat(path,'/B3.txt'), '-ascii', 'B' );
end

