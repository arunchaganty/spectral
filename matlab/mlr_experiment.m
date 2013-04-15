function mlr_experiment( N, K, D, M, sigma2 )
  for i = 1:100
    [y, X, B, P] = generateMLR( N, K, D, sigma2 );
    B2 = B * diag( P ) * B';

    [B2_, sigma2_] = recoverB2( y_, X_, sigma2 );

    X2 = X' * X / N;

    fprintf( '%.3f %.3f %.3f\n', norm( B2 - B2_ ), cond( X2 ), sigma2 );
  end
end

