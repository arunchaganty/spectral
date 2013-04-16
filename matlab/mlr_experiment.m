function mlr_experiment( N, K, D, sigma2, lambda )
  for i = 1:1
    [y, X, B, pi] = generateMLR( N, K, D, sigma2 );
    [N, d] = size(X);
    B2 = B * diag( pi ) * B';
    B3 = tensor( [d,d,d], 'decomp', B', pi );

    % Save data 
    save( 'data/y.txt', '-ascii', 'y' );
    save( 'data/X.txt', '-ascii', 'X' );
    save( 'data/sigma2.txt', '-ascii', 'sigma2' );
    save( 'data/lambda2.txt', '-ascii', 'lambda' );
    save( 'data/lambda3.txt', '-ascii', 'lambda' );

    % Solve
    sdpB2('data');
    sdpB3('data');

    B2_ = load('data/B2.txt', '-ascii');
    B3_ = load('data/B3.txt', '-ascii');
    B3_ = reshape( B3_, size(B3) );

    fprintf( '%.3f %.3f\n', norm( B2 - B2_ ), norm( vec( B3 - B3_ ) ) );
  end
end

