% Generate data from a mixture of linear regressions

function [y, X, B, pi] = mixtureOfLinearRegressions( N, K, D, sigma2 )
  B = rand( D, K );

  pi = ones( K, 1 ) / K;

  X = rand( N, D );
  y = zeros( N, 1 );

  % Choose the different distributions
  Pr = mnrnd( N, P );
  noffset = 1;
  for i = 1:K;
    n = Pr(i);
    y( noffset:noffset+n-1 ) = X( noffset:noffset+n-1, : ) * B(:,i);
    noffset = noffset + n;
  end;

  if sigma2 > 0.0
    y = y + randn(N, 1) * sigma2;
end
