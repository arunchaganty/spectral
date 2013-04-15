%% Jacobian of mixture models

% Compute the Jacobian by formula.
function J=jacobian( M, pi ) 
  [D, K] = size(M);

  % Compute the moments
  M2 = M * diag( pi ) * M';
  M3 = zeros( D, D, D );
  for a = [1:K]
    M3(a,:,:) = pi(a) * M(:,a) * M(:,a)';
  endfor

  J = zeros( numel(M) + numel(pi), numel(M2) + numel(M3) );

  % Derivatives with respect to M
  for a = [1:K]
    for d = [1:D]
      theta = (a-1)*D + d;
      % dM2/d\mu_{ad} = \pi_a (\delta_{id} \mu_{aj} + \delta_{jd}
          % \mu_{ai} )
      dM2 = zeros(D,D);
      dM2(d,:) = dM2(d,:) + M(:,a)';
      dM2(:,d) = dM2(:,d) + M(:,a);
      J( theta, 1:numel(M2) ) = pi(a) * dM2(:);

      % dM3/d\mu_{ad} = \pi_a (\delta_{id} \mu_{aj} \mu_{ak} + \delta_{jd}
          % \mu_{ai} \mu_{ak} + \delta_{kd} \mu_{aj} \mu_{ak} )
      dM3 = zeros(D,D,D);
      dM3(d,:,:) = dM3(d,:,:) + M(:,a)' * M(:,a);
      dM3(:,d,:) = dM3(:,d,:) + M(:,a)' * M(:,a);
      dM3(:,:,d) = dM3(:,:,d) + M(:,a)' * M(:,a);
      J( theta, numel(M2)+1:numel(M2)+numel(M3) ) = pi(a) * dM3(:);
    endfor
    theta = K*D + a;
    % dM2/d\pi_{a} = \mu_a \mu_a'
    dM2 = M(:,a) * M(:,a)';
    J( theta, 1:numel(M2) ) = dM2(:);
    dM3 = tensor( M(:,a), M(:,a), M(:,a) );
    J( theta, numel(M2)+1 : numel(M2)+numel(M3) ) = pi(a) * dM3(:);
  endfor
endfunction

function T=tensor(x1,x2,x3) 
  [D,ans] = size(x1);
  T = zeros(D,D,D);
  M = x2 * x3';
  for d = 1:D
    T(d,:,:) = x1(d) * M;
  endfor
endfunction 

