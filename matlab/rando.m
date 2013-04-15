% Random orthogonal matrix

function Q = rando( D )
  Z = randn( D, D );
  [Q,R] = qr(Z);
  d = diag(R);
  ph = d ./ abs(d);
  Q = Q .* repmat(ph, 1, D);
endfunction

