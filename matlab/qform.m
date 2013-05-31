function z = qform( X , M , Z )
  [N, ~] = size(X);
  z = zeros(N,1);
  for i = 1:N
    z(i) = X(i,:) * M * Z(i,:)';
  end;

end
