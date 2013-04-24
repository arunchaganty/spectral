
function T = tensor( D, type, x, pi )
  %% Make a tensor with dimensions D
  %% Type: 
  %% 'unit': construct x \otimes x \otimes x
  %% 'decomp': construct pi_i x_i \otimes x_i \otimes x_i

  if strcmpi( type, 'unit')
    T = 1;
    for i = 1:length(D)
      T = kron( x, T );
    end
    T = reshape( T, D );
  elseif strcmpi( type, 'decomp')
    T = zeros( D );
    for i = 1:length(pi)
      Xi = x(i, :);
      T = T + pi(i) * tensor( D, 'unit', Xi);
    end
  end
end
