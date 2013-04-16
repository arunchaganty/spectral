function M = mode_unfold( T, i )
  [D1, D2, D3] = size(T);

  M = reshape(T, D1, D2 * D3);
end
