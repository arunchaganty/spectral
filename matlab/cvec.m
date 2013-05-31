% Recover the second moments, B3.
function v = cvec(T)
  [d,d,d] = size(T);
  D = d * (d+1) * (d+2)/6;

  v = zeros(1,D);
  idx = 1;
  for d1 = 1:d
    for d2 = 1:d1
      for d3 = 1:d2
        if (d1 == d2 && d2 == d3 )
          multiplicity = 1;
        elseif (d1 == d2 || d2 == d3 || d1 == d3 )
          multiplicity = 3;
        else
          multiplicity = 6;
        end
        v( idx ) = multiplicity * T(d1,d2,d3);
        idx = idx + 1;
      end
    end
  end
end

