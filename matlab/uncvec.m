% Recover the second moments, B3.
function B = uncvec(v, d)
  B = zeros(d,d,d);
  D = d * (d+1) * (d+2)/6;

  idx = 1;
  for d1 = 1:d
    for d2 = 1:d1
      for d3 = 1:d2
        B(d1,d2,d3) = v(idx);
        B(d1,d3,d2) = v(idx);
        B(d2,d1,d3) = v(idx);
        B(d2,d3,d1) = v(idx);
        B(d3,d2,d1) = v(idx);
        B(d3,d1,d2) = v(idx);
        idx = idx + 1;
      end
    end
  end
end

