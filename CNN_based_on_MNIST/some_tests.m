desc = importdata(fullfile(char(cd), 'data', 'descriptors', '3.dat'));
tdesc = importdata(fullfile(char(cd), 'data', 'testdesc', '3.dat'));

fl = 0;
ri = logical(1);
for i = 1:80
    for j = 1:120
       if tdesc(i) == desc(j)
           fl = 1;
           break;
       end
    end
    
    if fl ~= 1
        res(ri) = tdesc(i);
        ri = ri + 1;
    end
    fl = 0;
end

res = res'
fl