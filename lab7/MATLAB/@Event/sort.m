function obj = sort(obj)

assert(isvector(obj));
N = length(obj);
t = nan(size(obj));
for i = 1:N
    t(i) = obj(i).time;
end
[~, idx] = sort(t);
obj = obj(idx);
