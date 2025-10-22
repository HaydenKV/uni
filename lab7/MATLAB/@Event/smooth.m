function obj = smooth(obj)

assert(isvector(obj));

for k = length(obj)-1:-1:1
    assert(obj(k).saveSystemState, 'Require system state to be saved to run smoother');
    obj(k).system = obj(k).system.smoothFromNext(obj(k + 1).system);
end
