function [obj, system] = process(obj, system)
arguments
    obj (1, 1) Event
    system (1, 1) SystemBase
end

if obj.verbosity > 0
    fprintf(1, '[t=%07.3fs] %s', obj.time, obj.getProcessString());
end

% Time update
system = system.predict(obj.time);

% Event implementation
[obj, system] = obj.update(system);

if obj.saveSystemState
    obj.system = system;   % Copy system state (System has value semantics since it doesn't inherit from handle)
end

if obj.verbosity > 0
    fprintf(1, ' done\n');
end
