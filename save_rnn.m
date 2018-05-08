function [  ] = save_rnn( net, filename, mode )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% mode
% 1: basic
% 2: save w, b, t_r

switch mode
    case 1
        save(filename, 'net');
    case 2
        net.er = {};
        net.y = {};
        net.e = {};
        net.jaco = [];
        net.f = [];
        save(filename, 'net');
    otherwise
        save(filename, 'net');
end

  
end