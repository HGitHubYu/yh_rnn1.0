function [ net ] = paramVector2wb_rnn( net )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
count = 0;
for d = 1 : net.maxDelay
    r = size( net.wr{d}, 1 );
    c = size( net.wr{d}, 2 );
    temp = reshape( net.paramVector( count+1 : count+r*c ), [r, c] );
    net.wr{d} = temp;
    count = count + r*c;
end
for n = 1: net.numHiddenLayers+1
    r = size( net.w{n}, 1 ) + 1;
    c = size( net.w{n}, 2 );
    temp = reshape( net.paramVector( count+1 : count+r*c ), [r, c] );
    net.b{n} = temp( 1, : )';
    net.w{n} = temp( 2:end, : );
    count = count + r*c;
end

end