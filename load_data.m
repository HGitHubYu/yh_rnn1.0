function [ data_i, data_o ] = load_data(  )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

a = rand(1, 500);
b = [a;
    0, a(1:end-1);
    0,0,a(1:end-2)];
c(1) = func( b(1,1), b(2,1), b(3,1), 0, 0 );
c(2) = func( b(1,2), b(2,2), b(3,2), c(1), 0 );
for n = 3:length(a)
    c(n) = func( b(1,n), b(2,n), b(3,n), c(n-1), c(n-2) );
end
data_i{1} = b;
data_o{1} = c;

end

function [ y ] = func( x1, x2, x3, x4, x5 )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

y = x1^2 + 0.6*x2 - 0.3*exp(x3) + 1/(1+exp(x4)) - 1.3*x1*x5 - 0.7*x2*x3*x4;

end


