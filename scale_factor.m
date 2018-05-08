function [ scale_factor ] = scale_factor( data )
[data_dimension, data_length] = size(data);
scale_factor = ones(data_dimension,1);
for n = 1:data_dimension
	scale_factor(n, 1) = max(abs(data(n,:)));
end

end
