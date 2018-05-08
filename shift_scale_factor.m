function [ shift_center_factor, scale_factor ] = shift_scale_factor( data )
[data_dimension, data_length] = size(data);
shift_center_factor = mean(data, 2);
shifted_data = data - shift_center_factor;
scale_factor = ones(data_dimension,1);
for n = 1:data_dimension
	scale_factor(n, 1) = max(abs(shifted_data(n,:)));
end

end
