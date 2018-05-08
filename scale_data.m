function [ data_scaled ] = scale_data( data, scale_factor )
[data_dimension, data_length] = size(data);
if data_dimension ~= length(scale_factor)
    error('data_dimension ~= length(scale_factor)');
end
data_scaled = NaN(data_dimension, data_length);
for n = 1:data_dimension
	data_scaled(n,:) = data(n,:)/scale_factor(n, 1);
end

end
