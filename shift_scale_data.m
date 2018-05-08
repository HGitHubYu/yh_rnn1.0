function [ data_shifted_scaled ] = shift_scale_data( data, shift_center_factor, scale_factor )
[data_dimension, data_length] = size(data);
if data_dimension ~= length(shift_center_factor)
    error('data_dimension ~= length(shift_center_factor)');
end
if data_dimension ~= length(scale_factor)
    error('data_dimension ~= length(scale_factor)');
end
data_shifted = data - shift_center_factor;
data_shifted_scaled = NaN(data_dimension, data_length);
for n = 1:data_dimension
	data_shifted_scaled(n,:) = data_shifted(n,:)/scale_factor(n, 1);
end

end
