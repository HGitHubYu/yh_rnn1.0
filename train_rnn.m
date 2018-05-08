function [ net ] = train_rnn( net, data_i, data_o )
%UNTITLED2 Summary of this function goes here
%   t_r: training record

switch net.trainMethod
    case 'train_bgd'
        [ net ] = train_bgd( net, data_i, data_o );
    case 'train_bgdm'
        [ net ] = train_bgdm( net, data_i, data_o );
    case 'train_lm'
        [ net ] = train_rnn_mbptt_lm( net, data_i, data_o );
    otherwise
        [ net ] = train_bgd( net, data_i, data_o );
end

end

