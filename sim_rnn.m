function [ net ] = sim_rnn( net, data_i )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

switch net.actFcn
    case 'tansig'
        [ net ] = sim_rnn_tansig( net, data_i );
    case 'logsig'
        [ net ] = sim_ffnn_logsig( net, data_i );
    otherwise
        fprintf('no activation selected\n');
end

end


function [ net ] = sim_rnn_tansig( net, data_i )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
numDataset_i = length(data_i);
for m = 1:numDataset_i
    [data_i_dimension(m), data_i_length(m)] = size(data_i{m});
end

hN1 = net.hiddenNeurons(1);
for m = 1 : numDataset_i
    for n = 1: net.numHiddenLayers+1
        net.y{n, m} = zeros( net.neurons(n+1), data_i_length(m) );
    end
    % first maxDelay steps
    for s = 1:net.maxDelay
        inSum = zeros( hN1 ,1);
        for d = 1: s-1
            inSum = inSum + net.wr{d}' * net.y{end, m}(:, s-d);
        end
        inSum = inSum + net.w{1}' * data_i{m}(:,s) + net.b{1};
        net.y{1, m}(:,s) = tansig(inSum);
        for n = 2: net.numHiddenLayers
            net.y{n, m}(:,s) = tansig( net.w{n}' * net.y{n-1, m}(:,s) + net.b{n} );
        end
        net.y{net.numHiddenLayers+1, m}(:,s) = net.w{net.numHiddenLayers+1}' * ...
            net.y{net.numHiddenLayers, m}(:,s) + net.b{net.numHiddenLayers+1};
    end
    % rest steps
    for s = net.maxDelay+1 : data_i_length(m)
        inSum = zeros( hN1 ,1);
        for d = 1: net.maxDelay
            inSum = inSum + net.wr{d}' * net.y{end, m}(:, s-d);
        end
        inSum = inSum + net.w{1}' * data_i{m}(:,s) + net.b{1};
        net.y{1, m}(:,s) = tansig(inSum);
        for n = 2: net.numHiddenLayers
            net.y{n, m}(:,s) = tansig( net.w{n}' * net.y{n-1, m}(:,s) + net.b{n} );
        end
        net.y{net.numHiddenLayers+1, m}(:,s) = net.w{net.numHiddenLayers+1}' * ...
            net.y{net.numHiddenLayers, m}(:,s) + net.b{net.numHiddenLayers+1};
    end
end

end


function [ net ] = sim_ffnn_logsig( net, data_i )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
net.y{1} = logsig( net.w{1}' * data_i + net.b{1} );
for n = 2: net.numHiddenLayers
    net.y{n} = logsig( net.w{n}' * net.y{n-1} + net.b{n} );
end
net.y{net.numHiddenLayers+1} = net.w{net.numHiddenLayers+1}' * net.y{net.numHiddenLayers} + net.b{net.numHiddenLayers+1};      
end