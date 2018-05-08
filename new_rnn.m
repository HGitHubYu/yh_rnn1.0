function [ net ] = new_rnn( hiddenNeurons, maxDelay, data_i, data_o )
% RNN with single output unit
%   Detailed explanation goes here
numDataset_i = length(data_i);
numDataset_o = length(data_o);
for n = 1:numDataset_i
    [data_i_dimension(n), data_i_length(n)] = size(data_i{n});
    [data_o_dimension(n), data_o_length(n)] = size(data_o{n});
end

net.numDataset_i = numDataset_i;
net.numDataset_o = numDataset_o;
net.data.i_dimension = data_i_dimension;
net.data.i_length = data_i_length;
net.data.o_dimension = data_o_dimension;
net.data.o_length = data_o_length;

net.maxDelay = maxDelay;
net.numInputUnits = data_i_dimension(1);
net.hiddenNeurons = hiddenNeurons;
net.numOutputUnits = data_o_dimension(1);
net.neurons = [net.numInputUnits, net.hiddenNeurons, net.numOutputUnits];
net.numAllUnits = net.numInputUnits + sum(hiddenNeurons) + net.numOutputUnits;
net.numHiddenLayers = length(hiddenNeurons);
net.numLayers = net.numHiddenLayers + 2;
net.paramVector = [];
% recurrency: output-to-hidden
for n = 1:maxDelay
    net.wr{n} = rand( net.numOutputUnits, net.hiddenNeurons(1) ); % recurrency weights
    net.paramVector = [ net.paramVector; net.wr{n}(:) ];
    for m = 1:numDataset_i
        net.er{n, m} = zeros( net.numOutputUnits, data_i_length(m) ); % recurrency derivative
    end
end
% feed-forward part
for n = 1: net.numHiddenLayers+1
    net.b{n} = rand( net.neurons(n+1), 1 ); % bias
    net.w{n} = rand( net.neurons(n), net.neurons(n+1) ); % weights
    temp = [ net.b{n}'; net.w{n} ];
    net.paramVector = [ net.paramVector; temp(:) ]; % store all w & b in one column vector
    for m = 1:numDataset_i
%         net.db{n} = zeros( net.neurons(n+1), 1 ); % derivative bias
%         net.dw{n} = zeros( net.neurons(n), net.neurons(n+1) ); % derivative weights
        net.y{n, m} = zeros( net.neurons(n+1), data_i_length(m) );
        net.e{n, m} = zeros( net.neurons(n+1), data_i_length(m) );
    end
end

    
net.numAllParam = length(net.paramVector);
net.h_lm = zeros(length(net.paramVector), 1); % for LM method
net.nonLM_iter = 10; % training iters before LM
net.mu = 100000; % for LM method: initial mu
net.mu_dec = 0.1;
net.mu_inc = 10;
net.mu_max = 1e30;
net.jaco = []; % Jacobian
net.f = []; % for LM method: fi = yi - Ai(x)
net.iter = 0;
net.iter_max = 300; % max number of iteration
net.lr = 0.1; % learning rate
net.momentum = 0.3; % momentum
net.actFcn = 'tansig'; % activation function
net.goal.rmse = 0.0001; % train goal, rmse
net.t_r.rmse = []; % training record, rmse of total datasets
net.t_r.datasets_rmse = []; % training record, rmse of each dataset
net.t_r.gMag = []; % training record, gradient magnitude
net.trainMethod = 'train_lm'; % training method

check_rules(net);
end


function [] = check_rules(net)

if net.data.i_length ~= net.data.o_length
    error('net.data.i_length ~= net.data.o_length');
end

end
