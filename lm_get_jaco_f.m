function [ net ] = lm_get_jaco_f( net, info, data_i, data_o )
%LM
% data_i{m}, data_o{m}
% initialize
net.jaco = [];
net.f = [];
y = net.y;
for n = 1:net.maxDelay
    for m = 1: info.numDataset_i
        er{n, m} = zeros( net.numOutputUnits, info.data_i_length(m) ); % recurrency derivative
    end
end
for n = 1: net.numHiddenLayers+1
    for m = 1:info.numDataset_i
        e{n, m} = zeros( net.neurons(n+1), info.data_i_length(m) );
    end
end
% calculate f and jaco
for m = 1: info.numDataset_i
    % f
    f = ( data_o{m} - y{end, m} )';
    net.f = [ net.f; f];
    
    jaco = zeros(net.numAllParam, info.data_i_length(m));
    %%%%%%%%% back propagation: MBPTT
    % net.e is differet in LM method. 
    % It's the derivatives regarding the output (net.f), not the squared error
    % bp without recurrency
    e{end, m} = ones( 1, info.data_o_length(m) );
    for n = net.numHiddenLayers : -1 : 1
        e{n,m} = ( net.w{n+1} * e{n+1,m} ).*( 1 - (y{n,m}).^2 ); % derivative tansig: 1-act.^2
    end
    for n = 1:net.maxDelay
        er{n, m} = net.wr{n} * e{1, m};
    end
    %%%%%%%%%%%%%%%% stamp jaco
    % first maxDelay steps
    for s = 1 : net.maxDelay
        deriv = []; % ffnn part derivative
        for d = 1:s-1
            deriv = [ deriv; y{end,m}(1,s-d) * e{1,m}(:,s)]; 
        end
        for d = s : net.maxDelay
            deriv = [ deriv; 0 * e{1,m}(:,s)]; 
        end
        for u = 1: net.neurons(2)
            deriv = [ deriv; e{1,m}(u,s)];
            deriv = [ deriv; data_i{m}(:,s) * e{1,m}(u,s)];
        end
        for lyr = 2 : net.numHiddenLayers+1
            for u = 1 : net.neurons(lyr+1)
                deriv = [ deriv; e{lyr,m}(u,s)];
                deriv = [ deriv; y{lyr-1,m}(:,s) * e{lyr,m}(u,s)];
            end
        end
        % + recurrency part
        jaco_1step = deriv;
        for dd = 1:s-1
            jaco_1step = jaco_1step + jaco(:,s-dd) * er{dd,m}(1,s);
        end
        jaco(:,s) = jaco_1step;
    end
    % rest steps
    for s = net.maxDelay+1 : info.data_i_length(m)
        deriv = []; % ffnn part derivative
        for d = 1:net.maxDelay
            deriv = [ deriv; y{end,m}(1,s-d) * e{1,m}(:,s)]; 
        end
        for u = 1: net.neurons(2)
            deriv = [ deriv; e{1,m}(u,s)];
            deriv = [ deriv; data_i{m}(:,s) * e{1,m}(u,s)];
        end
        for lyr = 2 : net.numHiddenLayers+1
            for u = 1 : net.neurons(lyr+1)
                deriv = [ deriv; e{lyr,m}(u,s)];
                deriv = [ deriv; y{lyr-1,m}(:,s) * e{lyr,m}(u,s)];
            end
        end
        % + recurrency part
        jaco_1step = deriv;
        for dd = 1:net.maxDelay
            jaco_1step = jaco_1step + jaco(:,s-dd) * er{dd,m}(1,s);
        end
        jaco(:,s) = jaco_1step;
    end
    net.jaco = [net.jaco, jaco];
end

net.er = er;
net.e = e;
net.jaco = -net.jaco';

end