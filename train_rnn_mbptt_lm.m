function [ net ] = train_rnn_mbptt_lm( net, data_i, data_o )
%LM
%   Detailed explanation goes here
info.numDataset_i = length(data_i);
info.numDataset_o = length(data_o);
for n = 1: info.numDataset_i
    [info.data_i_dimension(n), info.data_i_length(n)] = size(data_i{n});
    [info.data_o_dimension(n), info.data_o_length(n)] = size(data_o{n});
end
total_data_o_length = sum(info.data_o_length);
fprintf('total_data_o_length: %d\n\n', total_data_o_length);

[ net ] = sim_rnn( net, data_i );
[ net ] = lm_get_jaco_f( net, info, data_i, data_o );
for m = 1:info.numDataset_i
    se(m,1) = sum((( net.y{end, m} - data_o{m} ).^2 ));
end
epoch_datasets_rmse = sqrt(se./(info.data_o_length'));
net.t_r.datasets_rmse = epoch_datasets_rmse;
rmse = sqrt(sum(se) / total_data_o_length);
net.t_r.rmse = [ net.t_r.rmse, rmse ];

iter = 0;
epoch = 0;
while ( 1 )
    stop1 = iter >= net.iter_max;
    stop2 = rmse < net.goal.rmse;
    stop = stop1 || stop2;
    if stop
        fprintf('iter: %d\n', iter);
        fprintf('epoch: %d\n', epoch);
        [ dataset_rmse, datasetN ] = max(epoch_datasets_rmse);
        fprintf('max individual dataset rmse: %f   of   dataset %d\n', dataset_rmse, datasetN);
        fprintf('total rmse: %f\n\n', rmse);
        break;
    end
    iter = iter + 1;
    
    if iter <= net.nonLM_iter
        h_lm = ( net.mu * eye(net.numAllParam, net.numAllParam)) \ ( - net.jaco' * net.f );
    else
        h_lm = (net.jaco' * net.jaco + net.mu * eye(net.numAllParam, net.numAllParam)) \ ( - net.jaco' * net.f );
    end
    net2 = net;
    net2.paramVector = net2.paramVector + h_lm;
    net2 = paramVector2wb_rnn( net2 );
%     assignin('base','net2',net2);
    [ net2 ] = sim_rnn( net2, data_i );
    for m = 1:info.numDataset_i
        se2(m,1) = sum((( net2.y{end, m} - data_o{m} ).^2 ));
    end
    rmse2 = sqrt(sum(se2) / total_data_o_length);
    if rmse2 < rmse
        epoch = epoch + 1;
        net = net2;
        rmse = rmse2;
        net.t_r.rmse = [ net.t_r.rmse, rmse ];
        epoch_datasets_rmse = sqrt(se2./(info.data_o_length'));
        net.t_r.datasets_rmse = [ net.t_r.datasets_rmse, epoch_datasets_rmse ];
        [ net ] = lm_get_jaco_f( net, info, data_i, data_o );
        net.mu = net.mu * net.mu_dec;
    else
        net.mu = net.mu * net.mu_inc;
        if net.mu > net.mu_max
            net.mu = net.mu_max;
        end
%         fprintf('net.mu: %f\n\n', net.mu);
    end
    
%     if mod( iter, 10 )==1
    if 1
        fprintf('iter: %d\n', iter);
        fprintf('epoch: %d\n', epoch);
        fprintf('mu: %e\n', net.mu);
        [ dataset_rmse, datasetN ] = max(epoch_datasets_rmse);
        fprintf('max individual dataset rmse: %f   of   dataset %d\n', dataset_rmse, datasetN);
        fprintf('total rmse: %f\n\n', rmse);
    end
    
end
end


