clear;
close all;
clc;

[data.i, data.o] = load_data();

hiddenNeurons = [ 15 ]; % vector: multilayer
maxDelay = 2;
[ net ] = new_rnn( hiddenNeurons, maxDelay, data.i, data.o );
net = train_rnn(net, data.i, data.o);
net = sim_rnn(net, data.i);
data.sim = net.y{ net.numHiddenLayers+1, 1 };

%% plot
figure();
plot(net.t_r.rmse); % t_r.rmse

figure();
plot(data.o{1}, 'r-', 'Linewidth', 0.5);
hold on;
plot(data.sim, 'k--', 'Linewidth', 0.5);
fs = 14;
% xlabel('Normalized phase', 'FontSize', fs);
% ylabel('Output (V)', 'FontSize', fs);
% set(gca,'ylim',[-0.2 1.6], 'ytick',[0:0.5:1.5] );
set(gca,'FontSize',fs);
