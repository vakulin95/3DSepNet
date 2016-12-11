function net = cnn_init()

lr = [.1 2] ;
net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(3,3,3,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1) ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(3,3,32,32, 'single'), zeros(1,32,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; 

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(3,3,32,64, 'single'), zeros(1,64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; 

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(4,4,64,64, 'single'), zeros(1,64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,64,3, 'single'), zeros(1,3,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'softmaxloss') ;

net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = 0.05*ones(1,10);
net.meta.trainOpts.weightDecay = 0.0001;
net.meta.trainOpts.batchSize = 3;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

net = vl_simplenn_tidy(net) ;

end