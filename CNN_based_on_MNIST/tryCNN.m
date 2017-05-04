function tryCNN()

    run(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));

    % obtain and preprocess an image
    epochName = sprintf('net-epoch-20.mat');
    netdir = load(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'data', 'mnist-baseline-simplenn', epochName));

    net = netdir.net;
    net.layers{end}.type = 'softmax';
    net = vl_simplenn_tidy(net) ;

    impath = fullfile(char(cd), 'data', '0.png');
    im = imread(impath) ;
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, [28 28]) ;
    %im_ = im_ - net.normalization.averageImage ;

    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    
    figure(1) ; clf ; image(im) ;
    title(sprintf('%d, score %.3f',...
      best, bestScore)) ;

end