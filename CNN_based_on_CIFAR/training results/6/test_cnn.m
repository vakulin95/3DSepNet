function test_cnn(upboundepoch)
% testing of the cnn
run(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));

imdir = fullfile('D:\MEGA\Programs\MatConvNet\Data Bases\MyBase\Princeton v.2\');
klassdir = fullfile(char(cd), 'data', 'testdesc\');
expdir = fullfile(char(cd), 'RESULT\');

%upboundepoch = 10; 

for epochNum = 8 : upboundepoch %number of launched epoch

    epochName = sprintf('net-epoch-%d.mat', epochNum);
    netdir = load(fullfile(char(cd), 'data', 'imdbF', epochName));

    net = netdir.net;
    net.layers{end}.type = 'softmax';
    net = vl_simplenn_tidy(net) ;

    num_of_classes = 2; %number of classes

    result = cell(50, num_of_classes);

    for cl_i = 1 : num_of_classes

            result(1, cl_i) = net.meta.classes.name(cl_i);

            cl = cl_i;
            klass = sprintf('%d.dat', cl);
            imn = importdata(fullfile(klassdir, klass));

            for i = 1 : 80

                imfile = sprintf('m%d.bmp', imn(i));

                im = imread([imdir, imfile],'bmp');
                im = imresize(im, [32 32]) ;

                im_ = single(im);
                im_ = imresize(im_, [32 32]) ;

                res = vl_simplenn(net, im_) ;

                scores = squeeze(gather(res(end).x)) ;
                [bestScore, best] = max(scores) ;
                result(i+1, cl_i) = net.meta.classes.name(best); 

                clear im;
                clear imfile;
                clear scores;
                clear bestScore;
                clear best;
            end        
            clear cl;
            clear klass;
            clear imn;
    end
    
    expfn = sprintf('net_testing_result_ep-%d.dat', epochNum);
    fileID = fopen(fullfile(expdir, expfn),'w');
    fprintf('%s: Writing result on epoch %d.\n', mfilename, epochNum)
    
    formatSpec = '%20s | %20s \n';
    fprintf(fileID, formatSpec, result{1,:});
    fprintf(fileID, '\n');

    [nrows, ncols] = size(result);
    for row = 2:nrows
        fprintf(fileID, formatSpec, result{row,:});
    end

    fclose(fileID);

end

end

