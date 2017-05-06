function test_cnn(epoches)
% testing of the cnn
run(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));

imdir = fullfile('D:\files\SI_MC_db_dat\');
klassdir = fullfile(char(cd), 'data', 'testdesc');
expdir = fullfile(char(cd), 'RESULT');

%upboundepoch = 10; 

for epochNum = epoches(1) : epoches(2) %number of launched epoch

    epochName = sprintf('net-epoch-%d.mat', epochNum);
    netdir = load(fullfile(char(cd), 'data', 'export', epochName));

    net = netdir.net;
    net.layers{end}.type = 'softmax';
    net = vl_simplenn_tidy(net) ;

    num_of_classes = 3; %number of classes

    result = cell(80, num_of_classes);

    for cl_i = 1 : num_of_classes

            result(1, cl_i) = net.meta.classes.name(cl_i);

            cl = cl_i;
            klass = sprintf('%d.dat', cl);
            imn = importdata(fullfile(klassdir, klass));

            for i = 1 : 20

                imfile = sprintf('m%d.dat', imn(i));

                im = importdata(fullfile(imdir, imfile));
                
                maxel = max(max(im));
                M = int16((im(:,:) / maxel) * 255);
                
                im_ = single(M);

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
    fprintf('Writing result on epoch %d.\n', epochNum);
    
    formatSpec = '%20s | %20s | %20s\n';
    fprintf(fileID, formatSpec, result{1,:});
    fprintf(fileID, '\n');

    [nrows, ncols] = size(result);
    for row = 2:nrows
        fprintf(fileID, formatSpec, result{row,:});
    end

    fclose(fileID);

end

end