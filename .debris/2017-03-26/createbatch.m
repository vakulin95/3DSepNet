function batch = createbatch()

run(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));

metadir = fullfile(vl_rootnn, 'MyNet', 'data', 'descriptors\');
descdir = fullfile('D:\MEGA\Programs\MatConvNet\Data Bases\MyBase\Princeton v.1\'); %directory with images
outdir = fullfile(vl_rootnn, 'MyNet', 'data', 'batches\');

batch.data = [];
batch.labels = [];

num_of_classes = 3;
num_Per_Batch = 20; %number of images from the one class per one batch
%num_of_Batch = 0; %from 0

for num_of_Batch = 0 : 5
    for cl_i = 1 : num_of_classes

        cl = cl_i;
        klass = sprintf('%d.dat', cl);
        imn = importdata(fullfile(metadir, klass));
        
        idx = randperm(numel(imn));  %images in rand order
        imn(1:end) = imn(idx);

        begin_ind = num_of_Batch*num_Per_Batch + 1;
        end_ind = (num_of_Batch + 1)*num_Per_Batch;
        indinmatfile = 1;
        for i = begin_ind : end_ind
                     
                imfile = sprintf('m%d.bmp', imn(i));

            im = imread([descdir, imfile],'bmp');
            im = imresize(im, [32 32]) ;
            temp(:, :, 1) = reshape(im(:, :, 1)', 1, 1024);
            temp(:, :, 2) = reshape(im(:, :, 2)', 1, 1024);
            temp(:, :, 3) = reshape(im(:, :, 3)', 1, 1024);

            temp = reshape(temp, 1, 3072);

            str_num = (cl_i - 1)*num_Per_Batch + indinmatfile;
            batch.data(str_num,:) = temp; 
            batch.labels(str_num, 1) = cl_i - 1;
            
            indinmatfile = indinmatfile + 1;

            clear temp; 
            clear im;
            clear imfile;
            clear str_num;
        end
        
        clear cl;
        clear klass;
        clear imn;
        clear idx;
        clear begin_ind;
        clear end_ind;
        clear indinmatfile;
    end

    idb = randperm(numel(batch.labels()));
    batch.labels(1:end) = batch.labels(idb);
    batch.data(1:end, :) = batch.data(idb, :)
    
    if (num_of_Batch == 5)
        batch.batchlabel = 'testing batch 1 of 1';
        save(fullfile(outdir, 'test_batch'), '-struct', 'batch');
    else
        batch.batchlabel = sprintf('training batch %d of 5', num_of_Batch + 1);
        batch_name = sprintf('data_batch_%d', num_of_Batch+1);
        save(fullfile(outdir, batch_name), '-struct', 'batch');
    end
    
    clear idb;
    clear batch;
    clear batch_name;
end

batchesMeta(metadir, outdir)

end

function batches = batchesMeta(path, out)

batches.label_names = importdata(fullfile(path, 'meta.dat'));
save(fullfile(out, 'batches.meta.mat'), '-struct', 'batches');

end