function [net, info] = run_cnn()
%run the data prepairing, launching the model and train cnn

%run(fullfile(matlabroot, 'bin', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));
run(fullfile('D:\MEGA\Programs\MatConvNet', 'matconvnet-1.0-beta22', 'matlab', 'vl_setupnn.m'));
RESdir = fullfile(char(cd), 'RESULT\');

opts.expDir = fullfile(vl_rootnn, 'MyNet', 'data', 'imdbF') ;
opts.dataDir = fullfile(vl_rootnn, 'MyNet', 'data') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
%opts.train = struct() ;
opts.train.gpus = [1];

%model of cnn
net = cnn_init_1();

%data
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
net.meta.classes.name = imdb.meta.classes(:)' ;

%train
[net, info] = cnn_train(net, imdb, getBatch(opts), ...
                        'expDir', opts.expDir, ...
                        net.meta.trainOpts, ...
                        opts.train, ...
                        'val', find(imdb.images.set == 3)) ;

%error file writing
fileID = fopen(fullfile(RESdir, 'net_error_result.dat'),'w');

formatSpec = '%f %f \n';

fprintf(fileID, '# err\n# train   val\n');
%fprintf(fileID, '\n');

result = {info.train.objective; info.val.objective}';

[nrows, ncols] = size(result);
for row = 1:nrows
    fprintf(fileID, formatSpec, result{row,:});
end

fclose(fileID);

%result writing
test_cnn(net.meta.trainOpts.numEpochs);

function fn = getBatch(opts)
 fn = @(x,y) getSimpleNNBatch(x,y);
 
function [images, labels] = getSimpleNNBatch(imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

function imdb = getCifarImdb(opts)

unpackPath = fullfile(opts.dataDir, 'batches');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi});
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],360) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],360) ;
  W = z(:,set == 1)*z(:,set == 1)'/360 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;