function [net, info] = cnn_run(varargin)
%CNN_MNIST  Demonstrates MatConvNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(char(cd), 'data', 'export') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('D:\files\SI_MC_db_dat\') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.descPath = fullfile(char(cd), 'data', 'descriptors');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [1];

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%fix it
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:3,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted

num_of_classes = 3;
num_si_per_cl = 120;
num_of_si = num_si_per_cl * num_of_classes;

SI = zeros(28, 28, num_of_si);
LABELS = zeros(1, num_of_si);
for iclass = 1:num_of_classes
    
    klass = sprintf('%d.dat', iclass);
    imn = importdata(fullfile(opts.descPath, klass));

    idx = randperm(numel(imn));  %images in rand order
    imn(1:end) = imn(idx);
    
    for idat = 1:num_si_per_cl
        %fname = sprintf('%d.dat', imn(idat));
        fname = '1.dat';
        M = zeros(28, 28);
        M = importdata(fullfile(opts.dataDir, fname));
        si_num = (iclass - 1) * num_si_per_cl + idat; 
        SI(:, :, si_num) = M;
        LABELS(1, si_num) = iclass;
    end
end

idb = randperm(numel(LABELS()));
LABELS(1:end) = LABELS(idb);
SI(:,:,1:end) = SI(:,:,idb)

sitn = int16(num_of_si / 6);
set = [ones(1,num_of_si - sitn) 3*ones(1,sitn)];
data = single(reshape(SI,28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
% %fix labels---------------------------------------
% y1 = y1 - ones(size(y1));
% y2 = y2 - ones(size(y2));
%-------------------------------------------------
imdb.images.labels = LABELS;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:3,'uniformoutput',false) ;
