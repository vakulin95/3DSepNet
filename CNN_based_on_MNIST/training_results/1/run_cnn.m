function [net, info] = run_cnn(varargin)
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

net = init_cnn('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%fix it
%net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:3,'UniformOutput',false) ;
net.meta.classes.name = imdb.meta.classes(1,:);

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

%error file writing
fileID = fopen(fullfile(char(cd), 'RESULT', 'net_error_result.dat'),'w');

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
test_cnn([15; 20]);

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
num_si_per_cl = 160;
num_of_si = num_si_per_cl * num_of_classes * 4;

SI = int16(zeros(28, 28, num_of_si));
LABELS = zeros(1, num_of_si);
si_num = logical(1);
for iclass = 1:num_of_classes
    
    klass = sprintf('%d.dat', iclass);
    imn = importdata(fullfile(opts.descPath, klass));

    idx = randperm(numel(imn));  %images in rand order
    imn(1:end) = imn(idx);
    
    for idat = 1:num_si_per_cl
        fname = sprintf('m%d.dat', imn(idat));
        %fname = sprintf('%d.dat', mod(idat, 7) + 1);
        M = zeros(28, 28);
        r = fullfile(opts.dataDir, fname);
        M = importdata(fullfile(opts.dataDir, fname));
        %si_num = (iclass - 1) * num_si_per_cl + idat; 
        %SI(:, :, si_num) = M;
        %LABELS(1, si_num) = iclass;
        
        maxel = max(max(M));
        M_ = int16((M(:,:) / maxel) * 255);
        SI(:, :, si_num) = M_;
        LABELS(1, si_num) = iclass;
        si_num = si_num + 1;
        
        M1 = rot90(rot90(M_));
        SI(:, :, si_num) = M1;
        LABELS(1, si_num) = iclass;
        si_num = si_num + 1;
        
        M2 = -1 * (M_(:,:) - 255);
        SI(:, :, si_num) = M2;
        LABELS(1, si_num) = iclass;
        si_num = si_num + 1;
        
        M3 = rot90(rot90(M2));
        SI(:, :, si_num) = M3;
        LABELS(1, si_num) = iclass;
        si_num = si_num + 1;
    end
end

%����� �������� �������� ������� �� ��������� � ���������� �� �����������
idb = randperm(numel(LABELS()));
LABELS(1:end) = LABELS(idb);
SI(:,:,1:end) = SI(:,:,idb);

sitn = int16(num_of_si / 6);
%countVal(LABELS, num_of_si, sitn)
set = [ones(1, num_of_si - sitn) 3*ones(1,sitn)];
data = single(reshape(SI,28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = LABELS;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;

kln = importdata(fullfile(opts.descPath, 'meta.dat'))';
imdb.meta.classes = kln;


function res = countVal(LABELS, num_of_si, sitn)
    
    res = zeros(1, 3);    
    for i = (num_of_si - sitn):num_of_si
        switch LABELS(i)
            case 1
                res(1, 1) = res(1, 1) + 1;
            case 2
                res(1, 2) = res(1, 2) + 1;
            case 3
                res(1, 3) = res(1, 3) + 1;
        end
    end