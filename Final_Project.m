clear
clc


cimds = imageDatastore('E:\greyimages\datast','IncludeSubfolders',true,'FileExtensions','.JPG','LabelSource','foldernames');
numClasses = numel(categories(cimds.Labels)); % tells about the classes(noncan,can)
labelCount = countEachLabel(cimds);% label inf and its count
fname = cimds.Files;
A=numel(fname);
Noise_names=cimds.Labels;
% Display some of the images in the datastore.
 figure('Name','fig:3 Display some of the NOISE images in the datastore')
perm = randperm(A,20);
for i = 1:20
    subplot(4,5,i);
    imshow(cimds.Files{perm(i)});
end
 
%SPLIT IMAGES FOR TRAIN AND TEST
[cimdsTrain,cimdsTest] = splitEachLabel(cimds,0.7, 'randomize');
 
% check the guassian and saltpepper count of train data
numClasses_cimdsTrain = numel(categories(cimdsTrain.Labels)); % tells about the classes(saltpepper,gaussian)
labelCount_cimdsTrain = countEachLabel(cimdsTrain);% label inf and its count
fname_cimdsTrain = cimdsTrain.Files;
A_cimdsTrain=numel(fname_cimdsTrain);
cancernames_cimdsTrain=cimdsTrain.Labels;
%check the guassian and saltpepper count of test data
numClasses_cimdsTest = numel(categories(cimdsTest.Labels)); % tells about the classes(noncan,can)
labelCount_cimdsTest = countEachLabel(cimdsTest);% label inf and its count
fname_cimdsTest = cimdsTest.Files;
A_cimdsTest=numel(fname_cimdsTest);
cancernames_cimdsTest=cimdsTest.Labels; 
 
%DATA AUGMENTATION
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
 
 
 
% augmentation of images
augcimds=augmentedImageDatastore([224 224 3],cimds,'DataAugmentation',imageAugmenter);
augimds_Train = augmentedImageDatastore([224 224  3],cimdsTrain,'DataAugmentation',imageAugmenter);
augimds_Test = augmentedImageDatastore([224 224  3],cimdsTest,'DataAugmentation',imageAugmenter);
 disp(augimds_Test)

%googlenet implemented to classify images
net = googlenet;

%analyzeNetwork displays an interactive plot of the network architecture and a table containing information about the network layers.
analyzeNetwork(net);    

%The first element of the Layers property of the network is the image input layer. For a GoogLeNet network, this layer requires input images of size 224-by-224-by-3, where 3 is the number of color channels. 
net.Layers(1);  
 if isa(net,'SeriesNetwork')
  lgraph = layerGraph(net.Layers);
else
  lgraph = layerGraph(net);
 end
% findLayersToReplace(lgraph) finds the single classification layer and the
% preceding learnable (fully connected or convolutional) layer of the layer
% graph lgraph.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
%[learnableLayer,classLayer]
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
 
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
 
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
 
%TRAINING OPTIONS
valFrequency = floor(numel(augimds_Train.Files)/10);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
 
 
Train_actual=cimdsTrain.Labels;
Test_actual=cimdsTest.Labels;
 
[myNet, info] = trainNetwork(augimds_Train,lgraph,options);
 
[NoisePred,score] = classify(myNet,augimds_Test);
accuracy = mean(NoisePred == cimdsTest.Labels);
noisenames = categories(Test_actual);
 
y_test=numel(NoisePred);
accuracy_test = mean(NoisePred == Test_actual);
numCorrect_test = nnz(NoisePred==Test_actual);


y_test1=numel(NoisePred);
accuracy_test1 = numCorrect_test /y_test;
confMat = confusionmat(cimdsTest.Labels,NoisePred);
 [cm_TEST ,order]=confusionmat(cimdsTest.Labels,NoisePred);
 


 %image filtering and enhancement
 folder = 'C:\Users\User\Desktop\matlab output';
 for i=1:numel(NoisePred)
     if strcmp(string(NoisePred(i)),'gaussian')
         img1=imread(fname_cimdsTest{i});
          imgd=img1;
          for layer=1:3
             imgd(:,:,layer) = wiener2(img1(:,:,layer), [5 5]);
          end
          shadow_lab = rgb2lab(imgd);
           max_luminosity = 100;
           L = shadow_lab(:,:,1)/max_luminosity;
           shadow_imadjust = shadow_lab;
           shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
           shadow_imadjust = lab2rgb(shadow_imadjust);

           shadow_histeq = shadow_lab;
           shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
           shadow_histeq = lab2rgb(shadow_histeq);

           shadow_adapthisteq = shadow_lab;
           shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
           shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
%             figure
%             montage({imgd,shadow_imadjust,shadow_histeq,shadow_adapthisteq},'Size',[1 4])
%             title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")
          baseFileName = sprintf('testimagesave%d.jpg', i);
          fullFileName = fullfile(folder, baseFileName);
          imgpair=imshowpair(shadow_adapthisteq,imgd,'montage');title(NoisePred(i));
          saveas(imgpair, fullFileName); % Save current axes (not the whole figure).
    elseif strcmp(string(NoisePred(i)),'salt&pepper')
         img1=imread(fname_cimdsTest{i});
          imgd=img1;
          for layer=1:3
%              imgd(:,:,layer) = medfilt2(img1(:,:,layer), [5 5]);
            imgd(:,:,layer) = wiener2(img1(:,:,layer), [5 5]);
            imgd(:,:,layer) = imbilatfilt(imgd(:,:,layer));
          end
          shadow_lab = rgb2lab(imgd);
           max_luminosity = 100;
           L = shadow_lab(:,:,1)/max_luminosity;
           shadow_imadjust = shadow_lab;
           shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
           shadow_imadjust = lab2rgb(shadow_imadjust);

           shadow_histeq = shadow_lab;
           shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
           shadow_histeq = lab2rgb(shadow_histeq);

           shadow_adapthisteq = shadow_lab;
           shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
           shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
%             figure
%             montage({imgd,shadow_imadjust,shadow_histeq,shadow_adapthisteq},'Size',[1 4])
%             title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")
          baseFileName = sprintf('testimagesave%d.jpg', i);
          fullFileName = fullfile(folder, baseFileName);
          imgpair=imshowpair(shadow_adapthisteq,imgd,'montage');title(NoisePred(i));
          saveas(imgpair, fullFileName); % Save current axes (not the whole figure).
     else
         img1=imread(fname_cimdsTest{i});
          imgd=img1;
          for layer=1:3
             imgd(:,:,layer) = wiener2(img1(:,:,layer), [5 5]);
          end
          shadow_lab = rgb2lab(imgd);
           max_luminosity = 100;
           L = shadow_lab(:,:,1)/max_luminosity;
           shadow_imadjust = shadow_lab;
           shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
           shadow_imadjust = lab2rgb(shadow_imadjust);

           shadow_histeq = shadow_lab;
           shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
           shadow_histeq = lab2rgb(shadow_histeq);

           shadow_adapthisteq = shadow_lab;
           shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
           shadow_adapthisteq = lab2rgb(shadow_adapthisteq);
%             figure
%             montage({imgd,shadow_imadjust,shadow_histeq,shadow_adapthisteq},'Size',[1 4])
%             title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")
          baseFileName = sprintf('testimagesave%d.jpg', i);
          fullFileName = fullfile(folder, baseFileName);
          imgpair=imshowpair(shadow_adapthisteq,imgd,'montage'),title(NoisePred(i));
          saveas(imgpair, fullFileName); % Save current axes (not the whole figure).
       end
        
%      else
%          img1=imread(fname_ImgdsTest{i});
%          imgd=im2double(img1);
%          image=medfilt2(imgd,[5 5]);
%          J = adapthisteq(image,'clipLimit',0.02,'Distribution','rayleigh');
%           baseFileName = sprintf('testimagesave%d.jpg', i);
%           fullFileName = fullfile(folder, baseFileName);
%           imgpair=imshowpair(img1,J,'montage'),title(NoisePred(i));
%           saveas(imgpair, fullFileName); % Save current axes (not the whole figure).
%         end
 end
 
 
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
 
if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end
 
% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');
 
% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);
 
if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);
 
 
% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end
 
end
