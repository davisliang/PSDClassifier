function [] = runPSDClassifier(dataset_path)

%% example: dataset_path = /Users/Davis/Desktop/PSDClassifier/datatester
dataType = '*jpg';

trainPath = fullfile(dataset_path, 'train');
testPath = fullfile(dataset_path, 'test');
validPath = fullfile(dataset_path, 'valid');

savePath = '/Users/Davis/Desktop/PSDClassifier/params/param.mat';

%% upload datasets and labels
fprintf('loading training set... \n');
[trainData, trainLabels] = loadSets(trainPath, dataType); %takes PSD
fprintf('loading testing set... \n');
[validData, validLabels] = loadSets(validPath, dataType);
fprintf('loading testing set... \n');
[testData, testLabels] = loadSets(testPath, dataType);

%% normalize data
maxnum = max(max(trainData));

trainData = 2*((trainData/maxnum) - 0.5);   %data from -1 to 1 now.
validData = 2*((validData/maxnum) - 0.5);
testData = 2*((testData/maxnum) - 0.5);



%% run trainer (validation and testing are done inside training)
numIter = 50;
numhid = 100;
[whi, woh] = PSDClassifierTrain(trainData, trainLabels, validData, ...
    validLabels, testData, testLabels, numIter, numhid);


save(savePath, 'whi','woh', 'maxnum');


