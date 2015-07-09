function [data, labels, name] = loadSets(dataPath, dataType)
%THIS TAKES PSD!
fprintf('     processing images (fft)... \n');
cd(dataPath);

files = dir(dataType);

data = [];
labels = [];
name = {};

for i=1:length(files)
    filename = files(i).name;
    name{i} = filename;
    labels = [labels, str2num(filename(1))];  %labels(i)  gives label for image i
    filePath = fullfile(dataPath, filename); 
    data = [data, imagePSD(filePath)]; %data(:, i) gives PSD of image i
    
end

