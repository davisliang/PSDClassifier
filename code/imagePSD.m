function [psd_reshaped] = imagePSD(imagePath)

a = imread(imagePath);     %we read the image from a path

if size(a,3) == 3           %if RGB
    b = rgb2gray(a);        %we turn the rgb image into greyscale
else
    b = a;
end

c = fft2(b);            %we take the 2D FFT

psd = fftshift(log(abs(c).^2)); %we find the shifted log abs squared, PSD
psd_resized = imresize(psd, [256,256]); %make sure its 256x256
psd_reshaped = reshape(psd_resized, 256*256, 1);    %reshaped to a vector

%figure;
%image(psd);
%title(imagePath);



