clear all;

image = im2double(imread('church.jpg'));

sigma = 0.2;
image_noise = image + sigma * randn(size(image));

[m,n,l] = size(image);
missingFraction = 0.6;
weights = rand(m,n) > missingFraction;
image_noise(~cat(3, weights, weights, weights)) = 0;
% image_noise = image;

%%
tic
[error_admm4, sgmt_admm4] = ADMM4V(image_noise,0.4,2,0.004);
time_admm4 = toc;


%% Plot
PSNR = @(orig, input) 10*log10(numel(orig)*max(max(max(abs(orig))))^2 / sum(sum(sum((orig-input).^2))));
figure;
subplot(221);
imagesc(image); axis image;axis off;
title('Original Image');
subplot(222);
imagesc(image_noise); axis image;axis off;
title('Missing Image');
subplot(223);
imagesc(sgmt_admm88);
title(['Our Code']); axis image;axis off;
% Just a preview, get the output and send it to Storath's Code