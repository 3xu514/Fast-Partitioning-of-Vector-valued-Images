image = im2double(imread('church.jpg'));

image_noise = image + laprnd(size(image,1),size(image,2),0,0.1);
S_noise = ssim(image_noise,image);

%% L2 data term
tic
[error_admm8, sgmt_admm8] = ADMM8VN2(image_noise,0.5,0.005,0.00125);
time_admm8 = toc;
S_admm8 = ssim(sgmt_admm8,image);
%% L1 data term
tic
[error_admm8_l1, sgmt_admm8_l1] = ADMM8VN1(image_noise,3.5,0.035,0.00875);%Mu=mu/S
time_admm8_l1 = toc;
S_admm8_l1 = ssim(sgmt_admm8_l1,image);
%% Plot
PSNR = @(orig, input) 10*log10(numel(orig)*max(max(max(abs(orig))))^2 / sum(sum(sum((orig-input).^2))));
figure;
subplot(221);
imagesc(image); axis image; axis off
title('Original Image');
subplot(222);
imagesc(image_noise); axis image; axis off;
title('Image with Laplacian Noise');
subplot(223)
imagesc(sgmt_admm8); axis image; axis off;
title(['Restoration Using L2 Data Term. MSSIM = ', num2str(S_admm8)]);
 subplot(224);
imagesc(sgmt_admm8_l1); axis image; axis off;
title(['Restoration Using L1 Data Term. MSSIM = ',num2str(S_admm8_l1)]);
