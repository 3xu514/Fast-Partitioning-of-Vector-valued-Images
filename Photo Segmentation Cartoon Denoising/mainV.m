clear all;
image = im2double(imread('pic_015.jpg'));
image_noise = image + 0.3*randn(size(image)); % Noise
imwrite(image_noise, 'pic_015_n.jpg');

%% ADMM4
tic
[error_admm4, sgmt_admm4] = ADMM4V(image_noise,0.5,2,0.005);
time_admm4 = toc;

%% ADMM8
tic
[error_admm8, sgmt_admm8] = ADMM8V(image_noise,0.3,2,0.003);
time_admm8 = toc;

%% TV
addpath('toolbox_optim/');
addpath('toolbox_optim/toolbox/');
sgmt_tv = zeros(size(image_noise));
tic;
for s = 1:3
    y = squeeze(image_noise(:,:,s));
    lambda = 0.3;

    K = @(x)grad(x);
    KS = @(x)-div(x);

    Amplitude = @(u)sqrt(sum(u.^2,3));
    F = @(u)lambda*sum(sum(Amplitude(u)));
    G = @(x)1/2*norm(y-x,'fro')^2;
    Normalize = @(u)u./repmat( max(Amplitude(u),1e-10), [1 1 2] );
    ProxF = @(u,tau)repmat( perform_soft_thresholding(Amplitude(u),lambda*tau), [1 1 2]).*Normalize(u);
    ProxFS = compute_dual_prox(ProxF);
    ProxG = @(x,tau)(x+tau*y)/(1+tau);
    options.report = @(x)G(x) + F(K(x));
    options.niter = 500;
    
    [xAdmm,~] = perform_admm(y, K,  KS, ProxFS, ProxG, options);
    sgmt_tv(:,:,s) = xAdmm;
end
time_tv = toc;

%% Graph Cut
addpath('GCMex-master/');
addpath('GCMex-master/example');
% compile_gc;
tic;
sgmt_gc = gc_example('pic_015_n.jpg',4);
time_gc = toc;

%% Plot
PSNR = @(orig, input) 10*log10(numel(orig)*max(max(max(abs(orig))))^2 / sum(sum(sum((orig-input).^2))));
figure;
subplot(242);
imagesc(image); axis image; 
title('Original Image');
subplot(243);
imagesc(image_noise); axis image;
title(['Noisy Image, PSNR = ', num2str(PSNR(image, image_noise))]);
subplot(245);
imagesc(sgmt_admm4); axis image; 
title(['ADMM4 Segmentation, PSNR = ', num2str(PSNR(image, sgmt_admm4))]);
xlabel(['Time: ', num2str(time_admm4), ' s']);
subplot(246);
imagesc(sgmt_admm8); axis image;
title(['ADMM8 Segmentation, PSNR = ', num2str(PSNR(image, sgmt_admm8))]);
xlabel(['Time: ', num2str(time_admm8), ' s']);
subplot(247);
imagesc(sgmt_tv); axis image; 
title(['Total Variation Segmentation, PSNR = ', num2str(PSNR(image, sgmt_tv))]);
xlabel(['Time: ', num2str(time_tv), ' s']);
subplot(248);
imagesc(sgmt_gc); axis image; 
title(['Graph Cut Segmentation, PSNR = ', num2str(PSNR(image, sgmt_gc))]);
xlabel(['Time: ', num2str(time_gc), ' s']);
