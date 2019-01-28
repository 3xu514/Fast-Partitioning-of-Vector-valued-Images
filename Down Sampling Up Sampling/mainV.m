clear
im = im2double(imread('295087.jpg'));
subplot(131)
imagesc(im)
axis image
axis off
title('Original')


tic
u = ADMM4Vd(im,1.55,2,0.01*1.55);
t1 = toc;
u=round(u,3);
c1=cost(im,u,2);
subplot(133)
imagesc(u)
axis image
axis off
title(['ADMM4 Down-Sampling time:',num2str(t1)])


tic
u = ADMM4V(im,1.55,2,0.0155);
t2 = toc;
u=round(u,3);
c2=cost(im,u,1.55);
subplot(132)
imagesc(u)
axis image
axis off
title(['ADMM4 time:',num2str(t2)])

