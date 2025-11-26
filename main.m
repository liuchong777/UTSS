close all
clear
clc
I0=imread('im_0072.png');
[Bmask]=tex_recog(I0);%UTSS分割
figure,imshow(im2uint8(Bmask))
