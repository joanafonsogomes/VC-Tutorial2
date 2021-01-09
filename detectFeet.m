%% carregar imagens depth
depth = imread("gait_oneimage/gait_depth_oneimage.png");
color = imread("gait_oneimage/gait_RGB_oneimage.png");

imagemFinal = detectFeetMain(rgb2gray(color),depth);