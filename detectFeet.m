%% carregar imagens depth
depth = imread("gait/gait_depth_oneimage.png");
color = imread("gait/gait_RGB_oneimage.png");

imagemFinal = detectFeetMain(rgb2gray(color),depth);