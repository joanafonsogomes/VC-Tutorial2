%carregar imagens depth
%depthArray= ones(141,141,60,'uint16');
i=40;

RGB = VideoReader("gait/gait_groundtruth_RGB_60frames.avi");
frame = read(RGB,i);
depth=imread("gait\gait_depth\gait_depth_60frames_" +(i-1)+".png");

detectFeetMainB(rgb2gray(frame),depth);

