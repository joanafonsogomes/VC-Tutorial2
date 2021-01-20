%carregar imagens depth
%depthArray= ones(141,141,60,'uint16');
RGB = VideoReader("gait/gait_groundtruth_RGB_60frames.avi");

v = VideoWriter('newfile.avi','Motion JPEG AVI');
v.Quality = 95;
v.FrameRate = 30;
open(v);

for i=1:59
    frame = read(RGB,i);
    depth=imread("gait\gait_depth\gait_depth_60frames_" +(i-1)+".png");
    detectFeetMain(rgb2gray(frame),depth,v);
end

close(v);

