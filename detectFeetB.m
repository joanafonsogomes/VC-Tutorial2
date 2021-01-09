%% carregar imagens depth
%depthArray= ones(141,141,60,'uint16');
i=50;

RGB = VideoReader("gait_60frames/gait_RGB_60frames.avi");
frame = read(RGB,i);
depth=imread("gait_60frames\gait_depth\gait_depth_60frames_" +(i-1)+".png");

detectFeetMain(rgb2gray(frame),depth);

%{
figure;
for i=0:59
    image=imread("gait_60frames\gait_depth\gait_depth_60frames_" + i+".png");
    imgSize = size(image);
    depthCrop = depth(imgSize(1)/4+50 :imgSize(1)*3/4-50,imgSize(2)*3/8+30 :imgSize(2)*5/8+10);
    %depthArray(:,:,i+1)= depthCrop;

    %detectFeetMain(mat2gray(image),depthCrop);
end
%}
%% tamanho a imagem
%imgSize = size(depth);
%% cortar a imagem ao necessário
%depthCrop = depth(imgSize(1)/4+50 :imgSize(1)*3/4-50,imgSize(2)*3/8+30 :imgSize(2)*5/8+10);


%imagemFinal = detectFeetMain(rgb2gray(color),depthCrop);