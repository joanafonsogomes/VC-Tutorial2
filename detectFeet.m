%% carregar imagens depth
depth = imread("gait_oneimage/gait_depth_oneimage.png");
figure;imshow(mat2gray(depth));
%{
depthArray= ones(480,640,60,'uint16');
for i=0:59
    image=imread("gait_60frames\gait_depth\gait_depth_60frames_" + i+".png");
    depthArray(:,:,i+1)= image;
end
%}
%% tamanho a imagem
imgSize = size(depth);
%% cortar a imagem ao necessário
depthCrop = depth(imgSize(1)/4+50 :imgSize(1)*3/4-50,imgSize(2)*3/8+30 :imgSize(2)*5/8+10);
figure;imshow(mat2gray(depthCrop));

imagemFinal = detectFeetMain(depthCrop);