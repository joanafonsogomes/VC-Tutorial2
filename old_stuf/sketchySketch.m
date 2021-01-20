
img = imread("gait_oneimage/gait_RGB_oneimage.png");
depth = imread("gait_oneimage/gait_depth_oneimage.png");
imgSize = size(img);
depthArray= ones(480,640,60,'uint16');
for i=0:59
    image=imread("gait_60frames\gait_depth\gait_depth_60frames_" + i+".png");
    depthArray(:,:,i+1)= image;
end


grayimg = rgb2gray(img);
gaussKernel=fspecial('gaussian',imgSize(1)/4,1);
smooth=imfilter(grayimg,gaussKernel);
[thresh,~]= graythresh(grayimg);
threshedImg = imbinarize(grayimg,thresh);

[dt,~]= graythresh(depth);
threshedDepth = imbinarize(imfilter(depth,gaussKernel),dt);
se=strel('disk',15,4);
%se=strel('line',5,90);
closed=~imclose(~threshedDepth,se);


normalDepth=imclose(mat2gray(imfilter(depth,gaussKernel)),se);
normalDepth=normalDepth(imgSize(1)/4 :imgSize(1)*3/4,imgSize(2)*3/8 :imgSize(2)*5/8);

normalDepth=mat2gray(normalDepth);
figure
subplot(2,2,1),imshow(threshedImg)
subplot(2,2,2),imshow(threshedDepth)
subplot(2,2,3) ,imshow(normalDepth)
subplot(2,2,4) ,imshow(closed)
%}
%%%%%% DANGER : VERY SLOW
figure

for i=0:59
normalDepth=imclose(mat2gray(imfilter(depthArray(:,:,i+1),gaussKernel)),se);
normalDepth=normalDepth(imgSize(1)/4 :imgSize(1)*3/4,imgSize(2)*3/8 :imgSize(2)*5/8);
subplot(6,10,i+1),imshow(mat2gray(normalDepth))
end


