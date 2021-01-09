function [imagemFinal] = detectFeetMain(depthCrop)
%%
[sizeCropX,sizeCropY] = size(depthCrop);
for i = 0:5
    delta(i+1) = mean(depthCrop(sizeCropX-(i+1),:))- mean(depthCrop(sizeCropX-i,:));
end

deltaD = mean(delta);
newImage = depthCrop;
mean(depthCrop(sizeCropX-1,:))
for i=1:sizeCropX
    k = sizeCropX - i +1;
    dmax = (i)*deltaD + mean(depthCrop(sizeCropX,:));
    dmin = dmax - 200;
    for j = 1:sizeCropY
        if(newImage(k,j)>=dmax-10 || newImage(k,j)<dmin )
            newImage(k,j)=0;
        end
    end

end
figure;
imshow(mat2gray(newImage));

newImage = controloMorfologico(newImage);

figure;
imshow(newImage); hold on;
[B,L,N] = bwboundaries(newImage);
left = B{1}
plot(left(:,2), left(:,1), 'g','LineWidth',1);
right = B{2};
plot(right(:,2), right(:,1), 'r','LineWidth',1);


imagemFinal = newImage;
end


function extracao = controloMorfologico(newImage)

%erode para apagar as linhas no canto inferior direito
se=strel('disk',2,6);
%se = strel('square', 5)
newImage = imerode(newImage, se);
figure;
imshow(mat2gray(newImage));

se=strel('disk',15,0);
%se=strel('line',5,90);
newImage=imclose(newImage,se);
figure;
imshow(mat2gray(newImage));
%%
se=strel('disk',2,6);
newImage=imdilate(newImage,se);
figure;
imshow(mat2gray(newImage));
%%
se=strel('disk',2,0);
%se = strel('square', 5)
newImage = imerode(newImage, se);
newImage = imerode(newImage, se);
newImage = imerode(newImage, se);
newImage = imerode(newImage, se);
newImage=imdilate(newImage,se);
newImage=imdilate(newImage,se);
newImage=imdilate(newImage,se);
newImage=imdilate(newImage,se);
figure;
imshow(mat2gray(newImage));
extracao = newImage;
end




