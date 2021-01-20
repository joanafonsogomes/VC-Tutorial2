function [imagemFinal] = detectFeetMainB(original,depth,v)
imgSize = size(depth);
depthCrop = depth(imgSize(1)/4 :imgSize(1)*3/4,imgSize(2)*3/8 :imgSize(2)*5/8);
newImage = removeBackground(depthCrop);
newImage = controloMorfologico(newImage);

fig = figure('visible', 'off');
imshow(original); hold on;
imgSize = size(original);
padding = [imgSize(1)/4, imgSize(2)*3/8];


[B,~,~] = bwboundaries(newImage);
left = B{1} + padding;
plot(left(:,2), left(:,1), 'g','LineWidth',1);
right = B{2} + padding;
plot(right(:,2), right(:,1), 'r','LineWidth',1);

[~,indice] = min(left(:,1));
ponta_esquerda = left(indice,:);
plot(ponta_esquerda(2),ponta_esquerda(1),'s');

[~,indice] = max(left(:,1));
tornozelo_esquerdo = left(indice,:);
plot(tornozelo_esquerdo(2),tornozelo_esquerdo(1),'s');

[~,indice] = min(right(:,1));
ponta_direita = right(indice,:);
plot(ponta_direita(2),ponta_direita(1),'s');

[~,indice] = max(right(:,1));
tornozelo_direito = right(indice,:);
plot(tornozelo_direito(2),tornozelo_direito(1),'s');
writeVideo(v,getframe(fig));
close(fig);


imagemFinal = newImage;
end


function noBackground = removeBackground(depthCrop)
[sizeCropX,sizeCropY] = size(depthCrop);
for i = 0:20
    delta(i+1) = mean(depthCrop(sizeCropX-(i+1),:))- mean(depthCrop(sizeCropX-i,:));
end

deltaD = mean(delta);
newImage = depthCrop;
for i=1:sizeCropX
    k = sizeCropX - i +1;
    dmax = (i)*deltaD + mean(depthCrop(sizeCropX,:));
    dmin = dmax - 200;
    for j = 1:sizeCropY
        if(newImage(k,j)>=dmax-10 || newImage(k,j)<dmin )
            newImage(k,j)=0;
        end
    end
    deltaD=deltaD + 0.003;
end

noBackground = newImage;

end


function extracao = controloMorfologico(newImage)

%erode para apagar as linhas no canto inferior direito
se=strel('disk',2,6);
newImage = imerode(newImage, se);


se=strel('disk',8,0);
newImage=imopen(newImage,se);


se=strel('disk',2,6);
%se=strel('line',5,90);
newImage=imclose(newImage,se);
%
se=strel('disk',2,6);
newImage=imdilate(newImage,se);

extracao = newImage;
end




