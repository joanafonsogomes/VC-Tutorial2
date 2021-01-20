%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% WARNING : Ficheiro de testes temporário %%%%%%%%%
%           Isto (provavelmente) é inutil              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

depth = imread("datafixed/gait_depth_oneimage.png");
color = imread("gait_oneimage/gait_RGB_oneimage.png");

%%
figure;
subplot(4,4,3);
imshow(mat2gray(depth));
title('Imagem depth');
imgSize = size(depth);
depthCrop = depth(imgSize(1)/4-20 :imgSize(1)*3/4-50,imgSize(2)*3/8 :imgSize(2)*5/8);
subplot(4,4,1);
imshow(mat2gray(depthCrop));
title('Imagem depth já recortada');
%%
[sizeCropX,sizeCropY] = size(depthCrop);
depthCrop=impyramid(impyramid(depthCrop,'reduce'),'expand');
delta=zeros(6);
for i = 0:5
    delta(i+1) = mean(depthCrop(sizeCropX-(i+1),:))- mean(depthCrop(sizeCropX-i,:));
end
Olddepth=depth;
OldCrop=depthCrop;


noBack1 = noBack(depth,nan,'line');

noBack2 = noBack(depthCrop,nan,'pixel');

subplot(4,4,5);
imshow(mat2gray(noBack1));
title('Imagem depth noback');
subplot(4,4,6);
imshow(mat2gray(noBack2));
title('Imagem depth crop noback');
deltaD = mean(delta);
newImage = depthCrop;
mean(depthCrop(sizeCropX-1,:));

%{
for i=1:sizeCropX
    k = sizeCropX - i +1;
    dmax = (i)*deltaD + mean(depthCrop(sizeCropX,:));
    dmin = dmax - 200;
    for j = 1:sizeCropY
        if(all(newImage(k,j)>=dmax-10) || all(newImage(k,j)<dmin ))
            newImage(k,j)=0;
        end
    end

end
%}
subplot(4,4,2);
imshow(mat2gray(newImage));
title('Depth no background');
newImage=imbinarize(newImage);
se=strel('disk',2,6);
se=strel([0 1 0;1 1 1;0 1 0]);
%se = strel('square', 5)
erode = imerode(newImage, se);


%se=strel('cross',2,0);
%se=strel('line',5,90);
close=imclose(erode,se);

subplot(4,4,4);
imshow(mat2gray(close));
title('erodeclose');

function ret = noBack(depth,setAs,action)
[sizeCropX,sizeCropY] = size(depth);
for i=1:sizeCropX
H(i)= mean(double(depth(i,:)),'omitnan');
D(i)= std(double(depth(i,:)),'omitnan');
%fit(i) = fitdist(transpose(depth(i,:)),'Normal');
end

%H(:)=fit(:).mu;
%D(:)=(fit(:).sigma);

D=D/2;
if(strcmp(action,'pixel'))
for i=1:sizeCropX
    for j=1:sizeCropY
        if( all(depth(i,j)<=(H(i)+D(i))) && all(depth(i,j)>=(H(i)-D(i))) )
            ret(i,j)=setAs;
        else
            ret(i,j)=H(i);
        end
    end
end
elseif(strcmp(action,'linear'))
    ret= zeros(size(depth));
    for i=1:sizeCropX
        H(i)= mean(double(depth(i,:)),'omitnan');
    end
    vectr=1:211;
    vectr2=1:212;
    mdl=fitlm(vectr,H);
    H =predict(mdl,transpose(vectr));
    for i=1:sizeCropX
        ret(i,:)= ret(i,:)-(H(i)-20*D(i));
        for j=1:sizeCropY
            if(ret(i,j)<0)
                ret(i,j)=H(i);
            end
        end
    end

else
    ret=depth;
    for i=1:sizeCropX
        ret(i,:)= ret(i,:)-(H(i));
        for j=1:sizeCropY
            if(ret(i,j)<0)
                ret(i,j)=ret(i,j);
            end
        end
    end
    
end
end
