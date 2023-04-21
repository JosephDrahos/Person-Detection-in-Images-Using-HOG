clc, clear
%% Load in data
imgDir = 'Person_Dataset/PNGImages';
maskDir = 'Person_Dataset/PedMasks';
saveDir = 'Person_Dataset/CroppedPpl';

imgFiles = ls(imgDir);
maskFiles = ls(maskDir);

%save in format x1, y1, w, h
bboxs = zeros((length(imgFiles) - 2),4);
idx = 0;
for i = 3:(length(imgFiles))
    fileLoc = fullfile(maskDir,maskFiles(i,:));
    fileLoc2 = fullfile(imgDir,imgFiles(i,:));
    img = imread(fileLoc);
    pngImg = imread(fileLoc2);
    uniqPpl = unique(img);
    for j = 1:length(uniqPpl)-1
        idx = idx + 1;
        [yPixelList,xPixelList] = find(img == j);
        x1 = min(xPixelList);
        x2 = max(xPixelList) - 1;
        w = x2 - x1;
        y1 = min(yPixelList);
        y2 = max(yPixelList) - 1;
        h = y2 - y1;
        bboxs(idx,:) = [x1,y1,w,h];
        cropImg = pngImg(y1:y2,x1:x2,:);
        imgName = strcat('img_',int2str(idx),'.png');
        fileSave = fullfile(saveDir,imgName);
        cropImg = imresize(cropImg,[128,64]);
        imwrite(cropImg,fileSave);
    end
end

