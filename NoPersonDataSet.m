clc, clear
%% Load in data
imgDir = 'Person_Dataset/PNGImages';
maskDir = 'Person_Dataset/PedMasks';
saveDir = 'Person_Dataset/NoPpl';

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
    [h,w,c] = size(img);
    for j = 1:length(uniqPpl)-1
        truth = 1;
        while(truth == 1)
            for k = 1:64:h-127 
                for f = 1:32:w-63 
                    kernal = img(k:k+127,f:f+63, :);
                    if(sum(kernal) == 0)
                        img(k:k+127,f:f+63) = 255;
                        idx = idx + 1;
                        imgName = strcat('img_',int2str(idx),'.png');
                        fileSave = fullfile(saveDir,imgName);
                        imgCrop = pngImg(k:k+127,f:f+63, :);
                        imwrite(imgCrop,fileSave);
                        truth = 0;
                    end
                end
            end
        truth = 0;
        end
    end
end

