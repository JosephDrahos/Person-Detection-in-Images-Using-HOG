clc, clear
peopleDir = 'Person_Dataset/CroppedPpl';
noPeopleDir = 'Person_Dataset/NoPpl';

pplFiles = ls(peopleDir);   %423 people images
noPplFiles = ls(noPeopleDir);  %400 non-people images

featuresPeople = zeros(length(pplFiles)-2, 3780);
featuresNoPeople = zeros(length(noPplFiles)-2, 3780);
featuresTotal = zeros(length(pplFiles) + length(noPplFiles) - 4, 3780);
labels = [ones(1,423) zeros(1,400)]'; %training labels


for i = 3:length(pplFiles)
    fileLoc = fullfile(peopleDir,pplFiles(i,:));
    personImg = imread(fileLoc);
    hog = extractHOGFeatures(personImg);
    featuresPeople(i-2,:) = hog;
    featuresTotal(i-2,:) = hog;
end

for i = 3:length(noPplFiles)
    fileLoc = fullfile(noPeopleDir,noPplFiles(i,:));
    personImg = imread(fileLoc);
    hog = extractHOGFeatures(personImg);
    featuresNoPeople(i-2,:) = hog;
    featuresTotal(i+421,:) = hog;
end

crossValCnt = 10;
interval = 82;
for i = 1:crossValCnt
    accCount = 0;

    %10 fold cross validation 
    if i == 1
        valiDataX = featuresTotal(1:i*interval,:);
        valiDataY = labels(1:i*interval);

        trainDataX = [featuresTotal((i+1)*interval+1:end,:)];
        trainDataY = [labels((i+1)*interval+1:end)];
    elseif i == crossValCnt
        valiDataX = featuresTotal((i-1)*interval+1:end,:);
        valiDataY = labels((i-1)*interval+1:end);

        trainDataX = [featuresTotal(1:(i-1)*interval,:)];
        trainDataY = [labels(1:(i-1)*interval)];
    else
        valiDataX = featuresTotal((i*interval)+1:(i+1)*interval,:);
        valiDataY = labels((i*interval)+1:(i+1)*interval,:);
        trainDataX = [featuresTotal(1:(i*interval),:) ; featuresTotal((i+1)*interval+1:end,:)];
        trainDataY = [labels(1:(i*interval)) ; labels((i+1)*interval+1:end)];
    end   

    
    Model = fitcsvm(trainDataX,trainDataY,'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); 
    
    [label,score] = predict(Model,valiDataX);
    table(label,valiDataY,score,'VariableNames',{'Prediction','GroundTruth','Score'})
    for k = 1:length(label)
        if label(k) == valiDataY(k)
            accCount = accCount + 1;
        end
    end
    %Validation Accuracy
    valAccuracy = (accCount/length(label))*100
end



%% Sliding Window
%sliding window test on which ever image from the data set
testImg = imread('FudanPed00054.png');
[h,w,ch] = size(testImg);
figure(1)
hold on
imshow(testImg)
%sliding window size
winW = 80;
winH = 300;
for i = 1:10:w-winW
    for j = 1:10:h-winH
        win = testImg(j:j+winH,i:i+winW);
        reWin = imresize(win,[128,64]);
        HOGwin = extractHOGFeatures(reWin);
        [label,score] = predict(Model,HOGwin);
        
        if (label == 1) && (score(2) > 0.2)
            rectangle('Position',[i,j,winW,winH],'EdgeColor','r','Curvature',0.2)
        end
    end
end



