%==========================================================================
% Training
% Trains on both apples and non-apples, which takes some time. See called
% functions for documentation. The number of Gaussians used is 5. Data is
% saved to .mat files.
%==========================================================================
function Training

    close all;
    [Apples, NonApples] = load(); 
    ApplesEstimate = fitMixGauss(Apples,5);
    NonApplesEstimate = fitMixGauss(NonApples,5);
    save('Apples.mat','ApplesEstimate');
    save('NonApples.mat','NonApplesEstimate');
    
end


%==========================================================================
% E-M routine
% As in part C
%==========================================================================
function mixGaussEst = fitMixGauss(data,k)
    
    [nDim, nData] = size(data);
    responsibilities = zeros(k, nData);
    
    % Initialise to random values
    mixGaussEst.d = nDim;
    mixGaussEst.k = k;
    mixGaussEst.weight = (1/k)*ones(1,k);
    mixGaussEst.mean = 1*randn(nDim,k);
    for cGauss = 1:k
        mixGaussEst.cov(:,:,cGauss) = (0.1+1.5*rand(1))*eye(nDim,nDim);
    end

    nIter = 20;
    for (cIter = 1:nIter)

        % E-step
        like = getMixGaussLike(data,mixGaussEst); % Note not >log< likelihood
        responsibilities = like./repmat(sum(like,1),k,1); 

        % M-step
        for cGauss = 1:k
            totalResponsibilities = sum(responsibilities(cGauss,:),2);
            mixGaussEst.weight(cGauss) = totalResponsibilities/sum(sum(responsibilities));
            for i=1:nDim
                current = sum(responsibilities(cGauss,:).*data(i,:),2);
                mixGaussEst.mean(i,cGauss) = current./totalResponsibilities;
            end         
            current = (repmat(responsibilities(cGauss,:),nDim,1).*(data-repmat(mixGaussEst.mean(:,cGauss),1,nData)))*((data-repmat(mixGaussEst.mean(:,cGauss),1,nData))');
            mixGaussEst.cov(:,:,cGauss) = current./totalResponsibilities;
        end

    end
end


%==========================================================================
% LIKELIHOOD
% Gaussian likelihood function as in part C, but not >log< likelihood
%==========================================================================
function likelihood = getMixGaussLike(data,mixGaussEst)
    k = mixGaussEst.k;
    nData = size(data,2);
    likelihood = zeros(k,nData);
    for cGauss = 1:k
        for cData = 1:200:nData-200
            likelihood(cGauss,cData:cData+200) = mixGaussEst.weight(cGauss)*multivariatePDF(data(:,cData:cData+200),mixGaussEst,cGauss);
        end
        likelihood(cGauss,nData-200:nData) = mixGaussEst.weight(cGauss)*multivariatePDF(data(:,nData-200:nData),mixGaussEst,cGauss);
    end
end


%==========================================================================
% MULTIVARIATE PDF
% Gives Gaussian probability
%==========================================================================
function mPDF = multivariatePDF(data,struct,index)
    mPDF = ((exp(-0.5*diag((data-(repmat(struct.mean(:,index),1,size(data,2))))'*(struct.cov(:,:,index))^(-1)*(data-(repmat(struct.mean(:,index),1,size(data,2)))))))')/((2*pi)^((struct.d)/2)*det(struct.cov(:,:,index))^0.5);
end


%==========================================================================
% LOAD
% Load images as in LoadApplesScript.m, return 3xI matrix of RGB for Apples
%==========================================================================
function [Apples, NonApples] = load()
    
    % Initialise
    [ApplesRed, ApplesGreen, ApplesBlue, NonApplesRed, NonApplesGreen, NonApplesBlue]  = deal([]);
    
    % Read images
    Iapples = cell(3,1);
    Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
    Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
    Iapples{3} = 'apples/bobbing-for-apples.jpg';
    
    % Read masks
    IapplesMasks = cell(3,1);
    IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
    IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
    IapplesMasks{3} = 'apples/bobbing-for-apples.png';
    
    % Loop through images
    for iImage = 1:3
        
        % Read data and generate a mask for apple pixels and one for non-apple pixels 
        curI = double(imread(Iapples{iImage}))/255;
        curImaskApples = logical(rgb2gray(imread(IapplesMasks{iImage})));
        curImaskNonApples = ones(size(curImaskApples)) - curImaskApples;
        
        % Read RGB
        ApplesRGB  = curI.*repmat(curImaskApples,1,1,3);
        NonApplesRGB = curI.*repmat(curImaskNonApples,1,1,3);
        MoreRed = ApplesRGB(:,:,1);
        MoreGreen = ApplesRGB(:,:,2);
        MoreBlue = ApplesRGB(:,:,3);
        LessRed = NonApplesRGB(:,:,1);
        LessGreen = NonApplesRGB(:,:,2);
        LessBlue = NonApplesRGB(:,:,3);
        
        % Find RGB values
        ApplesRed = [ApplesRed; MoreRed(find(curImaskApples))];
        ApplesGreen = [ApplesGreen; MoreGreen(find(curImaskApples))];
        ApplesBlue = [ApplesBlue; MoreBlue(find(curImaskApples))];
        NonApplesRed = [NonApplesRed; LessRed(find(curImaskNonApples))];
        NonApplesGreen = [NonApplesGreen; LessGreen(find(curImaskNonApples))];
        NonApplesBlue = [NonApplesBlue; LessBlue(find(curImaskNonApples))];
        
    end
    
    % Output
    Apples = [ApplesRed,ApplesGreen,ApplesBlue]';
    NonApples = [NonApplesRed,NonApplesGreen,NonApplesBlue]';

end
