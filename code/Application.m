%==========================================================================
% Application
% Computes posterior probabilities and applies the machine vision model to
% new input.
%==========================================================================
function Application
    
    close all;
    
    % Load training data
    load('Apples.mat','ApplesEstimate');
    load('NonApples.mat','NonApplesEstimate');
    
    % Load Pictures and resize if too large for quick computation
    PictureOne = imread('testApples/Apples_by_MSR_MikeRyan_flickr.jpg');
    PictureTwo = imresize(imread('testApples/audioworm-QKUJj2wmxuI-original.jpg'),0.1);
    PictureThree = imresize(imread('testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg'),0.1);
    PictureFour = imread('ownApples/ownphoto1.jpg');
    PictureFive = imread('ownApples/ownphoto2.jpg');

    A = getPost(PictureOne,0.2,ApplesEstimate,NonApplesEstimate);
    B = getPost(PictureOne,0.5,ApplesEstimate,NonApplesEstimate);
    C = getPost(PictureTwo,0.3,ApplesEstimate,NonApplesEstimate);
    D = getPost(PictureTwo,0.6,ApplesEstimate,NonApplesEstimate);
    E = getPost(PictureThree,0.4,ApplesEstimate,NonApplesEstimate);
    F = getPost(PictureThree,0.6,ApplesEstimate,NonApplesEstimate);
    G = getPost(PictureFour,0.2,ApplesEstimate,NonApplesEstimate);
    H = getPost(PictureFour,0.5,ApplesEstimate,NonApplesEstimate);
    I = getPost(PictureFive,0.3,ApplesEstimate,NonApplesEstimate);
    J = getPost(PictureFive,0.6,ApplesEstimate,NonApplesEstimate);
    
    % Display results
    figure;
    subplot(3,5,1); imshow(PictureOne);
    subplot(3,5,2); imshow(PictureTwo);
    subplot(3,5,3); imshow(PictureThree);
    subplot(3,5,4); imshow(PictureFour);
    subplot(3,5,5); imshow(PictureFive);
    subplot(3,5,6); imagesc(A); title('Prior = 0.2'); axis off;
    subplot(3,5,7); imagesc(C); title('Prior = 0.3'); axis off;
    subplot(3,5,8); imagesc(E); title('Prior = 0.4'); axis off;
    subplot(3,5,9); imagesc(G); title('Prior = 0.2'); axis off;
    subplot(3,5,10); imagesc(I); title('Prior = 0.3'); axis off;
    subplot(3,5,11); imagesc(B); title('Prior = 0.5'); axis off;
    subplot(3,5,12); imagesc(D); title('Prior = 0.6'); axis off;
    subplot(3,5,13); imagesc(F); title('Prior = 0.6'); axis off;
    subplot(3,5,14); imagesc(H); title('Prior = 0.5'); axis off;
    subplot(3,5,15); imagesc(J); title('Prior = 0.6'); axis off;
    colormap(gray);
    
end


%==========================================================================
% Posterior Probabilities
% Computes the posterior probabilites as in the other parts. Note that the
% prior for non-apple pixels is simply the complement.
%==========================================================================
function post = getPost(picture,prior,ApplesEstimate,NonApplesEstimate)
    [imY, imX, ~] = size(picture);
    post = zeros(imY,imX);
    for y = 1:imY
            post(y,:) = ((getMGP((squeeze(double(picture(y,:,:)))')./255,ApplesEstimate))*prior)./((getMGP((squeeze(double(picture(y,:,:)))')./255,ApplesEstimate))*prior + (getMGP((squeeze(double(picture(y,:,:)))')./255,NonApplesEstimate))*(1-prior));
    end
end


%==========================================================================
% Mixed Gaussian Likelihood
% Computes probability for a mixture of Gaussians
%==========================================================================
function like = getMGP(X,mixGauss)
    like = zeros(size(X,2),1);
    for i = 1:(mixGauss.k)
        like = like + exp(-0.5*diag((X-(repmat(mixGauss.mean(:,i),1,size(X,2))))'*(mixGauss.cov(:,:,i))^(-1)*(X-(repmat(mixGauss.mean(:,i),1,size(X,2))))))/((2*pi)^((mixGauss.d)/2)*det(mixGauss.cov(:,:,i))^0.5);
    end
end
