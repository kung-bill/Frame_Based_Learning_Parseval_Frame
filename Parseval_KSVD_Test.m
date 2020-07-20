%% Part1: Learning a K-SVD dictionary for the initial dictionary for Parseval K-SVD
clear all
close all
clc

Im = imread('boat256.png');
Im = double(Im);

Im_block_col = im2col(Im, [8,8], 'distinct'); %Im block
Im_block_col_mean = mean(Im_block_col);
Y = Im_block_col - Im_block_col_mean; % E(Yi) = 0, for all i

%K-SVD

%create overcomplete DCT frame
Pn = 16 ;
DCT=zeros(8,Pn);
for k=0:1:Pn-1,
    V=cos([0:1:7]'*k*pi/Pn);
    if k>0, V=V-mean(V); end;
    DCT(:,k+1)=V/norm(V);
end;
DCT=kron(DCT,DCT);

param.K = 256;
param.numIteration = 5;
param.InitializationMethod =  'GivenMatrix';
param.initialDictionary = DCT;
param.displayProgress = 1;
param.preserveDCAtom = 1;
param.L = 63;
param.errorFlag = 0;
param.errorGoal = 1.0e-8;
[D_svd, output] = KSVD(Y, param);
X = full(output.CoefMatrix);

KsvdIm = plot_8by8_frame(D_svd);
figure;
imshow(KsvdIm); title('K-SVD'); %show D_svd element in image

%% Part2: Learning the Parseval K-SVD

Psi0 = D_svd;
% Phi0 = D_svd + randn(size(D_svd, 1), size(D_svd, 2));
% Phi0 = pinv(D_svd)';
Phi0 = D_svd;

X0 = X;
maxIter = 100;
t = 1.0e-10;
rho = [0.1, 1.0e+8, 1.0e+8];
IsRecord = true;
ShowDetail = true;

[Psi, Phi, X, Record] = ParsevalKSVD(Y, Psi0, Phi0, X0, maxIter, t, rho, IsRecord, ShowDetail);

PsiIm = plot_8by8_frame(Psi); 
PhiIm = plot_8by8_frame(Phi);
figure;
subplot(1, 2, 1); imshow(PsiIm); title('$\psi$','Interpreter','latex');
subplot(1, 2, 2); imshow(PhiIm); title('$\phi$','Interpreter','latex');

%Displaying the curve for the convergent behavior
figure;
plot(Record.augLag); xlabel('Iteration'); ylabel('L');

figure;
plot(Record.obj_rep); xlabel('Iteration'); ylabel('$\| Y - \psi X \|_F^2$', 'Interpreter','latex');

figure;
plot(Record.obj_total); xlabel('Iteration'); ylabel('$\rho_1 \| Y - \psi X \|_F^2 + \| \phi^\top Y -  \phi^\top \psi X \|_F^2$', 'Interpreter','latex');

figure;
plot(Record.con1); xlabel('Iteration'); ylabel('$\| \psi \phi^\top - I \|_F^2$', 'Interpreter','latex');

figure;
plot(Record.con2); xlabel('Iteration'); ylabel('$\| \psi - \phi \|_F^2$', 'Interpreter','latex');
%% Part3: Image Compression
% PSNR vesus bits per pixel (entropy) 

Bits = 1:13;

% The Parseval K-SVD case
E1 = zeros(1, length(Bits)); %bit/per pixel
P1 = zeros(1, length(Bits)); %PSNR
DualPsi = pinv(Psi)';
DualPhi = pinv(Phi)';

disp('Processing Parseval K-SVD Dictionary');
for i = Bits
    % Analysis frame is Psi
    % Synthesis frame is canonical dual frame DualPsi 
%     [e, p] = ComputeBPP(i, DualPsi, Psi, Im); 
    [e, p] = ComputeBPP(i, Phi, DualPhi, Im); 
    E1(i) = e;
    P1(i) = p;
    disp(['Computing  Bit ', num2str(i)]);
end

%bpp is not sorted, so we sorting the bpp.
curve1 = zeros(2, length(Bits));
[sortedE, r]= sort(E1); 
curve1(1,:) = sortedE;
curve1(2,:) = P1(r);

% The K-SVD case
Dual_D_svd = pinv(D_svd)';
E2 = zeros(1, length(Bits)); %bit/per pixel
P2 = zeros(1, length(Bits)); %PSNR
disp('K-SVD Dictionary');
for i = Bits
%     [e, p] = ComputeBPP(i, Dual_D_svd, D_svd, Im);
    [e, p] = ComputeBPP(i, D_svd, Dual_D_svd, Im);

    E2(i) = e;
    P2(i) = p;
    disp(['Computing  Bit ', num2str(i)]);
end

curve2 = zeros(2, length(Bits));
[sortedE, r]= sort(E2); 
curve2(1,:) = sortedE;
curve2(2,:) = P2(r);


figure; 
plot(curve1(1,:), curve1(2,:), 'LineStyle' , '-','Marker', '.', 'MarkerSize', 14);
hold on
plot(curve2(1,:), curve2(2,:), 'LineStyle' , '--', 'Marker', '.', 'MarkerSize', 14);
hold off
% ylim([20, 90]);
xlabel('Bits per pixels');
ylabel('PSNR(dB)');
legend('Parseval K-SVD', 'K-SVD');
