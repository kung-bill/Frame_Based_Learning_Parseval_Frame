function [e, p] = ComputeBPP(bitNum, Psi, Phi, Im)
% This function compute the BPP
% That is, z = (Phi^T)*X, and then approximate z by power of 2 value.
% Approximate z vector call z_hat,
ImD = double(Im); %convert img to double

Blk = im2col(ImD,[8,8],'distinct');
Blk_col_mean = mean(Blk); %Blk ���C�@�� Column �� Mean
Y = Blk - kron(Blk_col_mean, ones(64,1)); %��h�C�@��Column �� Mean

Coe = Phi'*Y; % Mapping zero-mean image blocks to coefficients
CoeSign = sign(Coe); % store  the Coe matrix sign information.
absCoe = abs(Coe);
MaxAbsCoe = max(max(absCoe));

%Create Coding table
codingTableValue = CodingTable(bitNum, MaxAbsCoe);
codingTableValueLen = length(codingTableValue);

% approximate absCoe by codingTableValue
% example:  
% MaxAbsCoe = 450
% codingTableValue=[0, 64, 128, 192, 256, 320, 384, 448]
%          [   4,   280]                        [   0,  256]
% absCoe = [34.3, 144.4]      => ApproxAbsCoe = [   0,  128]
%          [ 300,   450]                        [ 256,  448]

%start to approximate absCoe
ApproxAbsCoe = zeros(size(absCoe)); 
for i = 1:size(absCoe,1)
    for j = 1:size(absCoe,2)
        for k = 1:codingTableValueLen
            if(codingTableValue(codingTableValueLen-k+1) <= absCoe(i,j))
                ApproxAbsCoe(i,j) = codingTableValue(codingTableValueLen-k+1);
                break;
            end
        end
    end
end

ApproxCoe = CoeSign.*ApproxAbsCoe;
Imhat = col2im( Psi*ApproxCoe + kron(Blk_col_mean, ones(64,1)),[8,8],[512,512],'distinct');
p = DoublePsnr(Imhat, ImD);
e = entropy(ApproxCoe);
end

function [codingTableValue, codingTable] = CodingTable(bitNum, maxValue)
% N bit Coding Table
%==============================================================
% Input:
% bitNum:  integer. ex. 1,2,3,...
% maxValue: the maximum entry in coded matrix
% Output:
% create Coding Table
% codingTable:
% the first row of codingTable is coding key (binary)
% the second row of codingTable is the corresponding value of coding key 
% codingTableValue: 
% the second row of codingTable
%==============================================================

TableLen = 2^bitNum;
codingTable = cell(2, TableLen);
for i = 0:(TableLen-1)
    codingTable{1,i+1} = dec2bin(i, bitNum);
end

Big2Power = 1;
while Big2Power<=maxValue
    Big2Power = Big2Power*2;
end

delta = Big2Power/(2^bitNum); 

codingTableValue = zeros(1, TableLen);
Temp = 0;
for i = 1 : TableLen
    codingTableValue(i) = Temp;
    Temp = Temp + delta;
end

for i = 1:TableLen
    codingTable{2,i} = codingTableValue(i);
end

end