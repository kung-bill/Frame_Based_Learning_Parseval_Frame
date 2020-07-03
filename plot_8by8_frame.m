function [out] = plot_8by8_frame(D)
%author: BC. Kung
%version: 2019/6/3
%Visualization column of matrix in 8 by 8 block

ver_frame_num = 16;
hor_frame_num = 16;


frame_image = zeros(8*ver_frame_num + ver_frame_num + 1 , 8*hor_frame_num + hor_frame_num + 1, 3); %影像加上邊
frame_image(:,:,3) = 255; %initialize as blue




for i = 1 : ver_frame_num
    for j = 1 : hor_frame_num
        block_indi_start = (i-1)*9+2;
        block_indj_start = (j-1)*9+2;
        
        frame_image((block_indi_start:block_indi_start+7), (block_indj_start:block_indj_start+7), 1) = reshape(uint8(255*mat2gray(D(:, (i-1)*hor_frame_num + j))), 8,8);
        frame_image((block_indi_start:block_indi_start+7), (block_indj_start:block_indj_start+7), 2) = reshape(uint8(255*mat2gray(D(:, (i-1)*hor_frame_num + j))), 8,8);
        frame_image((block_indi_start:block_indi_start+7), (block_indj_start:block_indj_start+7), 3) = reshape(uint8(255*mat2gray(D(:, (i-1)*hor_frame_num + j))), 8,8);
    end
end


out = uint8(frame_image); 
end