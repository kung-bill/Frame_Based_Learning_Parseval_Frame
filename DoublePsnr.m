function p = DoublePsnr(Im1, Im2)
p = 20*log10(255/sqrt(mean((Im1(:)-Im2(:)).^2)));
end