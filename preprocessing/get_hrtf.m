function [hrtf,itd] = get_hrtf(filename,nfft,pos_num,f_ind)
hrtf = zeros(2,pos_num,length(f_ind));
itd = zeros(2,pos_num);
datatemp = SOFAload(filename);
for subj_ear = 1:2
    hrirs = datatemp.Data.IR(:,subj_ear,:);
    hrirs = squeeze(hrirs);
    itd(subj_ear,:) = get_itd(hrirs');
    for i=1:pos_num
        hrir = hrirs(i,:);
        hrir = squeeze(hrir);
        hrtf_temp = fft(hrir, nfft);
        hrtf_temp = abs(hrtf_temp);
        hrtf_temp = hrtf_temp(1:nfft/2);
        hrtf_temp = 20*log10(hrtf_temp+1e-6);
        hrtf_temp_temp = hrtf_temp(f_ind);
        hrtf(subj_ear,i,:) = hrtf_temp_temp;
        
    end
end



end

