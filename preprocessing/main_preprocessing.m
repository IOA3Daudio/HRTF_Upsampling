clc
clear all
close all

dataset_path = './sofa_measured/';
Fs = 44100;
L = 256;
f_ori = Fs*(0:(L/2-1))/L;
f_ind = find(f_ori>200 & f_ori<18000);
f = f_ori(f_ind);
data1 = SOFAload([dataset_path,'pp1_HRIRs_measured.sofa']);
positions = data1.SourcePosition(:,1:2);

target_order = 1;
pos_sparse_ind = get_lebedev_ind(positions,target_order);
% pos_sparse_ind = get_fliege_ind(positions,target_order);
pos_sparse = positions(pos_sparse_ind,:);

save('./hrir_configs_measured.mat','f', 'positions','pos_sparse');



fileExt = '*.sofa';
allFiles = dir(fullfile(dataset_path,fileExt));
numofsofa = size(allFiles,1);

hrtf_dense = zeros(numofsofa,2,length(positions),length(f_ind));
itd_dense = zeros(numofsofa,2,length(positions));
for i = 1:numofsofa
    sofa_name = allFiles(i,1).name;
    disp(sofa_name);
    [hrtf_temp,itd_temp] = get_hrtf(strcat(dataset_path,sofa_name),L,length(positions),f_ind);
    
    hrtf_dense(i,:,:,:) = hrtf_temp;
    itd_dense(i,:,:) = itd_temp;
    
    hrtf_sparse(i,:,:,:) = hrtf_temp(:,pos_sparse_ind,:);
    itd_sparse(i,:,:) = itd_temp(:,pos_sparse_ind);

end
save('hrtf_all_measured.mat','hrtf_dense','hrtf_sparse','-v6');
save('itd_all_measured.mat','itd_dense','itd_sparse','-v6');


