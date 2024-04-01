function onset = get_itd(HRIR)
onset = AKonsetDetect(HRIR, 10, -20, 'rel', [3000 44100]);
end