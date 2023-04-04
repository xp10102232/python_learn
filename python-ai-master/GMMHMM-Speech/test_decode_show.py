import librosa
import numpy as np
import os
from GMM_hmm import compute_B_map,decoder
from isolated_word_recognition import extract_MFCC
import matplotlib.pyplot as plt

if __name__ == "__main__":
 
    ''' viterb 译码测试'''
    # 加载模型
    models = np.load("models.npy",allow_pickle=True)
    model = models[1]
    test_dir ="test"
    
    # 读取测试 wav 并提取特征
    # wav_file = os.path.join(test_dir,str(2)+".wav")
    wav_file = "train\\2\\1.wav"
    fea = extract_MFCC(wav_file)
    
    # 进行viterbi译码
    B_map,_=compute_B_map(fea,model)
    prob_max,states = decoder(model,fea,B_map)
    print(states)
    
    # 读取音频文件并计算频谱
    y,sr = librosa.load(wav_file,sr=8000)   
    S = librosa.stft(y,n_fft=256, hop_length=80, win_length=256)
    S = np.abs(S)
    Spec = librosa.amplitude_to_db(S,ref=np.max)
    
    # 绘制谱图
    fig, ax = plt.subplots()
    ax.imshow(Spec,origin='lower')
    
    # 找到状态变化的位置并画线
    for i in range(1,len(states)):
        if states[i] != states[i-1]:
            plt.vlines(i-1, 0, 128, colors = "c", linestyles = "dashed")
    
    plt.show() 
    
    
    
    
    
    
    
    
    
    
    

    
    
    