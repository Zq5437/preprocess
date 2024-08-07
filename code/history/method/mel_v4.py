import librosa
import matplotlib.pyplot as plt
import numpy as np
import warnings
import gc
from PIL import Image

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def mel(file_name, name, save_path_prefix):
    try:
        # 加载音频文件
        y, sr = librosa.load(file_name)
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000)
        S_dB = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 使用 plt.imsave 直接保存梅尔频谱图数据为灰度图像
        save_path = save_path_prefix + name[0:-4] + '.png'
        
        # 标准化为16位整数图像
        S_dB_shifted = S_dB - S_dB.min()
        S_dB_normalized = (S_dB_shifted / S_dB_shifted.max() * 65535).astype(np.uint16)
        
        # 上下翻转图像
        S_dB_flipped = np.flipud(S_dB_normalized)
        
        # 使用 PIL 保存为16位PNG图像
        save_path = save_path_prefix + name[0:-4] + '_.png'
        img = Image.fromarray(S_dB_flipped)
        img.save(save_path)
        
        # 释放内存
        del y, sr, mel_spec, S_dB, save_path, S_dB_shifted, img
    except MemoryError as me:
        print(f"MemoryError: {me}")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
    finally:
        gc.collect()
        
    return save_path



mel_prompt_text = '''You are an expert on emotion recognition, especially based on the visualization of mel-spectrogram. 
We will provide 1 image of mel-spectrogram. Please recognize the emotion from the following categories:['happy','sad','neutral','angry','excited','frustrated'].
Here are some hints to help you: 
1. happy is characterized by higher pitch and frequent pitch variations, happy voices often display a more active energy distribution in the higher frequency regions and exhibit clear harmonic structures.
2. sadness is typically reflected in lower pitch tones, with energy concentrated in the lower frequency bands and a tendency for prolonged sound segments that indicate a slow pace.
3. a neutral emotion is often associated with a steady energy distribution across the frequency spectrum, minimal pitch fluctuations, and a stable fundamental frequency (F0) range.
4. angry voices are marked by high-energy and sharp spectral features, significant variations in fundamental frequency indicating agitation, and abrupt energy bursts, especially in the higher frequency areas.
5. excitement is characterized by rapid frequency changes, an increase in overall energy levels, particularly in the higher frequencies, and an expanded dynamic range reflecting heightened activity.
6. frustration may manifest as unstable energy patterns, a decrease in fundamental frequency suggesting a downcast mood, and a tense quality in the voice, which could be observed as enhanced frequency components in the spectrogram.
7. the gender difference is also important, differences in adult F0 typically range between 80 and 175 Hz among men, and 160 and 270 Hz among women.
'''
mel_answer_format_MELD = "This mel-spectrogram is expressing '%s' emotion by saying '%s'."
mel_answer_format_IEMOCAP = "This mel-spectrogram is from a '%s' person, expressing '%s' emotion by  saying '%s'."
mel_answer_format_MSP = "[%s][%s]"
mel_answer_format_EMORY = "This mel-spectrogram is expressing '%s' emotion by saying '%s'."