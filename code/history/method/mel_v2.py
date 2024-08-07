import librosa
import matplotlib.pyplot as plt
import numpy as np
import warnings
import gc

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def mel(file_name, name, save_path_prefix):
    try:
        y, sr = librosa.load(file_name)
        # mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmax=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000)
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 创建图形和子图，设置图像大小为1.28x1.28英寸
        plt.subplots(figsize=(1.28, 1.28), dpi=100)
        
        
        # print(mel_spec)
        
        # plt.imshow(mel_spec, aspect='auto', origin='lower', interpolation='none', cmap='magma')
        plt.imshow(mel_spec, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
        plt.axis('off')  # 关闭坐标轴
        
        
        # 移除所有边距
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # 保存频谱图，确保图像大小为128x128像素
        plt.savefig(save_path_prefix + name[:-4] + '.png', bbox_inches='tight', pad_inches=0)
        
        plt.close()
        
        del y, sr, mel_spec
    except MemoryError as me:
        print(f"MemoryError: {me}")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
    finally:
        gc.collect()

        
    return save_path_prefix + name[:-4] + '.png'


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