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
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000)
        
        # 计算宽高比
        melspec_height, melspec_width = mel_spec.shape
        aspect_ratio = melspec_width / melspec_height
        
        # 设置dpi和figsize以保持宽高比
        dpi = 100  # 使用100 dpi
        figsize = (aspect_ratio * 4, 4)  # 保持高度为4英寸，宽度根据宽高比调整
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        S_dB = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                                    y_axis='mel', sr=sr,
                                    fmax=8000, ax=ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(save_path_prefix + name[0:-4] + '.png', bbox_inches='tight', pad_inches=0)
        # print('saved:', name)
        plt.clf()
        plt.close(fig)
        
        del y, sr, mel_spec, S_dB, fig, ax, img, figsize, dpi
    except MemoryError as me:
        print(f"MemoryError: {me}")
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
    finally:
        gc.collect()
        
    return save_path_prefix + name + '.png'


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