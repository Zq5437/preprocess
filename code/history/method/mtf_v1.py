import librosa
import matplotlib.pyplot as plt
import numpy as np
import warnings
import gc
from pyts.image import MarkovTransitionField

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

image_size = 128

x = MarkovTransitionField(image_size=image_size)

def mtf(file_name, name, save_path_prefix):
    """
    Generate a Markov Transition Field (MTF) from given data in a file and save it as an image.

    Parameters:
    file_name (str): The path to the file containing the sequence of states.
    name (str): The name of the file to save the image.
    save_path_prefix (str): The prefix path where the image will be saved.

    Returns:
    None
    """
    
    y, sr=librosa.load(file_name)
    step = 100
    y = y[::step]
    
    tool = x
    if len(y) < image_size:
        tool = MarkovTransitionField(image_size=len(y))
    y = y.reshape(1, -1)
    y = tool.fit_transform(y)
    
    plt.axis('off')
    plt.imshow(y[0], cmap='jet')
    plt.savefig(save_path_prefix + name[:-4] + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    del y, sr, step, tool
    return



mtf_prompt_text = '''You are an expert on emotion recognition, especially based on the visualization of Markov Transition Fields (MTF).
We will provide 1 image of MTF. Please recognize the emotion from the following categories: ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'].
Here are some hints to help you:
1. happy is characterized by higher pitch and frequent pitch variations, happy voices often display a more active energy distribution in the higher frequency regions and exhibit clear harmonic structures.
2. sadness is typically reflected in lower pitch tones, with energy concentrated in the lower frequency bands and a tendency for prolonged sound segments that indicate a slow pace.
3. a neutral emotion is often associated with a steady energy distribution across the frequency spectrum, minimal pitch fluctuations, and a stable fundamental frequency (F0) range.
4. angry voices are marked by high-energy and sharp spectral features, significant variations in fundamental frequency indicating agitation, and abrupt energy bursts, especially in the higher frequency areas.
5. excitement is characterized by rapid frequency changes, an increase in overall energy levels, particularly in the higher frequencies, and an expanded dynamic range reflecting heightened activity.
6. frustration may manifest as unstable energy patterns, a decrease in fundamental frequency suggesting a downcast mood, and a tense quality in the voice, which could be observed as enhanced frequency components in the spectrogram.
7. the gender difference is also important, differences in adult F0 typically range between 80 and 175 Hz among men, and 160 and 270 Hz among women.
'''
mtf_answer_format_MELD = "This MTF is expressing '%s' emotion by saying '%s'."
mtf_answer_format_IEMOCAP = "This MTF is from a '%s' person, expressing '%s' emotion by saying '%s'."
mtf_answer_format_MSP = "[%s][%s]"
mtf_answer_format_EMORY = "This mel-spectrogram is expressing '%s' emotion by saying '%s'."


