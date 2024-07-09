import json
import os
import method
from tqdm import tqdm
import gc
import configparser
from PIL import Image
import numpy as np

# get the config for the MSP
config = configparser.ConfigParser()

# Attention!!! this path is based on the root path of "code", which means that you have to start the code from the root path of "code"
config.read("../config/MSP.ini")

label_json_path = config['dataset']['label_json_path']
label_json = json.load(open(label_json_path, 'rb'))

available_gends = ['Male', 'Female']
available_emos = ['H','S','N', 'A', 'U', 'D', 'F', 'C']

emo_dict = {'H':'happy', 'S':'sad', 'N':'neutral', 'A':'angry', 'U':'surprise', 'D':'disgust', 'F':'fear', 'C':'contempt'}

Train_set = ['Train', 'Development']
Test_set = ['Test1', 'Test2']

def func(dataset_path, transcript_path, range_set):
    info = []
    for wav_id in label_json:
        # decide whether to use this wav_id
        split_set = label_json[wav_id]['Split_Set']
        if split_set not in range_set:
            continue
        
        # print(wav_id, end='\r')
        gend = label_json[wav_id]['Gender']
        label = label_json[wav_id]['EmoClass']

        if gend not in available_gends:
            continue
        if label not in available_emos:
            continue

        sent = open(transcript_path+wav_id[:-3]+'txt','r').readline()

        file_name = dataset_path+wav_id
        
        try:
            info.append((file_name, wav_id, label, sent, gend))
        except Exception as e:
            print(e)
            continue

    return info


def get_file_name(config):
    dataset_path = config['dataset']['dataset_path']
    
    transcript_path = config['dataset']['transcript_path']
    
    
    
    train_info = func(dataset_path, transcript_path, Train_set)
    test_info = func(dataset_path, transcript_path, Test_set)
    info = func(dataset_path, transcript_path, Train_set + Test_set)

    return info, train_info, test_info

def process(info, method_name, config, kind):
    method_func = getattr(method, method_name)
    
    # make sure the result prefix exists
    save_path_prefix = config['result']['result_prefix'] + method_name + '/' + kind + '/'
    os.makedirs(save_path_prefix, exist_ok=True)
    
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent, gen_label = item
            # print(name)
            pbar.set_description(f"Processing {kind}: {name}")
            try:
                method_func(file_name, name, save_path_prefix)
            except MemoryError as me:
                print(f"MemoryError: {me}")
            except Exception as e:
                print(e)
                continue
            finally:
                del file_name, name, label, sent, gen_label
            
            if pbar.n % 100 == 0:
                gc.collect()

            pbar.update(1)


def get_json(info, method_name, config, kind, pattern='reopen'):
    prompt_text = getattr(method, method_name + '_prompt_text')
    answer_format = getattr(method, method_name + '_answer_format_MSP')
    
    save_path_prefix = config['result']['result_prefix'] + method_name + '/' + kind + '/'
    
    json_prefix = config['json']['json_prefix'] + method_name + '/'
    jsn_data = []
    
    # make sure that the json prefix exists
    os.makedirs(json_prefix, exist_ok=True)
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent, gen_label = item
            try:
                answer = answer_format%(emo_dict[label], gen_label)
                jsn_data.append({"img":save_path_prefix + name + '.png', "prompt":prompt_text, "label":answer})
            except Exception as e:
                continue
            finally:
                del file_name, name, label, sent, gen_label
            
            pbar.update(1)
            
    # get the name of the json
    json_name = kind + '_gen_utts_emo_json_detailedpromt_v2.json'
    
    if pattern == 'append':
        with open(json_prefix + json_name, 'r') as fp:
            former_data = json.load(fp)
        
        jsn_data = former_data + jsn_data
        
        del former_data
    
    # store the json
    with open(json_prefix + json_name, 'w') as fp:
        json.dump(jsn_data, fp)
    
    del jsn_data, prompt_text, answer_format, save_path_prefix, json_prefix, json_name
    gc.collect()


def save_npz(info, method_name, config, kind):
    """save the result as a npz file, which consists of name, label, sent and img
    name  : just the wav name
    label : the label of the emotion
    sent  : the word sentence
    img   : the image of the video, saved as narray

    Args:
        info (list): train/test/dev info
        method_name (str): method name
        config (_type_): config info
        kind (str): train/test/dev
    """
    npz_prefix = config['result']['npz_prefix'] + method_name + '/' + kind + '/'
    os.makedirs(npz_prefix, exist_ok=True)
    
    save_path_prefix = config['result']['result_prefix'] + method_name + '/' + kind + '/'
    
    if os.path.isfile(npz_prefix + f'MSP_{method_name}_{kind}.npz'):
        print(f'MSP_{method_name}_{kind}.npz exists! please delete it if you want to generate the new npz')
        return
    
    data_list = []
    
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent, gen_label = item
            pbar.set_description(f"Processing {kind}: {name}")
            
            save_path = save_path_prefix + name[:-4] + '.png'
            
            if os.path.isfile(save_path):
                img = Image.open(save_path).convert("RGB")
                # img = transforms.ToTensor()(img)
                img = np.array(img)
                
                data = {
                    'name': name,
                    'label': label,
                    'sent': sent,
                    'img': img,
                    'gen_label': gen_label
                }
                
                data_list.append(data)
                
            pbar.update(1)
    
    np.savez_compressed(npz_prefix + f'MSP_{method_name}_{kind}.npz', data_list=data_list)
    print(f'successfully saved MSP_{method_name}_{kind}.npz!')

def run(config, method_name):
    info, train_info, test_info = get_file_name(config)
    
    get_json(train_info, method_name, config, 'train')
    get_json(test_info, method_name, config, 'test')
    
    process(train_info, method_name, config, 'train')
    process(test_info, method_name, config, 'test')
    
    save_npz(train_info, method_name, config, 'train')
    save_npz(test_info, method_name, config, 'test')
    
    pass