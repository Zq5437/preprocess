from csv import DictReader
import os
import method
from tqdm import tqdm, trange
import gc
import importlib
import json
import pickle as pkl
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

def func(dataset_path, csv, skip_info=None):
    """get the info of the dataset

    Args:
        dataset_path (str): the dataset path, mainly for the train, test and dev
        csv (str): the csv file path

    Returns:
        tuple: the info of the dataset, including the file name, wav_id, label and sentence
    """
    info = []
    with open(csv, 'r') as f:
        dict_reader = DictReader(f)   
        list_of_dict = list(dict_reader)  
        
        for d in list_of_dict:
            wav_id = 'sea%s_ep%s_sc%s_utt%s.mp4'%(d['Season'], d['Episode'], d['Scene_ID'], d['Utterance_ID'])
            
            if skip_info is not None and wav_id in skip_info:
                # print("skip wav:", wav_id)
                continue

            label = d['Emotion']
            sent = d['Utterance']

            file_name = dataset_path+wav_id
            try:
                info.append((file_name, wav_id, label, sent))
            except Exception as e:
                print(e)
                continue
    return info
            
def get_file_name(config):
    # get the path of the dataset
    train_dataset_path = config['dataset']['train_dataset_path']
    test_dataset_path = config['dataset']['test_dataset_path']
    dev_dataset_path = config['dataset']['dev_dataset_path']
    
    train_csv = config['csv']['train_csv']
    test_csv = config['csv']['test_csv']
    dev_csv = config['csv']['dev_csv']
    
    train_info = func(train_dataset_path, train_csv)
    test_info = func(test_dataset_path, test_csv)
    dev_info = func(dev_dataset_path, dev_csv)
    return train_info, test_info, dev_info

def process(info, method_name, config, kind):
    method_func = getattr(method, method_name)
    
    # make sure the result prefix exists
    save_path_prefix = config['result']['result_prefix'] + method_name + '/' + kind + '/'
    os.makedirs(save_path_prefix, exist_ok=True)
    
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent = item
            # print(name)
            pbar.set_description(f"Processing {kind}: {name}")
            try:
                method_func(file_name, name, save_path_prefix)
            except MemoryError as me:
                print(f"MemoryError: {me}")
            except Exception as e:
                # print(e)
                continue
            finally:
                del file_name, name, label, sent
            
            if pbar.n % 100 == 0:
                gc.collect()

            pbar.update(1)


def get_json(info, method_name, config, kind, pattern='reopen'):
    prompt_text = getattr(method, method_name + '_prompt_text')
    answer_format = getattr(method, method_name + '_answer_format_EMORY')
    save_path_prefix = config['result']['result_prefix'] + method_name + '/' + kind + '/'
    json_prefix = config['json']['json_prefix'] + method_name + '/'
    jsn_data = []
    
    # make sure that the json prefix exists
    os.makedirs(json_prefix, exist_ok=True)
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent = item
            try:
                answer = answer_format%(label, sent)
                jsn_data.append({"img":save_path_prefix + name[:-4] + '.png', "prompt":prompt_text, "label":answer})
            except Exception as e:
                continue
            finally:
                del file_name, name, label, sent
            
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
    name  : just the mp4 name
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
            file_name, name, label, sent = item
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
                    'img': img
                }
                
                data_list.append(data)
                
            pbar.update(1)
    
    np.savez_compressed(npz_prefix + f'EMORY_{method_name}_{kind}.npz', data_list=data_list)
    print(f'successfully saved EMORY_{method_name}_{kind}.npz!')

def run(config, method_name):
    train_info, test_info, dev_info = get_file_name(config)
    
    get_json(train_info, method_name, config, 'tra')
    get_json(test_info, method_name, config, 'evl')
    get_json(dev_info, method_name, config, 'tra', 'append')
    
    process(train_info, method_name, config, 'train')
    process(test_info,method_name, config, 'test')
    process(dev_info, method_name, config, 'dev')
    
    save_npz(train_info, method_name, config, 'train')
    save_npz(test_info, method_name, config, 'test')
    save_npz(dev_info, method_name, config, 'dev')
    
    
    
    pass