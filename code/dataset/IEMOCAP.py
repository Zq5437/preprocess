import json
import pickle as pkl
import method
import os
from tqdm import tqdm
import gc
import configparser

# get the config for the IEMOCAP
config = configparser.ConfigParser()

# Attention!!! this path is based on the root path of "code", which means that you have to start the code from the root path of "code"
config.read("../config/IEMOCAP.ini")

answer_format = "This '%s' is from a '%s' person, expressing '%s' emotion by  saying '%s'."


video_ids, video_speakers, video_labels, video_text, \
video_audio, video_visual, video_sentence, trainVids, \
test_vids = pkl.load(open(config['dataset']['pickle_path'], 'rb'), encoding='latin1')

emo_dict = {0:'happy', 1:'sad', 2:'neutral', 3:'angry', 4:'excited', 5:'frustrated'}

def func(dataset_path, kind_ids):
    info = []
    for vid in kind_ids:
        fid, mid = 0,0
        init_str = '000'
        for gend, sent, label in zip(video_speakers[vid], video_sentence[vid], video_labels[vid]):
            if gend == 'F':
                gen_label = 'female'
                if fid<10:
                    num = f'{init_str[:-1]}{str(fid)}'
                elif fid>=10 and fid<100:
                    num = f'{init_str[:-2]}{str(fid)}'
                elif fid>=10:
                    num = f'{init_str[:]}{str(fid)}'
                wav_id = "%s_%s%s"%(vid,gend,num)
                # print(wav_id, sent, emo_dict[label])
                fid += 1
            if gend == 'M':
                gen_label = 'male'
                if mid<10:
                    num = f'{init_str[:-1]}{str(mid)}'
                elif mid>=10 and mid<100:
                    num = f'{init_str[:-2]}{str(mid)}'
                elif mid>=10:
                    num = f'{init_str[:]}{str(mid)}'
                wav_id = "%s_%s%s"%(vid,gend,num)
                # print(wav_id, sent, emo_dict[label])
                mid += 1

            file_name = dataset_path + wav_id + '.wav'
            
            try:
                info.append((file_name, wav_id, emo_dict[label], sent, gen_label))
            except Exception as e:
                print(e)
                continue
            
    return info


def get_file_name(config):
    dataset_path = config['dataset']['dataset_path']
    
    info = func(dataset_path, video_ids)
    train_info = func(dataset_path, trainVids)
    test_info = func(dataset_path, test_vids)
    
    return info, train_info, test_info

def process(info, method_name, config):
    method_func = getattr(method, method_name)
    
    # make sure the result prefix exists
    save_path_prefix = config['result']['result_prefix'] + method_name + '/'
    os.makedirs(save_path_prefix, exist_ok=True)
    
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent, gen_label = item
            # print(name)
            pbar.set_description(f"Processing {name}")
            try:
                method_func(file_name, name + ".wav", save_path_prefix)
            except MemoryError as me:
                print(f"MemoryError: {me}")
            except Exception as e:
                print(e)
                continue
            finally:
                del file_name, name, label, sent
            
            if pbar.n % 100 == 0:
                gc.collect()

            pbar.update(1)


def get_json(info, method_name, config, kind, pattern='reopen'):
    prompt_text = getattr(method, method_name + '_prompt_text')
    answer_format = getattr(method, method_name + '_answer_format_IEMOCAP')
    save_path_prefix = config['result']['result_prefix'] + method_name + '/'
    json_prefix = config['json']['json_prefix'] + method_name + '/'
    jsn_data = []
    
    # make sure that the json prefix exists
    os.makedirs(json_prefix, exist_ok=True)
    
    # generate the result
    with tqdm(total=len(info)) as pbar:
        for item in info:
            file_name, name, label, sent, gen_label = item
            try:
                answer = answer_format%(gen_label, label, sent)
                jsn_data.append({"img":save_path_prefix + name + '.png', "prompt":prompt_text, "label":answer})
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
    
    pass
    
    
def run(config, method_name):
    info, train_info, test_info = get_file_name(config)
    
    get_json(train_info, method_name, config, 'tra')
    get_json(test_info, method_name, config, 'evl')
    get_json(test_info, method_name, config, 'tra', 'append')
    
    process(info, method_name, config)
    
    pass