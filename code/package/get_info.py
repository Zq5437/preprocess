import argparse
import configparser

def get_config(dataset):
    config = configparser.ConfigParser()
    try:
        config.read(f'../config/{dataset}.ini')
    except Exception as e:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Failed to read config file")
        print(e)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<")
    
    return config

def get_info():
    """get the info of dataset and method

    Returns:
        str: the name of dataset
        str: the name of method
    """
    parser = argparse.ArgumentParser("Get the info of dataset and method")
    
    parser.add_argument('--dataset', '-d', type=str, help='dataset name', required=True)
    parser.add_argument('--method', '-m',type=str, help='method name', required=True)
    
    args = parser.parse_args()
    
    config = get_config(args.dataset)
    return args.dataset, args.method, config
    