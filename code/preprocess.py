from package.get_info import get_info
import dataset


# get the basic info
dataset_name, method_name, config = get_info()

# import the dataset module(process code)
module = dataset.get_dataset_module(dataset_name)

module.run(config, method_name)