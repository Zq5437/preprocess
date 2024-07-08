import importlib

def get_dataset_module(dataset_name):
    module_name = f"dataset.{dataset_name}"
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        return None
