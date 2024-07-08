import os
import importlib
import inspect

# 动态导入目录中的所有模块
package_dir = os.path.dirname(__file__)
module_names = [f[:-3] for f in os.listdir(package_dir) if f.endswith('.py') and f != '__init__.py']

for module_name in module_names:
    module = importlib.import_module(f'.{module_name}', package=__name__)
    
    # 获取模块中的所有函数
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # 将函数添加到当前命名空间
        globals()[name] = func
        
    # 获取模块中的所有变量并添加到当前命名空间
        for name, value in inspect.getmembers(module):
            if not inspect.isfunction(value) and not inspect.ismodule(value) and not name.startswith("__"):
                globals()[name] = value