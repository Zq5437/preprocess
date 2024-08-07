import os
from PIL import Image

dir_prefix = "/home/h666/Zq/2025ICASSP/preprocess/result"

dataset = [dir_prefix + "/" + item for item in os.listdir(dir_prefix) ]

uncorrect = 0

for i in dataset:
    print("processing:", i)
    method = [i + "/" + item for item in os.listdir(i)]
    
    for j in method:
        kind = [j + "/" + item for item in os.listdir(j)]
        
        for k in kind:
            imgs_path = [k + "/" + item for item in os.listdir(k)]
            
            for l in imgs_path:
                try:
                    img = Image.open(l)
                except Exception as e:
                    print("-------------", e, "-------------")
                    print(l)
                    uncorrect += 1
            
            
print("uncorrect:", uncorrect)