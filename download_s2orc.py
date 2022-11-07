import re
import wget
import os
import gzip
import io
import json
from tqdm import tqdm
import pickle

pattern = "(?P<url>https?://[^\s]+)"


file1 = open("./data/download.txt","r").read()
temp = []
for line in file1.split():
    if re.search(pattern, line):
        temp.append(re.search(pattern, line).group("url"))
        
t_list=[]
for i in range(len(temp)):
    
    idx=temp[i][temp[i].find("_")+1:temp[i].find(".jsonl")]
    if int(idx)<66:                         
        t_list.append(temp[i])

for e in t_list:
    wget.download(url=e.strip("'"))


targetfiles = [x for x in os.listdir() if x.endswith("gz")]

data = {}

for f in targetfiles:
    with gzip.open(f, 'rb') as gz:
        f = io.BufferedReader(gz)
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            mag_field_of_study = metadata_dict['mag_field_of_study']

            if mag_field_of_study and len(mag_field_of_study)==1 and ('Computer Science' in mag_field_of_study):     
                if metadata_dict["abstract"] and metadata_dict["year"] and (2021>=metadata_dict["year"]>=1994):
                    if metadata_dict["year"] not in data:
                        data[metadata_dict["year"]]=[metadata_dict["abstract"]]
                    else:
                        data[metadata_dict["year"]].append(metadata_dict["abstract"])

for year in data:
    print(f"In {year}, there are {len(data[year])} paper collected")
    
pickle.dump(data,open(b"extra_s2orc.pickle","wb"))

