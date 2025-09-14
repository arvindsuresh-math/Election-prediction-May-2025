import os
import json

files = [
    'raw_data/NHGIS/race_by_sex/race_by_sex_2006_2010_codebook.txt',
    'raw_data/NHGIS/race_by_sex/race_by_sex_2010_2014_codebook.txt',
    'raw_data/NHGIS/race_by_sex/race_by_sex_2014_2018_codebook.txt',
    'raw_data/NHGIS/race_by_sex/race_by_sex_2018_2022_codebook.txt'
]

years = [2008, 2012, 2016, 2020]

data = []

for file, year in zip(files, years):
    with open(file, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    in_estimates = False
    for line in lines:
        if 'Data Type (E):' in line:
            in_estimates = True
            continue
        if 'Data Type (M):' in line:
            in_estimates = False
            break
        if in_estimates and line.startswith('        ') and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                code = parts[0].strip()
                desc = parts[1].strip()
                data.append({"year": year, "code": code, "description": desc})

with open('race_by_sex_codes.json', 'w') as f:
    json.dump(data, f, indent=2)