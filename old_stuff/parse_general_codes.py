import json
import re

def parse_general_codebook(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the start of Data Dictionary
    start = content.find('Data Dictionary')
    if start == -1:
        return {}
    
    # Find the end before Margins of Error
    end = content.find('Margins of Error', start)
    if end == -1:
        end = len(content)
    
    section = content[start:end]
    
    # Pattern for indented lines: 8 spaces, code, :, 4 spaces, description
    pattern = re.compile(r'^        ([A-Z0-9]+):    (.+)$', re.MULTILINE)
    
    codes = {}
    for match in pattern.finditer(section):
        code = match.group(1)
        description = match.group(2).strip()
        codes[code] = description
    
    return codes

# Files and years
files_years = [
    ('raw_data/NHGIS/general/general_2006_2010_codebook.txt', 2008),
    ('raw_data/NHGIS/general/general_2010_2014_codebook.txt', 2012),
    ('raw_data/NHGIS/general/general_2014_2018_codebook.txt', 2016),
    ('raw_data/NHGIS/general/general_2018_2022_codebook.txt', 2020),
]

all_codes = []
for file_path, year in files_years:
    codes = parse_general_codebook(file_path)
    for code, description in codes.items():
        all_codes.append({
            "year": year,
            "code": code,
            "description": description
        })

# Save to JSON
with open('general_codes.json', 'w') as f:
    json.dump(all_codes, f, indent=4)

print("general_codes.json created successfully.")