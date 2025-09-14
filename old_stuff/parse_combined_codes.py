import json
import re
import os

def parse_codebook(file_path, is_race_by_sex=False):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the start
    if is_race_by_sex:
        start = content.find('Data Summary')
    else:
        start = content.find('Data Dictionary')
    if start == -1:
        return {}
    
    # Find the end
    end = content.find('Margins of Error', start)
    if end == -1:
        end = content.find('Citation and Use', start)
    if end == -1:
        end = len(content)
    
    section = content[start:end]
    
    codes = {}
    table_codes = {}
    
    lines = section.split('\n')
    i = 0
    current_table = ""
    current_nhgis = ""
    while i < len(lines):
        line = lines[i]
        if is_race_by_sex:
            # Check for table start
            if re.match(r'^\d+\.\s+', line.strip()):
                current_table = line.strip().split('.', 1)[1].strip()
                i += 1
                # Look for NHGIS code
                while i < len(lines):
                    line = lines[i]
                    if 'NHGIS code:' in line:
                        current_nhgis = line.split(':', 1)[1].strip()
                        table_codes[current_nhgis] = current_table
                        break
                    i += 1
                continue
            # Look for 'Data Type (E):'
            if 'Data Type (E):' in line:
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if 'Data Type (M):' in line:
                        break
                    if re.match(r'^\s*[A-Z0-9]+:', line):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            code = parts[0].strip()
                            desc = parts[1].strip()
                            # Find the table by matching prefix
                            table_name = ""
                            for nhgis, name in table_codes.items():
                                if code.startswith(nhgis):
                                    table_name = name
                                    break
                            full_desc = f"{table_name}: {desc}" if table_name else desc
                            codes[code] = full_desc
                    i += 1
                continue
        else:
            # For general: check for table start
            if line.startswith('Table '):
                match = re.match(r'Table \d+: \([^)]+\) (.+)', line)
                if match:
                    current_table = match.group(1)
                i += 1
                continue
            # Pattern for indented lines
            if re.match(r'^        [A-Z0-9]+:', line):
                match = re.match(r'^        ([A-Z0-9]+):    (.+)$', line)
                if match:
                    code = match.group(1)
                    desc = match.group(2).strip()
                    full_desc = f"{current_table}: {desc}" if current_table else desc
                    codes[code] = full_desc
        i += 1
    
    return codes

# Files and years
race_by_sex_files = [
    ('raw_data/NHGIS/race_by_sex/race_by_sex_2006_2010_codebook.txt', 2008),
    ('raw_data/NHGIS/race_by_sex/race_by_sex_2010_2014_codebook.txt', 2012),
    ('raw_data/NHGIS/race_by_sex/race_by_sex_2014_2018_codebook.txt', 2016),
    ('raw_data/NHGIS/race_by_sex/race_by_sex_2018_2022_codebook.txt', 2020),
]

general_files = [
    ('raw_data/NHGIS/general/general_2006_2010_codebook.txt', 2008),
    ('raw_data/NHGIS/general/general_2010_2014_codebook.txt', 2012),
    ('raw_data/NHGIS/general/general_2014_2018_codebook.txt', 2016),
    ('raw_data/NHGIS/general/general_2018_2022_codebook.txt', 2020),
]

# Collect all codes
all_codes = []
for file_path, year in race_by_sex_files + general_files:
    is_race = 'race_by_sex' in file_path
    codes = parse_codebook(file_path, is_race)
    for code, desc in codes.items():
        all_codes.append({
            "year": year,
            "code": code,
            "description": desc
        })

# Group by description
grouped = {}
for item in all_codes:
    desc = item['description']
    year = item['year']
    code = item['code']
    if desc not in grouped:
        grouped[desc] = {}
    grouped[desc][year] = code

# Convert to list format
result = []
for desc, codes_dict in grouped.items():
    result.append({
        "description": desc,
        "codes": codes_dict
    })

# Sort by description
result.sort(key=lambda x: x['description'])

# Save to JSON
with open('combined_codes.json', 'w') as f:
    json.dump(result, f, indent=4)

print("combined_codes.json created successfully.")