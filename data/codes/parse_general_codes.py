import csv
import re

# Define the table names as they appear in the Data Dictionary
tables = [
    "Total Population",
    "Persons by Sex [2]",
    "Persons of Hispanic or Latino Origin",
    "Total Households",
    "Persons of Marrying Age* by Sex [2] by Marital Status [6]",
    "Persons by Nativity [2]",
    "Persons 18 Years and Over by Educational Attainment [7]",
    "Persons 18 Years and Over by Sex [2] by Age [5] by Educational Attainment [7]",
    "Persons 16 Years and Over by Labor Force and Employment Status [6]",
    "Households by Income* in Previous Year [4]",
    "Median Household Income in Previous Year",
    "Per Capita Income in Previous Year",
    "Persons* below Poverty Level in Previous Year"
]

# File paths and corresponding years
files = {
    '/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/raw_data/NHGIS/general/general_2006_2010_codebook.txt': '2008',
    '/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/raw_data/NHGIS/general/general_2010_2014_codebook.txt': '2012',
    '/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/raw_data/NHGIS/general/general_2014_2018_codebook.txt': '2016',
    '/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/raw_data/NHGIS/general/general_2018_2022_codebook.txt': '2020'
}

# Function to parse a codebook file
def parse_codebook(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the Data Dictionary section
    start = content.find('Data Dictionary')
    end = content.find('Margins of Error')
    if end == -1:
        end = len(content)
    dict_section = content[start:end]
    
    # Parse tables
    table_data = {}
    current_table = None
    for line in dict_section.split('\n'):
        line = line.strip()
        if line.startswith('Table '):
            # Extract table number and name
            match = re.match(r'Table (\d+): \([^)]+\) (.+)', line)
            if match:
                table_num = int(match.group(1))
                table_name = match.group(2)
                current_table = table_name
                table_data[current_table] = []
        elif line and current_table and ':' in line and not line.startswith('Table'):
            # Parse code and description
            parts = line.split(':', 1)
            if len(parts) == 2:
                code = parts[0].strip()
                desc = parts[1].strip()
                table_data[current_table].append((desc, code))
    
    return table_data

# Collect all data
all_data = {}
for table in tables:
    all_data[table] = {}

# Parse each file
for file_path, year in files.items():
    table_data = parse_codebook(file_path)
    for table, entries in table_data.items():
        if table in all_data:
            for desc, code in entries:
                if desc not in all_data[table]:
                    all_data[table][desc] = {}
                all_data[table][desc][year] = code

# Function to generate col_name
def get_col_name(table, description):
    # Clean description
    desc_clean = description.lower().replace('persons:', '').replace('households:', '').replace('median income in previous year:', '').replace('per capita income in previous year', '').replace('poverty status is determined ~', '').strip()
    desc_clean = re.sub(r'[^\w\s]', '', desc_clean).replace(' ', '_').replace('__', '_')
    
    if table == "Total Population":
        if desc_clean == 'total':
            return 'population'
        else:
            return f"population_{desc_clean}"
    elif table == "Persons by Sex [2]":
        return f"sex_{desc_clean}"
    elif table == "Persons of Hispanic or Latino Origin":
        return 'hispanic'
    elif table == "Total Households":
        return 'households'
    elif table == "Persons of Marrying Age* by Sex [2] by Marital Status [6]":
        return f"marital_{desc_clean}"
    elif table == "Persons by Nativity [2]":
        return f"nativity_{desc_clean}"
    elif table == "Persons 18 Years and Over by Educational Attainment [7]":
        return f"education_{desc_clean}"
    elif table == "Persons 18 Years and Over by Sex [2] by Age [5] by Educational Attainment [7]":
        return f"education_{desc_clean}"
    elif table == "Persons 16 Years and Over by Labor Force and Employment Status [6]":
        return f"labor_{desc_clean}"
    elif table == "Households by Income* in Previous Year [4]":
        return f"income_{desc_clean}"
    elif table == "Median Household Income in Previous Year":
        return 'median_household_income'
    elif table == "Per Capita Income in Previous Year":
        return 'per_capita_income'
    elif table == "Persons* below Poverty Level in Previous Year":
        return 'poverty'
    else:
        return desc_clean

# Write to CSV
with open('/Users/arvindsuresh/Documents/Github/Election-prediction-May-2025/general_codes.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['table', 'description', 'code_2008', 'code_2012', 'code_2016', 'code_2020', 'col_name'])
    
    for table in tables:
        for desc in sorted(all_data[table].keys()):
            row = [table, desc]
            for year in ['2008', '2012', '2016', '2020']:
                row.append(all_data[table][desc].get(year, ''))
            row.append(get_col_name(table, desc))
            writer.writerow(row)