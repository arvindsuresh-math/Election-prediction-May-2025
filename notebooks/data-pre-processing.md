```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

years = [2008, 2012, 2016, 2020]
```

### Step 1: Clean the demographic data


```python
dfs_general = {year: pd.read_csv(f"../data/raw_data/NHGIS/general/general_{year-2}_{year+2}.csv") for year in years}

dfs_race_sex = {
    year: pd.read_csv(
        f"../data/raw_data/NHGIS/race_by_sex/race_by_sex_{year-2}_{year+2}.csv",
        encoding='latin-1'  # or 'cp1252' or 'iso-8859-1'
    ) 
    for year in years
}

# Import the csv with codes and col names (made using github copilot)
codes = pd.read_csv("../data/codes/combined_codes.csv")

# Create separate dataframes for each year
demographic_dfs = {}

for year in years:
    # Merge general and race_sex dataframes on GISJOIN
    df_general = dfs_general[year]
    df_race_sex = dfs_race_sex[year]
    merged_df = pd.merge(df_general, df_race_sex, on=['GISJOIN', 'STATE', 'COUNTY'], how='inner')

    id_cols = {
        'gisjoin': merged_df['GISJOIN'],
        'year': year,
        'state': merged_df['STATE'].apply(lambda x: x.upper()),
        'county': merged_df['COUNTY']
    }
    
    # Create new columns based on codes
    col_codes = codes[f'code_{year}'].unique().tolist()
    col_names = codes['col_name'].unique().tolist()
    col_mapping = dict(zip(col_codes, col_names))
    demographic_cols = {
        col_mapping[code]: merged_df[code] for code in col_codes if code in merged_df.columns
    }

    # Make the dataframe and print summary
    new_df = pd.DataFrame({**id_cols, **demographic_cols})
    print(f"Year: {year}")
    print(f"Shape: {new_df.shape}")
    print(f"Columns: {new_df.columns}")

    demographic_dfs[year] = new_df

# check that all dataframes have identical columns
cols_by_year = {year: set(df.columns) for year, df in demographic_dfs.items()}
common_cols = set.intersection(*cols_by_year.values())
for year, cols in cols_by_year.items():
    assert cols == common_cols, f"Columns for year {year} do not match common columns"

# examine Nans, print only cols with Nans
for year, df in demographic_dfs.items():
    print(f"Year: {year}")
    nans = df.isna().sum()
    nans = nans[nans > 0]
    print(nans)

# merge all years into a single dataframe
demographic_df = pd.concat(demographic_dfs.values(), ignore_index=True)

# Drop rows with any Nans
demographic_df.dropna(inplace=True)

# Keep only rows where GISJOIN exists for all 4 years
valid_gisjoins = demographic_df['gisjoin'].value_counts()
valid_gisjoins = valid_gisjoins[valid_gisjoins == 4].index
demographic_df = demographic_df[demographic_df['gisjoin'].isin(valid_gisjoins)]

# Convert all numeric columns to int
for col in demographic_df.columns:
    if col not in ['gisjoin', 'state', 'county']:
        demographic_df[col] = demographic_df[col].astype(int)

# Check size by year
print(demographic_df.groupby('year').size())

# Save the final dataframe
demographic_df.to_csv("../data/processed_data/demographic_data.csv", index=False)
```

    Year: 2008
    Shape: (3177, 328)
    Columns: Index(['gisjoin', 'year', 'state', 'county', 'population', 'female', 'male',
           'hispanic', 'households', 'female_divorced',
           ...
           'multi_female_18_and_19', 'multi_female_20_to_24',
           'multi_female_25_to_29', 'multi_female_30_to_34',
           'multi_female_35_to_44', 'multi_female_45_to_54',
           'multi_female_55_to_64', 'multi_female_65_to_74',
           'multi_female_75_to_84', 'multi_female_85_plus'],
          dtype='object', length=328)
    Year: 2012
    Shape: (3220, 328)
    Columns: Index(['gisjoin', 'year', 'state', 'county', 'population', 'female', 'male',
           'hispanic', 'households', 'female_divorced',
           ...
           'multi_female_18_and_19', 'multi_female_20_to_24',
           'multi_female_25_to_29', 'multi_female_30_to_34',
           'multi_female_35_to_44', 'multi_female_45_to_54',
           'multi_female_55_to_64', 'multi_female_65_to_74',
           'multi_female_75_to_84', 'multi_female_85_plus'],
          dtype='object', length=328)
    Year: 2016
    Shape: (3220, 328)
    Columns: Index(['gisjoin', 'year', 'state', 'county', 'population', 'female', 'male',
           'hispanic', 'households', 'female_divorced',
           ...
           'multi_female_18_and_19', 'multi_female_20_to_24',
           'multi_female_25_to_29', 'multi_female_30_to_34',
           'multi_female_35_to_44', 'multi_female_45_to_54',
           'multi_female_55_to_64', 'multi_female_65_to_74',
           'multi_female_75_to_84', 'multi_female_85_plus'],
          dtype='object', length=328)
    Year: 2020
    Shape: (3222, 328)
    Columns: Index(['gisjoin', 'year', 'state', 'county', 'population', 'female', 'male',
           'hispanic', 'households', 'female_divorced',
           ...
           'multi_female_18_and_19', 'multi_female_20_to_24',
           'multi_female_25_to_29', 'multi_female_30_to_34',
           'multi_female_35_to_44', 'multi_female_45_to_54',
           'multi_female_55_to_64', 'multi_female_65_to_74',
           'multi_female_75_to_84', 'multi_female_85_plus'],
          dtype='object', length=328)
    Year: 2008
    nativity_foreign_born    78
    nativity_native          78
    per_capita_income         1
    dtype: int64
    Year: 2012
    nativity_foreign_born    78
    nativity_native          78
    dtype: int64
    Year: 2016
    nativity_foreign_born         78
    nativity_native               78
    labor_16_plus_in_force         1
    labor_16_plus_civilian         1
    labor_16_plus_employed         1
    labor_16_plus_unemployed       1
    labor_16_plus_armed_forces     1
    labor_16_plus_not_in_force     1
    income_10k_to_15k              1
    income_15k_to_25k              1
    income_25k_plus                1
    income_less_than_10k           1
    median_household_income        1
    per_capita_income              1
    poverty                        1
    dtype: int64
    Year: 2020
    nativity_foreign_born      78
    nativity_native            78
    median_household_income     1
    dtype: int64
    year
    2008    3085
    2012    3085
    2016    3085
    2020    3085
    dtype: int64


### Step 2: Clean the election outcome data


```python
df = pd.read_csv("../data/raw_data/MIT_election_data/president_2000-2020_by_county.csv")

# Quick look at the data
print(df.columns)
print(df.head())

# drop unnecessary columns
df = df.drop(columns=['state', 'state_po', 'county_name', 'office', 'candidate', 'version', 'mode'])

# Keep only relevant years
df = df[df['year'].isin([2008, 2012, 2016, 2020])]

# Drop rows with NaN values in 'county_fips'
df = df.dropna(subset=['county_fips'])

# If party is not DEMOCRAT or REPUBLICAN, set it to OTHER
df['party'] = df['party'].apply(lambda x: x if x in ['DEMOCRAT', 'REPUBLICAN'] else 'OTHER')

# Aggregate "OTHER" parties into a single row per county per year
df = df.groupby(['year', 'county_fips', 'party'], as_index=False).agg({
    'candidatevotes': 'sum', 
    'totalvotes': 'first'})

# Make separate columns for democrat, republican, and other votes
df = df.pivot_table(index=['year', 'county_fips', 'totalvotes'], 
                    columns='party', 
                    values='candidatevotes').reset_index()

# get counties that occur for all 4 years
county_counts = df['county_fips'].value_counts()
counties_to_keep = county_counts[county_counts == 4].index
election_df = df[df['county_fips'].isin(counties_to_keep)]

# Convert all columns to int except state
for col in election_df.columns:
    if col != 'state':
        election_df[col] = election_df[col].astype(int)

# Create gisjoin column from county_fips
def fips_to_gisjoin(fips):
    state = str(fips // 1000).zfill(2)
    county = str(fips % 1000).zfill(3)
    return "G" + state + "0" + county + "0"

election_df['gisjoin'] = election_df['county_fips'].apply(fips_to_gisjoin)

# Drop county_fips column
election_df = election_df.drop(columns=['county_fips'])

# Rename the vote columns
election_df = election_df.rename(columns={
    'DEMOCRAT': 'democrat_voter',
    'REPUBLICAN': 'republican_voter',
    'OTHER': 'other_voter'
    })

# Get size by year
print(election_df.groupby('year').size())
print(election_df.head())

# Save to csv
election_df.to_csv("../data/processed_data/election_data.csv", index=False)
```

    Index(['year', 'state', 'state_po', 'county_name', 'county_fips', 'office',
           'candidate', 'party', 'candidatevotes', 'totalvotes', 'version',
           'mode'],
          dtype='object')
       year    state state_po county_name  county_fips        office  \
    0  2000  ALABAMA       AL     AUTAUGA       1001.0  US PRESIDENT   
    1  2000  ALABAMA       AL     AUTAUGA       1001.0  US PRESIDENT   
    2  2000  ALABAMA       AL     AUTAUGA       1001.0  US PRESIDENT   
    3  2000  ALABAMA       AL     AUTAUGA       1001.0  US PRESIDENT   
    4  2000  ALABAMA       AL     BALDWIN       1003.0  US PRESIDENT   
    
            candidate       party  candidatevotes  totalvotes   version   mode  
    0         AL GORE    DEMOCRAT            4942       17208  20220315  TOTAL  
    1  GEORGE W. BUSH  REPUBLICAN           11993       17208  20220315  TOTAL  
    2     RALPH NADER       GREEN             160       17208  20220315  TOTAL  
    3           OTHER       OTHER             113       17208  20220315  TOTAL  
    4         AL GORE    DEMOCRAT           13997       56480  20220315  TOTAL  
    year
    2008    3153
    2012    3153
    2016    3153
    2020    3153
    dtype: int64
    party  year  totalvotes  democrat_voter  other_voter  republican_voter  \
    0      2008       23641            6093          145             17403   
    1      2008       81413           19386          756             61271   
    2      2008       11630            5697           67              5866   
    3      2008        8644            2299           83              6262   
    4      2008       24267            3522          356             20389   
    
    party   gisjoin  
    0      G0100010  
    1      G0100030  
    2      G0100050  
    3      G0100070  
    4      G0100090  


### Step 3: Create the county area data


```python
# load cleaned dataset from previous project to get the census areas
area_df = pd.read_csv("../data/raw_data/all_years.csv")

# Add gisjoin column
area_df['gisjoin'] = area_df['fips'].apply(fips_to_gisjoin)

# Keep only relevant columns
area_df = area_df[['year','gisjoin', 'CENSUSAREA']]

# Re-name CENSUSAREA to area
area_df = area_df.rename(columns={'CENSUSAREA': 'area'})
```

### Step 4: Merge into one full dataset


```python
# Merge demographic and area data on gisjoin and year
demo_with_area_df = pd.merge(demographic_df, area_df, on=['year', 'gisjoin'], how='inner')

# Merge election data with demographic + area data on gisjoin and year
merged_df = pd.merge(demo_with_area_df, election_df, on=['year', 'gisjoin'], how='inner')

# Drop rows where totalvotes > population
temp = merged_df[merged_df['totalvotes'] > merged_df['population']]
bad_gisjoins = temp['gisjoin'].unique()
merged_df = merged_df[~merged_df['gisjoin'].isin(bad_gisjoins)]

# Check for nans
nans = merged_df.isna().sum()
print(nans[nans > 0])

# Check 
print(merged_df.groupby('year').size())

# Save to csv
merged_df.to_csv("../data/processed_data/full_dataset.csv", index=False)
```

    Series([], dtype: int64)
    year
    2008    3056
    2012    3056
    2016    3056
    2020    3056
    dtype: int64


### Step 5: Create a more compact dataset


```python
# id: 4 columns
id_cols = ['gisjoin', 'year', 'state', 'county']

# miscellanous: 3 columns
misc_cols = ['population', 'area', 'hispanic']

# nativity: 2 columns
nativity_cols = ['nativity_foreign_born', 'nativity_native']

# households by income: 5 columns
house_cols = [
    'households', 
    'income_10k_to_15k', 
    'income_15k_to_25k',
    'income_25k_plus', 
    'income_less_than_10k'
    ]

# labor force: 5 columns
labor_cols = [
    'labor_16_plus_in_force',
    'labor_16_plus_employed',
    'labor_16_plus_unemployed',
    'labor_16_plus_armed_forces',
    'labor_16_plus_not_in_force',
    ]

# economics: 3 columns
economic_cols = [
    'median_household_income',
    'per_capita_income',
    'poverty'
]

# race by sex: 7*2=14 columns
race_sex_cols = [
    f"{race}_{sex}" for race in ['white','black','asian','aian','nhpi','other','multi'] for sex in ['male', 'female']
    ]

compact_df = merged_df[id_cols + misc_cols + nativity_cols + house_cols + labor_cols + economic_cols + race_sex_cols]

# age by edu cols: 3*4=12 columns
age_dict = {
    '18_to_24': 'low',
    '25_to_34': 'low',
    '35_to_44': 'mid',
    '45_to_64': 'mid',
    '65_plus': 'high'
}
edu_dict = {
    'less_than_9th': 'low',
    'hs_no_diploma': 'low',
    'ged': 'mid',
    'some_college': 'mid',
    'associate': 'high',
    'bachelors': 'high',
    'graduate': 'very_high'
}

for (age, age_code) in age_dict.items():
    for (edu, edu_code) in edu_dict.items():
        compact_df[f"age_{age_code}_edu_{edu_code}"] = merged_df[[f'male_{age}_{edu}', f'female_{age}_{edu}']].copy().sum(axis=1)

# marital status: 5 columns
for status in ['divorced', 'single', 'married', 'separated', 'widowed']:
    compact_df[f"marital_{status}"] = merged_df[f"male_{status}"] + merged_df[f'female_{status}']
        
# targets: 4 columns
voter_cols = ['democrat_voter', 'republican_voter', 'other_voter']
compact_df[voter_cols] = merged_df[voter_cols].copy()
compact_df['non_voter'] = compact_df['population'] - merged_df['totalvotes']
voter_cols.append('non_voter')

# Save to csv
compact_df.to_csv("../data/processed_data/compact_dataset.csv", index=False)

print(len(compact_df.columns))
compact_df.columns.tolist()
```

    57





    ['gisjoin',
     'year',
     'state',
     'county',
     'population',
     'area',
     'hispanic',
     'nativity_foreign_born',
     'nativity_native',
     'households',
     'income_10k_to_15k',
     'income_15k_to_25k',
     'income_25k_plus',
     'income_less_than_10k',
     'labor_16_plus_in_force',
     'labor_16_plus_employed',
     'labor_16_plus_unemployed',
     'labor_16_plus_armed_forces',
     'labor_16_plus_not_in_force',
     'median_household_income',
     'per_capita_income',
     'poverty',
     'white_male',
     'white_female',
     'black_male',
     'black_female',
     'asian_male',
     'asian_female',
     'aian_male',
     'aian_female',
     'nhpi_male',
     'nhpi_female',
     'other_male',
     'other_female',
     'multi_male',
     'multi_female',
     'age_low_edu_low',
     'age_low_edu_mid',
     'age_low_edu_high',
     'age_low_edu_very_high',
     'age_mid_edu_low',
     'age_mid_edu_mid',
     'age_mid_edu_high',
     'age_mid_edu_very_high',
     'age_high_edu_low',
     'age_high_edu_mid',
     'age_high_edu_high',
     'age_high_edu_very_high',
     'marital_divorced',
     'marital_single',
     'marital_married',
     'marital_separated',
     'marital_widowed',
     'democrat_voter',
     'republican_voter',
     'other_voter',
     'non_voter']



### Step 6: Convert to probabilities and log odds


```python
# Start with id cols
df_probs = compact_df[id_cols]

# Weight cols
households_by_year = compact_df.groupby('year')['households'].transform('sum')
population_by_year = compact_df.groupby('year')['population'].transform('sum')
area_by_year = compact_df.groupby('year')['area'].transform('sum')
df_probs['P(C)'] = compact_df['population'] / population_by_year
df_probs['P(house_in_C)'] = compact_df['households'] / households_by_year
df_probs['P(area_in_C)'] = compact_df['area'] / area_by_year

# Economic columns (excluding poverty)
df_probs[economic_cols[:-1]] = compact_df[economic_cols[:-1]].copy()

# Population density
df_probs['population_density'] = compact_df['population'] / compact_df['area']

# Household income probabilities
for col in house_cols[1:]:
    df_probs[f'P({col}|house_in_C)'] = compact_df[col] / compact_df['households']

# Columns of person counts to convert to probabilities (features)
person_cols = \
    ['poverty', 'hispanic', 'nativity_native'] + \
    labor_cols[1:] + \
    race_sex_cols + \
    [col for col in compact_df.columns if col.startswith('age_')] + \
    [col for col in compact_df.columns if col.startswith('marital_')]

# Convert person count columns to probabilities
for col in person_cols:
    df_probs[f'P({col}|C)'] = compact_df[col] / compact_df['population']

# Diversity columns
def entropy(p):
    return -np.sum(p * np.log2(p + 1e-10))

race_sex_cols = [f'P({col}|C)' for col in race_sex_cols]
df_probs['race_sex_diversity'] = df_probs[race_sex_cols].apply(entropy, axis=1)

# Target probabilities
for col in voter_cols:
    df_probs[f'P({col}|C)'] = compact_df[col] / compact_df['population']

# Voting diversity
voter_cols = [f'P({col}|C)' for col in voter_cols]
df_probs['voting_diversity'] = df_probs[voter_cols].apply(entropy, axis=1)

# Voting log-odds
df_probs['log_odds_dem_rep'] = np.log((df_probs['P(democrat_voter|C)'] + 1e-10) / (df_probs['P(republican_voter|C)'] + 1e-10))
df_probs['log_odds_voter_nonvoter'] = np.log((1 - df_probs['P(non_voter|C)'] + 1e-10) / (df_probs['P(non_voter|C)'] + 1e-10))
```

```python
# Save to csv
df_probs.to_csv("../data/processed_data/probability_dataset.csv", index=False)

df_probs.columns.tolist()
```




    ['gisjoin',
     'year',
     'state',
     'county',
     'P(C)',
     'P(house_in_C)',
     'P(area_in_C)',
     'median_household_income',
     'per_capita_income',
     'population_density',
     'P(income_10k_to_15k|house_in_C)',
     'P(income_15k_to_25k|house_in_C)',
     'P(income_25k_plus|house_in_C)',
     'P(income_less_than_10k|house_in_C)',
     'P(poverty|C)',
     'P(hispanic|C)',
     'P(nativity_native|C)',
     'P(labor_16_plus_employed|C)',
     'P(labor_16_plus_unemployed|C)',
     'P(labor_16_plus_armed_forces|C)',
     'P(labor_16_plus_not_in_force|C)',
     'P(white_male|C)',
     'P(white_female|C)',
     'P(black_male|C)',
     'P(black_female|C)',
     'P(asian_male|C)',
     'P(asian_female|C)',
     'P(aian_male|C)',
     'P(aian_female|C)',
     'P(nhpi_male|C)',
     'P(nhpi_female|C)',
     'P(other_male|C)',
     'P(other_female|C)',
     'P(multi_male|C)',
     'P(multi_female|C)',
     'P(age_low_edu_low|C)',
     'P(age_low_edu_mid|C)',
     'P(age_low_edu_high|C)',
     'P(age_low_edu_very_high|C)',
     'P(age_mid_edu_low|C)',
     'P(age_mid_edu_mid|C)',
     'P(age_mid_edu_high|C)',
     'P(age_mid_edu_very_high|C)',
     'P(age_high_edu_low|C)',
     'P(age_high_edu_mid|C)',
     'P(age_high_edu_high|C)',
     'P(age_high_edu_very_high|C)',
     'P(marital_divorced|C)',
     'P(marital_single|C)',
     'P(marital_married|C)',
     'P(marital_separated|C)',
     'P(marital_widowed|C)',
     'race_sex_diversity',
     'P(democrat_voter|C)',
     'P(republican_voter|C)',
     'P(other_voter|C)',
     'P(non_voter|C)',
     'voting_diversity',
     'log_odds_dem_rep',
     'log_odds_voter_nonvoter']


