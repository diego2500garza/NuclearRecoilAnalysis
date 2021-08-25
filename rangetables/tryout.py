import pandas as pd

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


info_header = ['energy', 'dEdx Elec', 'dEdx Nucl', 'Proj. range', 'long. straggling', 'lateral straggling']
df = pd.read_csv('SRIM.csv',header=None,names=info_header) 


energies = df['energy']
new_energies = []


for energy in energies:
    split_energy = energy.split(' ')
    unit = split_energy[-2]
    for item in split_energy:
        if is_float_try(item):
            energy = float(item)
            
            if unit == 'MeV':
                energy *= 1000
            new_energies.append(energy)
    

ranges = df['lateral straggling']
num_ranges = len(ranges)
new_proj = []
new_long = []
new_lat = []

for i in range(num_ranges):
    #proj range
    proj_range_split = df['Proj. range'][i].split()
    proj_range = float(proj_range_split[0])
    proj_range_units = proj_range_split[1]
    
    if(proj_range_units == 'mm'):
        proj_range *= 1000
    new_proj.append(proj_range)
    
    long_range_split = df['long. straggling'][i].split()
    long_range = float(long_range_split[0])
    long_range_units = long_range_split[1]
    
    if(long_range_units == 'mm'):
        long_range *= 1000
    new_long.append(long_range)
    
    lat_range_split = df['lateral straggling'][i].split()
    lat_range = float(lat_range_split[0])
    lat_range_units = lat_range_split[1]
    
    if(lat_range_units == 'mm'):
        lat_range *= 1000
    new_lat.append(lat_range)
    

new_df = pd.DataFrame()
new_df[info_header[0]] = new_energies
new_df[info_header[1]] = df["dEdx Elec"]
new_df[info_header[2]] = df["dEdx Nucl"]
new_df[info_header[3]] = new_proj
new_df[info_header[4]] = new_long
new_df[info_header[5]] = new_lat

new_df.to_csv('new_SRIM.csv')