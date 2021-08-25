import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re
from scipy.optimize import curve_fit


int_regex = r'\d+'
float_regex = r'-?\d+\.\d*(?:E[+-]\d+)?'
element_regex = r'[A-Z][a-z]?'

cascade_regex = r'³\s*(' + r')\s*³\s*('.join([int_regex, float_regex, float_regex, float_regex, float_regex, float_regex, element_regex, float_regex]) + r')\s*³'
def format_ion_cascade(index, ion, energy, x, y, z, stopping_energy, atom_hit, recoil_energy):
    return int(index), int(ion), float(energy), float(x), float(y), float(z), float(stopping_energy), atom_hit, float(recoil_energy)
def parse_ion_cascade(ion_block):
    return [format_ion_cascade(m.start(), *m.groups()) for m in re.finditer(cascade_regex, ion_block)]


exyz_regex = r'\s*(' + r')\s*\s*('.join([int_regex, float_regex, float_regex, float_regex, float_regex, float_regex, float_regex]) + r')\s*'
def format_exyz_cascade(index, ion, energy, x, y, z, stopping_power, recoil_energy_lost):
    return int(index), int(ion), float(energy), float(x), float(y), float(z), float(stopping_power), float(recoil_energy_lost)
def parse_exyz_cascade(ion_block):
    return [format_exyz_cascade(m.start(), *m.groups()) for m in re.finditer(exyz_regex, ion_block)]


range3D_regex = r'\s*(' + r')\s*\s*('.join([int_regex, float_regex, float_regex, float_regex]) + r')\s*'
def format_Range3D_cascade(index, ion, x, y, z):
    return int(index), int(ion), float(x), float(y), float(z)
def parse_Range3D_cascade(ion_block):
    return [format_Range3D_cascade(m.start(), *m.groups()) for m in re.finditer(range3D_regex, ion_block)]


lateral_regex = r'\s*(' + r')\s*\s*('.join([float_regex, float_regex, float_regex, float_regex, float_regex]) + r')\s*'
def format_Lateral_cascade(index, depth, lat_proj_range, proj_straggling, lat_radial, radial_straggling):
    return int(index), float(depth), float(lat_proj_range), float(proj_straggling), float(lat_radial), float(radial_straggling)
def parse_Lateral_cascade(ion_block):
    return [format_Lateral_cascade(m.start(), *m.groups()) for m in re.finditer(lateral_regex, ion_block)]

        
# classes for each file
class Collision_Info:
    ## energy: keV
    ## depth/x,y,z: Angstrom
    ## electronic stopping power: eV/Angstrom
    ## atom hit: element symbol
    ## recoil energy: eV (energy transferred to atom)
    '''
    The Collision file tracks all the important details of the simulation run.
    I used the 'Ion Distribution and Quick Calculation of Damage'
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        
        with open(file_name, encoding='latin1') as f:
            self.collision = f.read()
        self.raw_data = parse_ion_cascade(self.collision)
        self.df_labels = ['index', 'ion', 'energy', 'x', 'y', 'z', 'stopping_energy', 'atom_hit', 'recoil_energy']
        self.data = pd.DataFrame(self.raw_data, columns=self.df_labels).set_index('index')

        f.close()


class EXYZ_Info:
    ## x,y,z: Angstrom
    ## energy: keV
    ## electronic stopping power: eV/Angstrom
    ## energy lost to last recoil: eV
    '''
    The EXYZ file tracks 3d position at intervals in eV.
    How I set it up (starting energy ---> increments):
        <100keV --> 1,000eV
        100-400keV --> 10,000eV
        400-700keV --> 50,000eV
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        
        with open(file_name, encoding='latin1') as f:
            self.collision = f.read()
        self.raw_data = parse_exyz_cascade(self.collision)
        self.df_labels = ['index', 'ion', 'energy', 'x', 'y', 'z', 'stopping_power', 'recoil_energy_lost']
        self.data = pd.DataFrame(self.raw_data, columns=self.df_labels).set_index('index')

        f.close()


class Range3D_Info:
    ## x,y,z: Angstrom
    '''
    The RANGE_3D file recoids the final positions of the ions
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        
        with open(file_name, encoding='latin1') as f:
            self.collision = f.read()
        self.raw_data = parse_Range3D_cascade(self.collision)
        self.df_labels = ['index', 'ion', 'x', 'y', 'z']
        self.data = pd.DataFrame(self.raw_data, columns=self.df_labels).set_index('index')

        f.close()


class Lateral_Info:
    ## depth, lat_proj_range, proj_straggling, lat_radial, radial_straggling: Angstrom
    '''
    This file is a summary of lateral range. Defn of straggling is sqrt of the variance
    These are the equations used:
        depth (R_p) : <x>
        lat_proj_range (R_y): < |y| >
        proj_straggling (sigma): [ (sum(x**2) / N) - R_p**2   ]**(1/2) where N is num of ions, and R_p is <x>
        lat_radial (sigma_r): sum( (y**2 + z**2)**1/2 )/N where N is num of ions
    
    The data points for these equations use the final x,y,z positions
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        
        with open(file_name, encoding='latin1') as f:
            self.collision = f.read()
        self.raw_data = parse_Lateral_cascade(self.collision)
        self.df_labels = ['index', 'depth', 'lat_proj_range', 'proj_straggling', 'lat_radial', 'radial_straggling']
        self.data = pd.DataFrame(self.raw_data, columns=self.df_labels).set_index('index')

        f.close()


# cascade class holds each txt thing
class Cascade:
    def __init__(self, num_sim, sim_dir, energy_domain, num_ions, fixed_lims=False, collision = True, exyz=True, lateral=True, range3d=True):
        '''
        num_sim is the number simulation it is [Ar1, Ar2, ...]
        sim_dir is the simulation directory that holds the directories
        energy domain is the entire domain of energies I used 
        num_ions is the number of ions used in the simulation

        '''
        self.num_sim = num_sim
        self.collision_file = sim_dir + f"Collisions/Ar{self.num_sim}_COLLISON.txt"
        self.exyz_file = sim_dir + f"EXYZ/Ar{self.num_sim}_EXYZ.txt"
        self.lateral_file = sim_dir + f"Lateral/Ar{self.num_sim}_LATERAL.txt"
        self.range3d_file = sim_dir + f"Range3D/Ar{self.num_sim}_RANGE_3D.txt"
        self.starting_E = energy_domain[self.num_sim-1] #keV
        self.max_lim = 0.25
        if self.starting_E > 50 and self.starting_E < 100:
            self.max_lim = 0.5
        elif self.starting_E > 100 and self.starting_E < 300:
            self.max_lim = 1
        elif self.starting_E > 300 and self.starting_E < 600:
            self.max_lim = 2
        elif self.starting_E > 600:
            self.max_lim = 3
        if fixed_lims==True:
            self.max_lim = 3
        self.num_ions = num_ions
        
        if collision:
            self.collision = Collision_Info(self.collision_file)
        if exyz:
            self.exyz = EXYZ_Info(self.exyz_file)
        if lateral:
            self.lateral = Lateral_Info(self.lateral_file)
        if range3d:
            self.range3d = Range3D_Info(self.range3d_file)



# simulation class
class Simulation:
    def __init__(self, num_sims, sim_dir, num_ions, low_E = 10, high_E = 1000, collision=True, exyz=True, lateral=True, range3d=True):
        assert exyz or lateral or range3d
        
        self.low_E, self.high_E = low_E, high_E
        self.num_ions = num_ions
        self.collision, self.exyz, self.lateral, self.range3d = collision, exyz, lateral, range3d
        self.num_sims = num_sims
        self.sim_dir = sim_dir
        self.energy_domain = np.logspace(np.log10(self.low_E), np.log10(self.high_E), num_sims)
        self.runs = {}
        
        if collision:
            self.runs_collisions = {}
        if self.exyz:
            self.runs_exyz = {}
        if self.lateral:
            self.runs_lateral = {}
        if self.range3d:
            self.runs_range3d = {}
        
        for run_num in range(self.num_sims):
            curr_cascade = Cascade(run_num+1, self.sim_dir, self.energy_domain, self.num_ions,
                                       collision=self.collision, exyz=self.exyz, 
                                       lateral=self.lateral, range3d=self.range3d)
            self.runs[run_num] = curr_cascade
            self.runs_collisions[run_num] = curr_cascade.collision
            if self.exyz:
                self.runs_exyz[run_num] = curr_cascade.exyz
            if self.lateral:
                self.runs_lateral[run_num] = curr_cascade.lateral
            if self.range3d:
                self.runs_range3d[run_num] = curr_cascade.range3d
            
            print(f'done assigning info for run {run_num+1}')




### functions
def plot_traj(cascade, plot_every_n_part, axes, save_fig = False, file_dest = "", hist=False, bins_num=50):
    '''
    straight up just plot the x-y coords for this collision run
    '''
    x = axes[0]
    y = axes[1]
    if not hist:
        for i in range(cascade.num_ions):
            if i%plot_every_n_part != 0:
                continue

            mask = cascade.collision.data['ion']==i
            plt.plot(cascade.collision.data[mask][x]/1e7, cascade.collision.data[mask][y]/1e7, alpha=0.7, linewidth=0.75)

            mask2 = cascade.range3d.data['ion']==i
            plt.scatter(cascade.range3d.data[mask2][x]/1e7, cascade.range3d.data[mask2][y]/1e7)
        plt.xlim(0, cascade.max_lim)
        if x=="y" or x=="z":
            plt.xlim(-cascade.max_lim/2, cascade.max_lim/2)
        plt.ylim(-cascade.max_lim/2, cascade.max_lim/2)
    else:
        plt.hist2d(cascade.range3d.data[x]/1e7, cascade.range3d.data[y]/1e7, bins=bins_num)
        plt.colorbar()
    
    plt.title(f'{x.capitalize()} vs {y.capitalize()} with Starting Ion Energy {cascade.starting_E:.2f} keV')
    plt.xlabel(f'{x.capitalize()} [mm]')
    plt.ylabel(f'{y.capitalize()} [mm]')

    if save_fig:
        plt.savefig(file_dest)
        
    plt.show()


def plot_ESE(cascade, plot_every_n_part, save_fig = False, file_dest = "", hist=False, bins_num=50):
    '''
    plot the ranges (final x values) as a function of E
    '''
    if not hist:
        for i in range(cascade.num_ions):
            if i%plot_every_n_part != 0:
                continue

            mask1 = cascade.exyz.data['ion']==i
            mask2 = cascade.exyz.data['energy']!=0
            mask3 = cascade.exyz.data['stopping_power']!=0
            mask = mask1 & mask2 & mask3
            # have to convert eV/A to keV/microm
            stopping_power = cascade.exyz.data[mask]['stopping_power'] #eV/A
            stopping_power /= 1e3 #keV/A
            stopping_power *= 1e4 #keV/microm
            
            plt.plot(cascade.exyz.data[mask]['energy'][:-1], stopping_power[:-1], alpha=0.7, linewidth=0.75)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e0, cascade.starting_E)
        plt.ylim(1e-2, 1e0)
    else:
        bins_logd = [10 ** np.linspace(np.log10(1e0), np.log10(cascade.starting_E), bins_num), 
                    10 ** np.linspace(np.log10(1e-2), np.log10(1e0), bins_num)]
        plt.hist2d(cascade.exyz.data['energy'], cascade.exyz.data['stopping_power']*1e1, bins=bins_logd, norm=matplotlib.colors.LogNorm(vmin=1e1),
                range=[[1e0, cascade.starting_E], [1e-2, 1e0]])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e0, cascade.starting_E)
        plt.ylim(1e-2, 1e0)
        plt.colorbar()
    
    plt.title(f'Energy vs Stopping Power with initial energy {cascade.starting_E:.2f} keV')
    plt.xlabel('Energy [$keV$]')
    plt.ylabel('Stopping Power [$keV/\mu m$]')
    
    if save_fig:
        plt.savefig(file_dest)
    
    plt.show()



def gauss(x,mu,sigma,A):
    norm = A/(np.sqrt(2*np.pi)*sigma)
    exp  = np.exp(-((x-mu)**2)/(2*sigma*sigma))
    return norm * exp
    
def FitGauss(vals, binned, bin_edges):
    variance = np.std(vals)
    mean = np.mean(vals)

    bcenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
    guess = [mean,variance/2.,np.max(binned)]
    
    popt,popv = curve_fit(gauss,bcenters,binned,p0=guess)
    pope = np.sqrt(np.diag(popv))
    
    return popt,pope 

def plot_dist(cascade, save_fig = False, file_dest = "", bins_num=100, transverse=False, radial=False, gauss_fit=False, gauss_vals = '', plot=True):
    '''
    plot the distribution of the final x positions
    '''
    histtype='bar'
    alpha=1.0
    if transverse:
        histtype='stepfilled'
        alpha=0.3

    x_arr = cascade.range3d.data['x'] / 1e7
    x_hist, x_bin_edges = np.histogram(x_arr, bins=bins_num)
    if plot:
        plt.hist(x_arr, x_bin_edges, histtype = histtype, alpha=alpha, label='Final X position')
    if gauss_fit:
        x_popt, x_pope = FitGauss(x_arr, x_hist, x_bin_edges)
        x_mu, x_sigma, x_A = x_popt[0], x_popt[1], x_popt[2]
        if plot:
            plt.plot(x_bin_edges, gauss(x_bin_edges,*x_popt),'r--', label=f"X Gauss $\mu$={x_mu:.2} $\sigma$ = {x_sigma:.2}")
        if gauss_vals=='x':
            return x_mu, x_sigma

    if transverse:
        y_arr = cascade.range3d.data['y'] / 1e7
        y_hist, y_bin_edges = np.histogram(y_arr, bins=bins_num)
        if plot:
            plt.hist(y_arr, y_bin_edges, histtype = histtype, alpha=alpha, label='Final Y position')

        z_arr = cascade.range3d.data['z'] / 1e7
        z_hist, z_bin_edges = np.histogram(z_arr, bins=bins_num)
        if plot:
            plt.hist(z_arr, z_bin_edges, histtype = histtype, alpha=alpha, label='Final Z position')
        
        if gauss_fit:
            y_popt, y_pope = FitGauss(y_arr, y_hist, y_bin_edges)
            y_mu, y_sigma, y_A = y_popt[0], y_popt[1], y_popt[2]
            if plot:
                plt.plot(y_bin_edges, gauss(y_bin_edges,*y_popt),'k--', label=f"Y Gauss $\mu$={y_mu:.2} $\sigma$ = {y_sigma:.2}")

            z_popt, z_pope = FitGauss(z_arr, z_hist, z_bin_edges)
            z_mu, z_sigma, z_A = z_popt[0], z_popt[1], z_popt[2]
            if plot:
                plt.plot(z_bin_edges, gauss(z_bin_edges,*z_popt),'k-*', label=f"Z Gauss $\mu$={z_mu:.2} $\sigma$ = {z_sigma:.2}")
    
    if radial:
        y_arr = np.array(cascade.range3d.data['y']) / 1e7
        z_arr = np.array(cascade.range3d.data['z']) / 1e7
        r_arr = np.sqrt(y_arr**2 + z_arr**2)
        r_hist, r_bin_edges = np.histogram(r_arr, bins=bins_num)

        if plot:
            plt.hist(r_arr, r_bin_edges, histtype = histtype, alpha=alpha, label='Final Radial position')
        
        if gauss_fit:
            r_popt, r_pope = FitGauss(r_arr, r_hist, r_bin_edges)
            r_mu, r_sigma, r_A = r_popt[0], r_popt[1], r_popt[2]
            if plot:
                plt.plot(r_bin_edges, gauss(r_bin_edges,*y_popt),'k--', label=f"Radial Gauss $\mu$={y_mu:.2} $\sigma$ = {y_sigma:.2}")
            if gauss_vals == 'r':
                return r_mu, r_sigma

    if plot:
        plt.legend()
        plt.xlabel("Depth [mm]")
        plt.ylabel("Num. of Ions")
        plt.title(f'Projected Ion Ranges with Starting Energy {cascade.starting_E:.2f} keV')
        
        if save_fig:
            plt.savefig(file_dest)
        
        plt.show()


def plot_dist_path(cascade, save_fig = False, file_dest = "", bins_num=100):
    '''
    add up the changes in each data points as dr to find the total Euclidean
    path length. plot distribution of lengths taken
    '''

    distances = np.zeros(cascade.num_ions)
    for i in range(cascade.num_ions):
        mask = cascade.collision.data['ion']==i
        x_arr = np.array(cascade.collision.data[mask]['x']) / 1e7
        y_arr = np.array(cascade.collision.data[mask]['y']) / 1e7
        z_arr = np.array(cascade.collision.data[mask]['z']) / 1e7

        dx_arr = x_arr[1:] - x_arr[:-1]
        dy_arr = y_arr[1:] - y_arr[:-1]
        dz_arr = z_arr[1:] - z_arr[:-1]

        dist_traveled = np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2) # dist between each step
        distances[i] = np.sum(dist_traveled) # sum the distances

    hist, bin_edges = np.histogram(distances, bins=bins_num)
    plt.hist(distances, bin_edges)

    plt.xlabel("Total Length of Path Taken [mm]")
    plt.ylabel("Num. of Ions")
    plt.title(f'Path Length with initial energy {cascade.starting_E:.2f} keV')
    
    if save_fig:
        plt.savefig(file_dest)
    
    plt.show()


def plot_sim_range_E(simulation, save_fig = False, file_dest = ""):
    energy = simulation.energy_domain
    ranges = np.zeros(simulation.num_sims * simulation.num_ions)
    
    for run_num, collision in sorted(simulation.runs.items()):
        x_arr = np.array(collision.range3d.data['x']) / 1e4
        plt.scatter(energy[run_num-1], x_arr)
    
    plt.xlim(1e1,1e3)
    plt.ylim(1e1, 1e4)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Energy [keV]")
    plt.ylabel("Projected Ranges [$\mu m$]")
    plt.title('Projected range vs Energy')

    if save_fig:
        plt.savefig(file_dest)
    
    plt.show()