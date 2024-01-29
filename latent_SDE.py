# Joshua Fagin
# Ji Won Park: older vserion of code https://github.com/jiwoncpark/magnify 
# Also inspired by https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

import logging 
# I also use logging.warning() along with print() for so I can see the output in the job.s.e files in linux. 
# Otherwise, sometimes the print() statements don't show up in the job.s.o files.
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from latent_sde.model import LatentSDE
import torchsde
from glob import glob
from astropy.io import fits
from losses.gaussian_nll import FullRankGaussianNLL
import corner
import imageio.v2 as imageio
from scipy.fftpack import rfft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import convolve
import scipy.stats as stats
from astroML.time_series import generate_damped_RW # Just used to generate the DRW
from scipy.constants import c
import time
from multiprocessing import Pool, Process
from sys import platform
import gpytorch  # For GPR
import botorch   # For GPR

# For debugging
#torch.autograd.set_detect_anomaly(True)

# to ignore numpy warnings
np.seterr(all='ignore')

# So that we have the same validation set each time
np.random.seed(0)

size = 13
plt.rc('font', size=size)          
plt.rc('axes', titlesize=size)     
plt.rc('axes', labelsize=size)   
plt.rc('xtick', labelsize=size)   
plt.rc('ytick', labelsize=size)    
plt.rc('legend', fontsize=size)    
plt.rc('figure', titlesize=size) 

tick_length_major = 7
tick_length_minor = 3
tick_width = 1
#If plot then display the plots. Otherwise they are only saved
plot = False
#What type of file to save figures as e.g pdf, eps, png, etc.
save_file_format = 'pdf' 
#Path to directory where to save the results
save_path = 'results' 

#make directory to save results
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/recovery', exist_ok=True)

#The different parameters to predict 
give_redshift = False 
parameters_keys = ['SPIN','CHEIGHT','Z_Q','INC_ANG','BETA','eddingtons','MASS','log_tau','SFinf'] # Mass is actually log_10(M/M_sun)
#parameters_keys = ['MASS','log_tau','SFinf']
if give_redshift:
    # Redshift is not a parameter to predict since we are giving it to the model if give_redshift is True
    if 'Z_Q' in parameters_keys:
        parameters_keys.remove('Z_Q')
    
parameters_keys_drw = ['log_tau','SFinf'] 

#This is used for plotting. Variables with units for axis plotting.
plotting_labels = dict()
plotting_labels['MASS'] = r"$\log_{10}\left(M/M_\odot\right)$"
plotting_labels['SPIN'] = r'$a$'
plotting_labels['INC_ANG'] = r'$\theta_{\mathrm{inc}}$ [deg]'
plotting_labels['CHEIGHT'] = r'$h$ [$R_g$]'
plotting_labels['log_tau'] = r'$\log_{10}(\tau/\mathrm{day})$'
plotting_labels['SFinf'] = r'$\mathrm{SF}_\infty$ [mag]'
plotting_labels['Z_Q'] = r'$z$'
plotting_labels['BETA'] = r'$\beta$'
plotting_labels['eddingtons'] = r'$\lambda_{\mathrm{Edd}}$'

#This is used for plotting. Variables without units.
plotting_labels_no_units = dict()
plotting_labels_no_units['MASS'] = r"$\log_{10}\left(M/M_\odot\right)$"
plotting_labels_no_units['SPIN'] = r'$a$'
plotting_labels_no_units['INC_ANG'] = r'$\theta_{\mathrm{inc}}$'
plotting_labels_no_units['CHEIGHT'] = r'$h$'
plotting_labels_no_units['log_tau'] = r'$\log_{10}(\tau/\mathrm{day})$'  #r'$\log_{10}(\tau)$' #changed to tau/day
plotting_labels_no_units['SFinf'] = r'$\mathrm{SF}_\infty$'
plotting_labels_no_units['Z_Q'] = r'$z$'
plotting_labels_no_units['BETA'] = r'$\beta$'
plotting_labels_no_units['eddingtons'] = r'$\lambda_{\mathrm{Edd}}$'

#Define the max and min ranges of the parameter space
min_max_dict = dict()
min_max_dict['MASS'] = (7.0,10.0) 
min_max_dict['SPIN'] = (-1.0,1.0)
min_max_dict['INC_ANG'] = (0.0,80.0) 
min_max_dict['CHEIGHT'] = (10.0,50.0)
min_max_dict['log_tau'] = (-0.5,3.5) 
min_max_dict['SFinf'] = (0.0,0.5)
min_max_dict['Z_Q'] = (0.1,6.0)
min_max_dict['BETA'] = (0.3, 1.0)
min_max_dict['eddingtons'] = (0.01,0.3)

days_per_year = 365
hours_per_day = 24
num_years = 11.0

bandpasses=list('ugrizy')
#effective frequency of LSST bands
lambda_effective = np.array([3671, 4827, 6223, 7546, 8691, 9712]) #in Angstrom
lambda_effective = 1e-10*lambda_effective #in meter
freq_effective = c/lambda_effective #in Hz

mag_mean = [21.52,21.06,20.80,20.64,20.49,20.41]
mag_std = [1.29,1.08,1.01,0.98,0.95,0.94]

mag_mean = float(np.array(mag_mean).mean())
mag_std = float(np.array(mag_std).mean())


path = "TFs"
file_name_for_cadence = "cadence"
num_training_LC = 100

#Fraction of the total data set used as validation set.
validation_rate = 0.1
test_rate = 0.1

num_validation_LC = int(validation_rate*num_training_LC)
num_test_LC = int(test_rate*num_training_LC)
num_LC = num_training_LC + num_validation_LC + num_test_LC

file_names = glob(f'{path}/*.fits')
total_files = len(file_names)

assert (total_files > 0), f"No files found in the directory path: {path}"

def load_transfer_functions(file_path):
    with fits.open(file_path,memmap=False) as hdul:

        transfer_function = hdul[0].data

        parameter = dict()
        for par in parameters_keys:
            if par not in parameters_keys_drw:
                parameter[par] = hdul[0].header[par]
        if give_redshift:
            parameter['Z_Q'] = hdul[0].header['Z_Q']


    # either 'days' or 'hours'
    units = hdul[0].header['units'].lower().strip()

    assert(np.isnan(transfer_function).sum() == 0), f"{np.isnan(transfer_function).sum()} NaNs in transfer function: {file_path}"
    return transfer_function, parameter, units


train_files = file_names[:int((1-2*validation_rate)*total_files)]
val_files = file_names[int((1-2*validation_rate)*total_files):int((1-validation_rate)*total_files)]
test_files = file_names[int((1-validation_rate)*total_files):]

file_names_train = []
for i in range(num_training_LC):
    if i < len(train_files):
        file_name = train_files[i]
    else:
        file_name = np.random.choice(train_files) # Choose from the training set
    #get random number
    rand_num = np.random.randint(100000,999999) # Can use more digits if needed
    # Add to file name so there are more combinations for random seeds
    # This is so we can have a fixed validation set based on the file names
    file_names_train.append(file_name+'_'+str(rand_num))


file_names_val = []
for i in range(num_validation_LC):
    if i < len(val_files):
        file_name = val_files[i]
    else:
        file_name = np.random.choice(val_files) # Choose from the training set
    #get random number
    rand_num = np.random.randint(100000,999999) # Can use more digits if needed
    # Add to file name so there are more combinations for random seeds
    # This is so we can have a fixed validation set based on the file names
    file_names_val.append(file_name+'_'+str(rand_num))

file_names_test = []
for i in range(num_test_LC):
    if i < len(test_files):
        file_name = test_files[i]
    else:
        file_name = np.random.choice(test_files) # Choose from the training set
    #get random number
    rand_num = np.random.randint(100000,999999) # Can use more digits if needed
    # Add to file name so there are more combinations for random seeds
    # This is so we can have a fixed validation set based on the file names
    file_names_test.append(file_name+'_'+str(rand_num))

del file_names 

cadence = 1.0 #Time spacing of data set in days. We nominally use 1 day. The light curve has int((num_years*days_per_year)/cadence) time steps.

n_params = len(parameters_keys)
length_x = round(num_years*days_per_year/cadence)
output_dim = 2*len(bandpasses) #num bands and uncertainty
num_bands = len(bandpasses)    #num bands

days = cadence*np.arange(length_x) 

cadence_files = glob(f'{file_name_for_cadence}/*.dat')
num_cadences = len(cadence_files)//len(bandpasses)
del cadence_files

def make_light_curve(transfer_function, units, log_tau, SFinf, mean_mag, z, random_state, plot=False):
    """
    This function generated a light curve by convolving it with a transfer function. 

    transfer_function: numpy array of shape (length_t, num_bands), transfer function used as a kernel to convolve with DRW.
    units: 'days' or 'hours' representing the units of the transfer function.
    log_tau: float, damping time scale of the DRW.
    SFint: float, amplitude of the DRW.
    mean_mag: float, mean magnitude of the light curve.
    random_state: int or None, random state used to generate the DRW.
    plot: bool, if True, plot the light curve and DRW. For debugging purposes.

    return: numpy array of shape (length_t, num_bands), light curve.
    """

    tau = 10.0**log_tau

    # Extra time points to make sure the convolution is done correctly at the edges
    if units == 'days':
        extra_time = transfer_function.shape[1]
    else:
        extra_time = transfer_function.shape[1]/hours_per_day

    num_days = int(num_years*days_per_year)

    t_hourly_extra = np.linspace(-extra_time,num_days,int(extra_time*hours_per_day)+int(num_days*hours_per_day))
    
    t_hourly = np.linspace(0,num_days,int(num_days*hours_per_day))
    t = np.linspace(0,num_days,round(float(num_days)/cadence))

    # generate a damped random walk with astroML in units of mag
    # DRW is set at z = 0 so that the redshift only affects the transfer functions
    DRW = generate_damped_RW(t_hourly_extra, xmean = mean_mag, tau = tau, SFinf = SFinf, z = z, random_state = random_state)
    DRW = mag_to_flux(DRW) # convert to flux in units of erg/s/cm^2/Hz
    DRW = np.mean(freq_effective)*DRW # convert to flux in units of erg/s/cm^2

    light_curve_hourly = np.zeros((len(t_hourly),len(bandpasses)))
    light_curve = np.zeros((len(t),len(bandpasses))) 

    for i in range(len(bandpasses)):
        #normalize the transfor function
        
        tf = transfer_function[i,:]
        if units == 'days':
            # convert the transfer function to hours from days using interpolation
            tf = np.interp(np.arange(tf.shape[0]*hours_per_day)/hours_per_day,np.arange(tf.shape[0]),tf) 

        tf = tf/np.sum(tf) #normalize the transfer function so that the integral is 1

        conv = convolve(DRW,tf,mode='valid')
        t_hourly = t_hourly_extra[len(tf)-1:]
        light_curve[:,i] = interp1d(t_hourly,conv)(t)

    #convert the light curve to from flux to flux per frequency
    light_curve = light_curve/freq_effective #convert from erg/s/cm^2 to erg/s/cm^2/Hz
    
    #convert the light curve to magnitude
    light_curve = flux_to_mag(light_curve) #convert from erg/s/cm^2/Hz to mag

    # plot the DRW and then the convolved light curve for each band
    if plot:
        for i in range(len(bandpasses)):
            plt.plot(transfer_function[i,:],label=bandpasses[i])
        min_val = np.argmax((transfer_function[0,:]>0.0001)+np.array(range(transfer_function.shape[-1])) * 1e-5)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.xlim(0,min_val)
        plt.legend()
        plt.xlabel(f'time [{units}]')
        plt.ylabel('density')
        plt.savefig('transfer_function.pdf',bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 3))
        for i in range(len(bandpasses)):
            plt.plot(t,light_curve[:,i],label=bandpasses[i])
        #DRW_mag = flux_to_mag(DRW/freq_effective.mean())
        #DRW_mag = DRW_mag - DRW_mag.mean()+light_curve.max()+0.05
        #plt.plot(t_hourly_extra,DRW_mag,color='black',label='DRW')
        plt.gca().invert_yaxis()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.xlim(t.min(),t.max())
        plt.legend(loc='upper left')
        plt.xlabel('time [days]')
        plt.ylabel('magnitude')
        plt.savefig('DRW_light_curve.pdf',bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return light_curve

def load_LC(file_path,random_state):
    transfer_function, params, units = load_transfer_functions(file_path)

    log_tau = np.random.uniform(min_max_dict["log_tau"][0],min_max_dict["log_tau"][1]) 
    SFinf = np.random.uniform(min_max_dict["SFinf"][0],min_max_dict["SFinf"][1])

    params["log_tau"] = log_tau 
    params["SFinf"] = SFinf 
    z = params["Z_Q"]

    # Choose the mean magnitude of the light curve
    mean_mag = np.random.normal(mag_mean, mag_std)

    LC = make_light_curve(transfer_function, units, log_tau, SFinf, mean_mag, z, random_state)

    return LC, params

def logit(x,numpy=False):
    assert(type(numpy)==bool)
    if numpy:
        return np.log(x/(1.-x))
    else:
        return torch.log(x/(1.-x))

def expit(x,numpy=False):
    assert(type(numpy)==bool)
    if numpy:
        return 1./(1.+np.exp(-x))
    else:
        return 1./(1.+torch.exp(-x))

def LSST_photometric_noise(mag,m_5s,band_num):
    #https://arxiv.org/pdf/2203.09540.pdf good paper for reference
    #They assume sigma_sys = 0.004 mag since its expected sigma_sys < 0.005 mag

    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]
    gamma = gammas[band_num]

    x = 10.0 ** (0.4 * (mag - m_5s))
    sigma_rand = np.sqrt((0.04 - gamma)*x + gamma*(x**2))

    return sigma_rand

def flux_to_mag(flux):
    """
    This function converts a flux in units of  to a magnitude in units of AB mag.

    flux: flux per frequency in units of erg/s/cm^2/Hz 
    return: magnitude in units of AB mag
    """
    flux = np.clip(flux,1e-50,1e10)
    mag = -2.5*np.log10(flux) - 48.60
    return np.clip(mag,14,28)

def mag_to_flux(mag):
    """
    This function converts a magnitude in units of AB mag to a flux in units of erg/s/cm^2/Hz.

    mag: magnitude in units of AB mag
    returns: flux per frequency in units of erg/s/cm^2/Hz
    """
    return 10.0**(-0.4*(mag+48.60))

def delta_mag_to_flux(delta_mag,flux):
    """
    This function converts a delta magnitude in units of AB mag to a flux in units of erg/s/cm^2/Hz.
    This is an approximation.

    delta_mag: delta magnitude in units of AB mag
    returns: flux per frequency in units of erg/s/cm^2/Hz
    """
    return (np.log(10)/2.5)*delta_mag*flux

def delta_flux_to_mag(delta_flux,flux):
    """
    This function converts a delta magnitude in units of AB mag to a flux in units of erg/s/cm^2/Hz.
    This is an approximation.

    delta_flux: delta flux in units of erg/s/cm^2/Hz
    returns: delta magnitude in units of AB mag
    """
    return (2.5/np.log(10))*(delta_flux/flux)

def get_observed_LC(LC,cadence_index,cadence):
    """
    This function takes in a light curve at a fixed cadence and returns an observed light curve
    using LSST sampling and photometric noise.

    LC: numpy array, light curve with fixed cadence in units of magnitude
    cadence_index: int, index of cadence file to use
    cadence: int, cadence of light curve in days

    returns: numpy array, observed light curve, numpy array, photometric noise
    """
    JD_min = 60218
    time_list = []
    m_5s_list = []

    for i,band in enumerate(bandpasses):
        file = f"{file_name_for_cadence}/sample_{cadence_index}_band_{band}_dates.dat"
        time = np.loadtxt(file, usecols=0)
        m_5s = np.loadtxt(file, usecols=1)

        # add random shift in the time spacing. This is to avoid the season gaps all aligning.
        time -= JD_min

        max_obs = LC.shape[0]
        time = (time/cadence).round().astype(int)
        # Incase we use less than the full 10 years for some reason
        #m_5s = m_5s[time <= max_obs] 
        #time = time[time <= max_obs]
        time_list.append(time)
        m_5s_list.append(m_5s)

    min_time = np.min([time_list[i].min() for i in range(len(time_list))])
    max_time = np.max([time_list[i].max() for i in range(len(time_list))])
    #print(min_time,max_time,LC.shape[0])
    # get number of observations
    #print(np.sum([len(time_list[i]) for i in range(len(time_list))]))
    time_shift = np.random.randint(-min_time,LC.shape[0]-(max_time+1))
    #print(time_shift)
    for i,band in enumerate(bandpasses):
        time_list[i] += time_shift

        #print(time_list[i])
        #print(time_shift)
        #print()
    #print(time_list)
    #assert(np.min([time_list[i].min() for i in range(len(time_list))]) >= 0)
    #assert(np.max([time_list[i].max() for i in range(len(time_list))]) <= LC.shape[0])
    LC_obs = np.zeros(LC.shape)
    stdev = np.zeros(LC.shape)

    for i in range(len(bandpasses)):
        time = time_list[i]
        m_5s = m_5s_list[i]
        
        sigma_list = []
        for j,t in enumerate(time):
            mag = LC[t]
            sigma = LSST_photometric_noise(mag,m_5s[j],i)
            sigma_list.append(sigma)
        sigma_list = np.array(sigma_list)

        time_unique,index_unique =  np.unique(time,return_index=True)
        sigma_unique = []
        for j in range(len(index_unique)):
            if j+1 < len(index_unique):
                sigma_list_at_t = sigma_list[index_unique[j]:index_unique[j+1]]
            else:
                sigma_list_at_t = sigma_list[index_unique[j]:]

            # combine using variance weighting by the precision
            new_sigma = 1/np.sqrt(np.sum(1.0/sigma_list_at_t**2))
            sigma_unique.append(new_sigma)

        sigma_unique = np.array(sigma_unique)

        #adding systematic and rand errors in quadrature to the photometric noise
        sigma_sys = 0.005
        sigma_unique = np.sqrt(sigma_sys**2 + sigma_unique**2)

        for t,sigma in zip(time_unique,sigma_unique):
            stdev[t,i] = sigma
            LC_obs[t,i] = LC[t,i] + np.random.normal(0.0,sigma)

    return LC_obs, stdev

def build_data_set(file_name, mag_mean, cadence, augment, save_true_LC):
    """
    This function builds a data set for a single light curve. Used by the Pytorch dataloader.

    file_name: string, name of file to load in
    mag_mean: numpy array, mean magnitude of light curve
    cadence: int, cadence of light curve
    augment: bool, whether to augment the data set or not
    save_true_LC: bool, whether to save the true light curve or not

    returns: numpy array, observed light curve, numpy array, photometric noise, numpy array, true light curve
    """

    if augment:
        #random seed each time it is called. This does data augmentation during training.
        current_time = time.time()
        seed = int(1000000*(current_time-int(current_time)))
        np.random.seed(seed)
    else:
        #get random seed from filename so the seed is the same each time its called
        #print(file_name)
        #seed = sum(map(ord, file_name))
        seed = int(file_name[-6:])
        np.random.seed(seed)

    #load in the LC data

    LC , params = load_LC(file_name[:-7],seed) # get rid of the extra random 4 digits we added
    max_time = LC.shape[0]

    # Get parameters we want to predict and scale them from 0 to 1
    params_list = [] 
    for par in parameters_keys:
        value = params[par]
        #scale all the parameters from 0 to 1
        scaled_value = (value-min_max_dict[par][0])/(min_max_dict[par][1]-min_max_dict[par][0])
        params_list.append(scaled_value)
    params_list = np.array(params_list)

    if give_redshift:
        #scale redshift from 0 to 1
        redshift = params['Z_Q']
        redshift = (redshift-min_max_dict['Z_Q'][0])/(min_max_dict['Z_Q'][1]-min_max_dict['Z_Q'][0])
    params_list = params_list.astype(np.float32)

    # assert that the parameters are in the correct range
    assert(np.all(params_list >= 0.0) and np.all(params_list <= 1.0)), "Parameters are not in the correct range!"

    LC = LC.astype(np.float32)

    #save the true LC for validation
    if save_true_LC:
        true_LC = np.copy(LC)
    
    #Produce an observed LC with realistic cadence and photometric noise
    flag = True
    while flag:
        try:
            cadence_index = np.random.randint(num_cadences)
            LC, stdev = get_observed_LC(LC,cadence_index,cadence)
            flag = False
        except:
            pass

    mask = (LC != 0.0).astype(np.float32)
    assert(np.sum(mask) > 0), "There are no observed data points in the light curve!"

    x = np.array(list(range(LC.shape[0]))).astype(np.float32)
    x = (x-x.min())/(x.max()-x.min()) #scale x from 0 to 1

    #subtract the mean magnitude to zero during training
    LC = LC - mag_mean

    #mask the unobserved data again
    LC = mask*LC
    stdev = mask*stdev

    #add error bars to light curve array 
    LC = np.concatenate((LC,stdev),axis=1).astype(np.float32)
    
    # replace nan and inf with 0 just in case
    LC = np.nan_to_num(LC,nan=0.0,posinf=0.0,neginf=0.0)
    true_LC = np.nan_to_num(true_LC,nan=0.0,posinf=0.0,neginf=0.0)    

    sample = {"x":x, "y":LC, "params":params_list} 
    if save_true_LC:
        sample["true_LC"] = true_LC
    if give_redshift:
        sample["redshift"] = redshift

    return sample

class Dataset_Loader(Dataset):
    """
    Pytorch Dataloader
    """
    def __init__(self,file_names,cadence,mag_mean,augment=False,save_true_LC=False,load_before=False):
        """
        file_names: list of file names to load in 
        cadence: time spacing of data set
        mag_mean: list of mean magnitude to use for each band
        augment: if true, augment data set by adding noise
        save_true_LC: if true, save true light curve for validation
        """
        self.file_names = file_names
        self.augment = augment
        self.cadence = cadence
        self.mag_mean = mag_mean
        self.save_true_LC = save_true_LC
        self.load_before = load_before

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        
        if self.load_before:
            file_name = self.file_names[index]
            sample = np.load(file_name,allow_pickle=True).item()

        else:
            file_name = self.file_names[index]
            sample = build_data_set(file_name, self.mag_mean, self.cadence, self.augment, self.save_true_LC)

        return sample


def plot_confusion_matrices(mean_values_dict,test_label_dict,num_bins=20):
    """
    This function plots the confusion matrices for the test set.

    mean_values_dict: dictionary of mean values for each parameter
    test_label_dict: dictionary of labels for each parameter
    num_bins: number of bins to use for confusion matrix
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(4*num_cols,3*num_rows))

    i=0
    j=0

    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
        if k < num_parameters:
            par = parameters_keys[k]

            mean_val = (min_max_dict[par][1]-min_max_dict[par][0])*mean_values_dict[par]+min_max_dict[par][0]
            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*test_label_dict[par]+min_max_dict[par][0]

            ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   

            im = ax.hist2d(truth_val,mean_val,bins=(num_bins, num_bins),
                        range=[[min_max_dict[par][0],min_max_dict[par][1]],[min_max_dict[par][0], min_max_dict[par][1]]])
            if j == num_cols-1 or k == num_parameters-1:
                fig.colorbar(im[3], ax=ax, label='number LC')
            else:
                fig.colorbar(im[3], ax=ax)
            ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
            ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
            ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
            
            ax.minorticks_on()
            
            ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.yaxis.set_major_locator(plt.LinearLocator(3))
            
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            ax.set_aspect('equal', adjustable='box')
        
        else:
            axes[i,j].set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_together_{num_bins}_bins.{save_file_format}",bbox_inches='tight')
    plt.savefig(f"{save_path}/confusion_matrix_together_{num_bins}_bins.png",bbox_inches='tight',dpi=1000)
    if plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=20,log_mass_limit=None):
    """
    This function plots the confusion matrices for the test set.

    mean_values_dict: dictionary of mean values for each parameter
    test_label_dict: dictionary of labels for each parameter
    num_bins: number of bins to use for confusion matrix
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(4*num_cols,3*num_rows))

    # create new mean_values_dict with only values above log_mass_limit
    if log_mass_limit is not None:
        new_mean_values_dict = dict()
        new_test_label_dict = dict()
        for i in range(len(mean_values_dict['MASS'])):
            mass_val = (min_max_dict['MASS'][1]-min_max_dict['MASS'][0])*test_label_dict['MASS'][i]+min_max_dict['MASS'][0]
            if mass_val > log_mass_limit:
                for key in mean_values_dict.keys():
                    if key not in new_mean_values_dict.keys():
                        new_mean_values_dict[key] = []
                        new_test_label_dict[key] = []
                    new_mean_values_dict[key].append(mean_values_dict[key][i])
                    new_test_label_dict[key].append(test_label_dict[key][i])
        # make numpy arrays
        for key in new_mean_values_dict.keys():
            new_mean_values_dict[key] = np.array(new_mean_values_dict[key])
            new_test_label_dict[key] = np.array(new_test_label_dict[key])
        new_min_max_dict = dict()
        for key in min_max_dict.keys():
            if key not in new_min_max_dict.keys():
                new_min_max_dict[key] = []
            if key == 'MASS':
                new_min_max_dict[key].append(log_mass_limit)
            else:
                new_min_max_dict[key].append(min_max_dict[key][0])
            new_min_max_dict[key].append(min_max_dict[key][1])
    else:
        new_min_max_dict = min_max_dict
        new_mean_values_dict = mean_values_dict
        new_test_label_dict = test_label_dict

    i=0
    j=0
    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
        if k < num_parameters:
            par = parameters_keys[k]

            mean_val = (min_max_dict[par][1]-min_max_dict[par][0])*new_mean_values_dict[par]+min_max_dict[par][0]
            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*new_test_label_dict[par]+min_max_dict[par][0]

            ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   

            im = ax.hist2d(truth_val,mean_val,bins=(num_bins, num_bins),
                        range=[[min_max_dict[par][0],min_max_dict[par][1]],[min_max_dict[par][0], min_max_dict[par][1]]])
            if j == num_cols-1 or k == num_parameters-1:
                fig.colorbar(im[3], ax=ax, label='number LC')
            else:
                fig.colorbar(im[3], ax=ax)
            ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
            ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
            
            #ax.set_xlim(new_min_max_dict[par][0], new_min_max_dict[par][1])
            ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
            ax.set_ylim(min_max_dict[par][0], min_max_dict[par][1])

            ax.minorticks_on()
            
            ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.yaxis.set_major_locator(plt.LinearLocator(3))
            
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.set_aspect('equal', adjustable='box')
        else:
            axes[i,j].set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_together_mass_lim_{log_mass_limit}_{num_bins}_bins.{save_file_format}",bbox_inches='tight')
    plt.savefig(f"{save_path}/confusion_matrix_together_mass_lim_{log_mass_limit}_{num_bins}_bins.png",bbox_inches='tight',dpi=1000)
    if plot:
        plt.show()
    else:
        plt.close()

def plot_scatter_plots(param_labels,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=None,max_num_samples=None):
    """
    plot a scatter plot of the predictions with the median and the 68% confidence interval vs the true values

    param_labels, the true values of the parameters (but normalized between 0 and 1), shape (num_samples,num_parameters)
    eval_median, the median of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_lower_bound, the lower bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_upper_bound, the upper bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    log_mass_limit, the log mass limit of the data, if None, no mass limit is applied
    max_num_samples, the maximum number of samples to plot, if None, all samples are plotted
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(4*num_cols,3*num_rows))

    if log_mass_limit is not None:
        new_min_max_dict = dict()
        for key in min_max_dict.keys():
            if key not in new_min_max_dict.keys():
                new_min_max_dict[key] = []
            if key == 'MASS':
                new_min_max_dict[key].append(log_mass_limit)
            else:
                new_min_max_dict[key].append(min_max_dict[key][0])
            new_min_max_dict[key].append(min_max_dict[key][1])
    else:
        new_min_max_dict = min_max_dict


    if log_mass_limit is not None:
        eval_median_new = []
        eval_lower_bound_new = []
        eval_upper_bound_new = []

        param_labels_new = []

        for i in range(len(eval_median)):
            mass_val = (min_max_dict['MASS'][1]-min_max_dict['MASS'][0])*param_labels[i,parameters_keys.index('MASS')]+min_max_dict['MASS'][0]
            if mass_val > log_mass_limit:
                eval_median_new.append(eval_median[i])
                eval_lower_bound_new.append(eval_lower_bound[i])
                eval_upper_bound_new.append(eval_upper_bound[i])
                param_labels_new.append(param_labels[i])
        # make numpy arrays
        eval_median_new = np.array(eval_median_new)
        eval_lower_bound_new = np.array(eval_lower_bound_new)
        eval_upper_bound_new = np.array(eval_upper_bound_new)
        param_labels_new = np.array(param_labels_new)

    else:
        eval_median_new = eval_median
        eval_lower_bound_new = eval_lower_bound
        eval_upper_bound_new = eval_upper_bound
        param_labels_new = param_labels

    if max_num_samples is not None:
        max_num_samples = min(max_num_samples,eval_median_new.shape[0])
        eval_median_new = eval_median_new[:max_num_samples]
        eval_lower_bound_new = eval_lower_bound_new[:max_num_samples]
        eval_upper_bound_new = eval_upper_bound_new[:max_num_samples]
        param_labels_new = param_labels_new[:max_num_samples]
    
    i=0
    j=0
    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
        if k < num_parameters:
            par = parameters_keys[k]

            lower_error = eval_median_new[:,k] - eval_lower_bound_new[:,k]
            upper_error = eval_upper_bound_new[:,k] - eval_median_new[:,k]
            median_val = (min_max_dict[par][1]-min_max_dict[par][0])*eval_median_new[:,k]+min_max_dict[par][0]
            lower_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*lower_error
            upper_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*upper_error

            asymmetric_error = [lower_error_val, upper_error_val]

            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*param_labels_new[:,k]+min_max_dict[par][0]


            # get default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   
            ax.errorbar(truth_val,median_val,yerr=asymmetric_error,fmt='o',markersize=2,elinewidth=0.5,capsize=1.5,capthick=0.5,color=colors[0])

            ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
            ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
            
            #ax.set_xlim(new_min_max_dict[par][0], new_min_max_dict[par][1])
            ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
            ax.set_ylim(min_max_dict[par][0], min_max_dict[par][1])
            
            ax.minorticks_on()
            
            ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.yaxis.set_major_locator(plt.LinearLocator(3))
            
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            
            
            #FIX
            #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.set_aspect('equal', adjustable='box')
        else:
            axes[i,j].set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/recovery/scatter_together_mass_lim_{log_mass_limit}_max_samples_{max_num_samples}.{save_file_format}",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_scatter_DRW_params(param_labels,eval_median,eval_lower_bound,eval_upper_bound, GPR_log_tau, GPR_SF_inf, max_num_samples=None):
    """
    plot a scatter plot of the predictions with the median and the 68% confidence interval vs the true values

    param_labels, the true values of the parameters (but normalized between 0 and 1), shape (num_samples,num_parameters)
    eval_median, the median of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_lower_bound, the lower bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_upper_bound, the upper bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    GPR_log_tau, the GPR model for the log_tau parameter, shape (num_samples,num_parameters)
    GPR_SF_inf, the GPR model for the SF_inf parameter, shape (num_samples,num_parameters)
    max_num_samples, the maximum number of samples to plot, if None, all samples are plotted
    """
    num_parameters = len(parameters_keys)

    fig, axes = plt.subplots(1, 2,figsize=(10,5))

    if max_num_samples is not None:
        max_num_samples = min(max_num_samples,eval_median.shape[0])
        eval_median = eval_median[:max_num_samples]
        eval_lower_bound = eval_lower_bound[:max_num_samples]
        eval_upper_bound = eval_upper_bound[:max_num_samples]
        param_labels_new = param_labels[:max_num_samples]
        GPR_log_tau = GPR_log_tau[:max_num_samples]
        GPR_SF_inf = GPR_SF_inf[:max_num_samples]
        
    DRW_param_indexes = [parameters_keys.index('log_tau'),parameters_keys.index('SFinf')]
    
    j = 0
    for k in DRW_param_indexes:
  
        ax = axes[j]
    
        par = parameters_keys[k]

        lower_error = eval_median[:,k] - eval_lower_bound[:,k]
        upper_error = eval_upper_bound[:,k] - eval_median[:,k]
        median_val = (min_max_dict[par][1]-min_max_dict[par][0])*eval_median[:,k]+min_max_dict[par][0]
        lower_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*lower_error
        upper_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*upper_error
        truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*param_labels_new[:,k]+min_max_dict[par][0]

        asymmetric_error = [lower_error_val, upper_error_val]

        # get default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   
        ax.errorbar(truth_val,median_val,yerr=asymmetric_error,fmt='o',markersize=2,elinewidth=0.5,capsize=1.5,capthick=0.5,color=colors[0],label='SDE')
        if par == 'log_tau':
            ax.plot(truth_val, GPR_log_tau, 'o', color=colors[1], label='GPR',markersize=2)
        elif par == 'SFinf':
            ax.plot(truth_val, GPR_SF_inf, 'o', color=colors[1], label='GPR', markersize=2)

        ax.legend(loc='upper left',fontsize=10)
        ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
        ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
        ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
        ax.set_ylim(min_max_dict[par][0], min_max_dict[par][1])
        ax.minorticks_on()
        
        ax.xaxis.set_major_locator(plt.LinearLocator(3))
        ax.yaxis.set_major_locator(plt.LinearLocator(3))
        
        ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        
        ax.set_aspect('equal', adjustable='box')

        j += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/recovery/scatter_compare_GPR_SDE_{max_num_samples}.{save_file_format}",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()


def plot_residual_histograms(mean_values_dict,test_label_dict):
    """
    This function plots the residual histograms for the parameter predictions

    mean_values_dict: dictionary of the mean values of the predictions
    test_label_dict: dictionary of the true values of the parameters
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(7*num_cols,3*num_rows))

    i=0
    j=0

    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
            
        if k < num_parameters:
            par = parameters_keys[k]

            mean_val = (min_max_dict[par][1]-min_max_dict[par][0])*mean_values_dict[par]+min_max_dict[par][0]
            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*test_label_dict[par]+min_max_dict[par][0]

            diff = mean_val-truth_val
            y,x,_ = ax.hist(diff, bins=20) 
            ax.set_xlabel(f"$\Delta${plotting_labels[par]}",fontsize=12)

            if j==0:
                ax.set_ylabel("number LC")
            ax.minorticks_on()
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            mean = np.mean(diff)
            stdev= np.std(diff)
            plt.xlim(mean-6*stdev,mean+6*stdev)
            textstr = '\n'.join((
                r'$\mu=%.2e$' % (mean, ),
                r'$\sigma=%.2e$' % (stdev, )))
            #this is just for the to display the mean and width
            ax.text(0.42*np.max(x), 0.8*np.max(y), textstr,verticalalignment='top',
                      bbox = dict(boxstyle = "square",facecolor='white',alpha = 1))
        else:
            ax.set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/diff_hist_together.{save_file_format}",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_residual_corner(mean,truth,LC_reconstruction_mean_diff=np.array([]),plot_truth=False,zoom=False,zoom_range=1.0,):
    """
    This function plots a corner plot of the residuals of the parameter predictions

    mean: array of the mean values of the predictions
    truth: array of the true values of the parameters
    LC_reconstruction_mean_diff: array of the residuals of the LC reconstruction
    zoom: boolean to decide if the corner plot should be zoomed into the maximum residual or else use the full parameter range
    zoom_range: the range to zoom in the corner plot of the full range of the parameters. Only used if zoom is False.
    """

    var_name_list = []
    var_name_list_no_units = []
    truth_list = []   

    if plot_truth:
        for j,par in enumerate(parameters_keys):
            truth_list.append(0)
            var_name_list.append(plotting_labels[par])
            var_name_list_no_units.append(plotting_labels_no_units[par])

    for j,par in enumerate(parameters_keys):
        mean[:,j] = (min_max_dict[par][1]-min_max_dict[par][0])*mean[:,j]+min_max_dict[par][0]
        truth[:,j] = (min_max_dict[par][1]-min_max_dict[par][0])*truth[:,j]+min_max_dict[par][0]
        
        #The true value is zero since it is a residual
        truth_list.append(0)
        var_name_list.append(f"$\Delta${plotting_labels[par]}")
        var_name_list_no_units.append(f"$\Delta${plotting_labels_no_units[par]}")

    # Then we are including the residuals of the LC reconstruction
    if len(LC_reconstruction_mean_diff) > 0:
        truth_list.append(0)
        var_name_list.append(r"$|\Delta m_{\mathrm{LC}}|$ [mag]")
        var_name_list_no_units.append(r"$|\Delta m_{\mathrm{LC}}|$")

    if plot_truth:
        parameters_keys_new = []
        for j,par in enumerate(parameters_keys):
            parameters_keys_new.append(par+"_truth")
        for j,par in enumerate(parameters_keys):
            parameters_keys_new.append(par)
    else:
        parameters_keys_new = parameters_keys.copy()
    if len(LC_reconstruction_mean_diff) > 0:
        parameters_keys_new.append("LC")

    new_min_max_dict = {}

    for j,par in enumerate(parameters_keys):
        if zoom:
            max_diff = 1.025*np.max(np.abs(mean[:,j]-truth[:,j]))
            new_min_max_dict[par] = [-max_diff,max_diff]
        else:
            new_min_max_dict[par] = [-(min_max_dict[par][1]-min_max_dict[par][0])/zoom_range,(min_max_dict[par][1]-min_max_dict[par][0])/zoom_range]
        if plot_truth:
            new_min_max_dict[par+"_truth"] = [min_max_dict[par][0],min_max_dict[par][1]]

    if len(LC_reconstruction_mean_diff) > 0:
        max_diff = 1.025*LC_reconstruction_mean_diff.max()
        # what do we do about this?!!!! This is already the abs of diff
        new_min_max_dict["LC"] = [-0.025*max_diff,max_diff]


    sample = np.zeros((len(mean),len(parameters_keys_new)))
    if plot_truth:
        sample[:,:len(parameters_keys)] = truth
        sample[:,len(parameters_keys):2*len(parameters_keys)] = mean-truth
    else:
        sample[:,:len(parameters_keys)] = mean-truth

    if len(LC_reconstruction_mean_diff) > 0:
        sample[:,-1] = LC_reconstruction_mean_diff

    if plot_truth:
        figure = corner.corner(sample,smooth=True,labels=var_name_list,
                            label_kwargs={'fontsize':14.5},plot_datapoints=False,quantiles=[0.159, 0.5, 0.841],
                            show_titles=True,titles=var_name_list_no_units,title_kwargs={"fontsize": 11.},
                            levels=(0.683,0.955,0.997))
    else:
        figure = corner.corner(sample,smooth=True,labels=var_name_list,truths=truth_list,
                            label_kwargs={'fontsize':14.5},plot_datapoints=False,quantiles=[0.159, 0.5, 0.841],
                            show_titles=True,titles=var_name_list_no_units,title_kwargs={"fontsize": 11.},
                            levels=(0.683,0.955,0.997))

    axes = np.array(figure.axes).reshape((len(parameters_keys_new), len(parameters_keys_new)))
    
    # Loop over the diagonal
    for i,par in enumerate(parameters_keys_new):
        ax = axes[i, i]
        
        ax.set_xlim(new_min_max_dict[par][0],new_min_max_dict[par][1])
        
        ax.minorticks_on()
        #ax.xaxis.set_major_locator(plt.LinearLocator(3))
        ax.tick_params(which='major',direction='in',top=False, right=True,length=tick_length_major,width=tick_width)
        ax.tick_params(which='minor',direction='in',top=False, right=True,length=tick_length_minor,width=tick_width)

    # Loop over the histograms
    for i in range(len(parameters_keys_new)):
        for j in range(i):
            ax = axes[i, j]  
            ax.set_xlim(new_min_max_dict[parameters_keys_new[j]][0],new_min_max_dict[parameters_keys_new[j]][1])
            ax.set_ylim(new_min_max_dict[parameters_keys_new[i]][0],new_min_max_dict[parameters_keys_new[i]][1])
            ax.minorticks_on()
            ax.tick_params(which='major',direction='inout',top=True, right=True,length=10,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

    figure.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(f'{save_path}/Residual_corner_plot_zoom_{zoom}range_{zoom_range}_truth_{plot_truth}.{save_file_format}',bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()      

def plot_metric_vs_epoch(training_list,val_list,ylabel):
    """
    This function plots the training and validation metrics as a function of the epoch

    training_list: list of training metrics
    val_list: list of validation metrics
    ylabel: label of the y-axis
    """
    if training_list != None:
        plt.plot(training_list,label='training')
    if val_list != None:
        plt.plot(val_list,label='validation')
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    if training_list != None and val_list != None:
        plt.legend()
    save_name = ylabel.replace(" ", "_")
    plt.savefig(f"{save_path}/{save_name}_vs_epoch.{save_file_format}",bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_LC_reconstruction_sigma_eval(sigma1,sigma2,sigma3,name):
    """
    This function plots the convergent probability for the light curve reconstruction for 1,2,3 sigma.

    sigma1:  convergent probability for 1 sigma reconstruction
    sigma2:  convergent probability for 2 sigma reconstruction
    sigma3:  convergent probability for 3 sigma reconstruction
    """
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    probability_in_sigmas = np.array([sigma1,sigma2,sigma3])
    ax.bar(np.arange(3)/1.5+0.25, probability_in_sigmas, width=0.5)

    plt.xlim(-0.45,1.9)
    plt.hlines(.683,-0.18,6.5,linestyle="dashed",color='black')
    plt.text(-0.38,.683, '0.683', ha ='left', va ='center')
    plt.hlines(.955,-0.18,6.5,linestyle="dashed",color='black')
    plt.text(-0.38,.955, '0.955', ha ='left', va ='center')
    plt.hlines(.997,-0.18,6.5,linestyle="dashed",color='black')
    plt.text(-0.38,0.997, '0.997', ha ='left', va ='center')
    plt.ylim(0,1.1)
    plt.ylabel("coverage probability")
    my_xticks = ['$1\sigma$', '$2\sigma$', '$3\sigma$']
    plt.xticks([0.25,1/1.5+0.25,2/1.5+0.25], my_xticks)
    plt.xlabel('confidence interval')
    fig.savefig(f"{save_path}/{name}.{save_file_format}",bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.close()

def plot_sigma_eval(coverage_prob_dict,quantiles):
    """
    This function plots the convergent probability for the parameter reconstruction for the quantiles given in the quantiles list as a bar graph.

    coverage_prob_dict: dictionary with the convergent probability for each parameter for each quantile in the quantiles list
    quantiles: list of quantiles for which the convergent probability is calculated
    """
    probability_in_quantiles = []
    for par in parameters_keys:
        quantile_list = []
        for quantile in quantiles:
            quantile_list.append(coverage_prob_dict[quantile][par])
        probability_in_quantiles.append(quantile_list)
    print()
    print(f"coverage probabilities for quantiles {quantiles}")
    print(probability_in_quantiles)
    print()
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    avg_prob_in_sigmas = np.zeros(3)
    for j,par in enumerate(parameters_keys):
        
        ax.bar(np.arange(3)/1.5+j*0.5/len(parameters_keys), probability_in_quantiles[j],
               width = 0.5/len(parameters_keys),label=plotting_labels_no_units[par])
        
        avg_prob_in_sigmas += np.array(probability_in_quantiles[j])
    avg_prob_in_sigmas /= len(parameters_keys)

    xtick = []
    for i in range(3):
        xtick.append(i/1.5+0.25-0.25/len(parameters_keys))
    ax.bar(xtick,avg_prob_in_sigmas,width = 0.6,alpha = 0.1,color='black')

    plt.xlim(-0.70,1.9)
    plt.hlines(.683,-0.32,6.5,linestyle="dashed",color='black')
    plt.text(-0.55,.683, '0.683', ha ='left', va ='center')
    plt.hlines(.955,-0.32,6.5,linestyle="dashed",color='black')
    plt.text(-0.55,.955, '0.955', ha ='left', va ='center')
    plt.hlines(.997,-0.32,6.5,linestyle="dashed",color='black')
    plt.text(-0.55,0.997, '0.997', ha ='left', va ='center')
    plt.ylim(0,1.1)
    plt.ylabel("coverage probability")
    my_xticks = []
    for quantile in quantiles:
        my_xticks.append(f"{round(100*quantile,4)}%")
    plt.xticks(xtick, my_xticks) 
    plt.xlabel('confidence interval')
    plt.legend(loc="lower left",fontsize=11.5)
    fig.savefig(f"{save_path}/coverage_probability_bar.{save_file_format}",bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.close()

def plot_sigma_eval_all(all_coverage_prob_dict,all_quantiles):
    """
    This function plots the convergent probability for the parameter reconstruction for the quantiles given in the quantiles list as a graph.

    coverage_prob_dict: dictionary with the convergent probability for each parameter for each quantile in the quantiles list
    quantiles: list of quantiles for which the convergent probability is calculated
    """
    for i in range(3):
        if i == 0 or i == 1:
            for par in parameters_keys:
                probability_in_quantiles = []
                for quantile in all_quantiles:
                    probability_in_quantiles.append(all_coverage_prob_dict[quantile][par])
                plt.plot(all_quantiles,probability_in_quantiles,label=plotting_labels_no_units[par])
        if i == 1 or i == 2:
            recovery_keys = ["recovery","recovery_GPR"]
            recovery_pars = ["recovery SDE","recovery GPR"]
            for key, label in zip(recovery_keys,recovery_pars):
                probability_in_quantiles = []
                for quantile in all_quantiles:
                    probability_in_quantiles.append(all_coverage_prob_dict[quantile][key])
                if all_coverage_prob_dict[quantile][key] != 0.0:
                    plt.plot(all_quantiles,probability_in_quantiles,label=label)
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot a dotted line across the diagonal of the plot
        plt.plot([0,1],[0,1],linestyle="dashed",color='black')
        
        #plt.ylabel("coverage probability",fontsize=13)
        #plt.xlabel('confidence interval',fontsize=13)
        plt.ylabel('fraction of truth in probability volume',fontsize=13)
        plt.xlabel('fraction of posterior probability volume',fontsize=13)

        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        if len(parameters_keys) > 4 and i != 2:
            if i ==0:
                plt.legend(loc="upper left",fontsize=11.5)
            else:
                plt.legend(loc="upper left",fontsize=11.0)
        else:
            plt.legend(loc="upper left",fontsize=13)
        if i == 0:
            plt.savefig(f"{save_path}/coverage_probability_graph.{save_file_format}",bbox_inches="tight")
        elif i == 1:
            plt.savefig(f"{save_path}/coverage_probability_graph_with_recovery.{save_file_format}",bbox_inches="tight")
        elif i == 2:
            plt.savefig(f"{save_path}/coverage_probability_graph_just_recovery.{save_file_format}",bbox_inches="tight")
        if plot:
            plt.show()
        else:
            plt.close()


def corner_plot(sample,truth,num):
    """
    Makes a corner plot of the posterior distribution of the parameters.

    sample: posterior sample of the parameters
    truth: true values of the parameters
    """
    
    var_name_list = []
    var_name_list_no_units = []
    truth_list = []   

    sample = np.copy(sample)

    for j,par in enumerate(parameters_keys):
        sample[:,j] = (min_max_dict[par][1]-min_max_dict[par][0])*sample[:,j]+min_max_dict[par][0]
        truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*truth[j]+min_max_dict[par][0]

        truth_list.append(truth_val)
        var_name_list.append(plotting_labels[par])
        var_name_list_no_units.append(plotting_labels_no_units[par])

    figure = corner.corner(sample,smooth=True,labels=var_name_list,truths=truth_list,
                          label_kwargs={'fontsize':14.5},plot_datapoints=False,quantiles=[0.159, 0.5, 0.841],
                          show_titles=True,titles=var_name_list_no_units,title_kwargs={"fontsize": 11.5},
                          levels=(0.683,0.955,0.997))

    axes = np.array(figure.axes).reshape((len(parameters_keys), len(parameters_keys)))
    # Loop over the diagonal
    for i,par in enumerate(parameters_keys):
        ax = axes[i, i]
        
        ax.set_xlim(min_max_dict[par][0],min_max_dict[par][1])
        
        ax.minorticks_on()
        #ax.xaxis.set_major_locator(plt.LinearLocator(3))
        ax.tick_params(which='major',direction='in',top=False, right=True,length=tick_length_major,width=tick_width)
        ax.tick_params(which='minor',direction='in',top=False, right=True,length=tick_length_minor,width=tick_width)


    # Loop over the histograms
    for i in range(len(parameters_keys)):
        for j in range(i):
            ax = axes[i, j]  
            ax.set_xlim(min_max_dict[parameters_keys[j]][0],min_max_dict[parameters_keys[j]][1])
            ax.set_ylim(min_max_dict[parameters_keys[i]][0],min_max_dict[parameters_keys[i]][1])
            ax.minorticks_on()
            ax.tick_params(which='major',direction='inout',top=True, right=True,length=10,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

    figure.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig(f'{save_path}/Corner_plot_{num}.{save_file_format}',bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def make_gifs(name,num_epochs):
    '''
    makes gifs of the prior and recovery

    name: name of the image
    num_epochs: number of epochs
    '''
    with imageio.get_writer(f'{save_path}/{name}.gif',mode='I',duration=20/num_epochs,loop=1) as writer:
        for epoch in range(num_epochs):
            file_name = f"{save_path}/recovery/{name}_epoch_{epoch}.png"
            image = imageio.imread(file_name)
            writer.append_data(image)
            os.remove(file_name)

def power_spectrum(sig, dt):
    '''
    Plots the power spectrum of a signal

    sig: signal
    dt: time step
    '''
    N = len(sig)
    freq = np.fft.fftfreq(N, dt)
    ps = np.abs(np.fft.fft(sig))**2
    i = np.argsort(freq)
    return freq[i], ps[i]

def plot_power_spectrum(true_LC,reconstructed_LC,num,epoch):
    """
    This function plots the power spectrum of the true and reconstructed light curves.
    """
    # get the power spectrum
    num_bands = true_LC.shape[1]

    for i in range(2):
        # plot the power spectrum
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):

            freq, power_spectrum_true = power_spectrum(true_LC[:,band_i],cadence)
            axes[band_i].plot(freq, power_spectrum_true,linewidth=1.,label='truth')

            freq, power_spectrum_reconstructed = power_spectrum(reconstructed_LC[:,band_i],cadence)
            axes[band_i].plot(freq, power_spectrum_reconstructed,linewidth=1.,label='reconstructed')
            
            axes[band_i].set_xlim(0.001,1.0/(2.0*cadence))
            #axes[band_i].set_ylim(0.9*min(power_spectrum_true.min(),power_spectrum_reconstructed.min()), 1.1*max(power_spectrum_true.max(),power_spectrum_reconstructed.max()))
            axes[band_i].set_ylim(0.01, 1.1*max(power_spectrum_true[1:].max(),power_spectrum_reconstructed[1:].max()))
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            axes[band_i].set_yscale('log')
            if i == 1:
                axes[band_i].set_xscale('log')
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band pow.',fontsize = 13)
            if band_i == 0:
                axes[band_i].legend(fontsize=13)
                
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('frequency [1/days]',fontsize = 13)
        #plt.ylabel('power spectrum')
        if i == 0 :
            plt.savefig(f"{save_path}/recovery/recovery_power_spectrum_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/recovery/recovery_power_spectrum_log_freq_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()

def plot_power_spectrum_with_uncertainty(true_LC,combined_samples,num,epoch):
    """
    This function plots the power spectrum of the true and reconstructed light curves.
    """
    # get the power spectrum
    num_bands = true_LC.shape[1]

    combined_samples += mag_mean

    power_spectrum_reconstructed = np.zeros((combined_samples.shape))
    for i in range(combined_samples.shape[2]):
        for band_i in range(num_bands):
            freq, power = power_spectrum(combined_samples[:,band_i,i],cadence)
            power_spectrum_reconstructed[:,band_i,i] = power
    #take the log
    power_spectrum_reconstructed = np.log10(power_spectrum_reconstructed)
    power_spectrum_reconstructed_mean = np.mean(power_spectrum_reconstructed,axis=2)
    power_spectrum_reconstructed_std = np.std(power_spectrum_reconstructed,axis=2)



    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(2):
        # plot the power spectrum
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):

            freq, power_spectrum_true = power_spectrum(true_LC[:,band_i],cadence)
            axes[band_i].plot(freq, power_spectrum_true,linewidth=1.1,color=colors[0],label='truth')

            axes[band_i].plot(freq, power_spectrum_reconstructed_mean[:,band_i],linewidth=1.1,label='mean rec.',color=colors[1])
            alpha_list = [0.3,0.2,0.1]
            num_sigma_list = [1,2,3]
            for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                axes[band_i].fill_between(freq, 
                                        power_spectrum_reconstructed_mean[:,band_i]-num_sigma*power_spectrum_reconstructed_std[:,band_i],
                                        power_spectrum_reconstructed_mean[:,band_i]+num_sigma*power_spectrum_reconstructed_std[:,band_i],
                                        color=colors[1], 
                                        alpha=alpha, 
                                        label=f"{num_sigma}$\sigma$ unc.")
            
            axes[band_i].set_xlim(0.001,1.0/(2.0*cadence))
            #axes[band_i].set_ylim(0.9*min(power_spectrum_true.min(),power_spectrum_reconstructed.min()), 1.1*max(power_spectrum_true.max(),power_spectrum_reconstructed.max()))
            axes[band_i].set_ylim(min(power_spectrum_true[1:].min(),power_spectrum_reconstructed[1:].min())-0.1, 1.1*max(power_spectrum_true[1:].max(),power_spectrum_reconstructed[1:].max()))
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            if i == 1:
                axes[band_i].set_xscale('log')
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band pow.',fontsize = 13)
            if band_i == 0:
                axes[band_i].legend(fontsize=12,ncol=3)
                
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('frequency [1/days]',fontsize = 13)
        #plt.ylabel('power spectrum')
        if i == 0 :
            plt.savefig(f"{save_path}/recovery/recovery_power_spectrum_combined_samples_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/recovery/recovery_power_spectrum_combined_samples_log_freq_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()

def get_acf(y):
    """Get the ACF using numpy. Works best for finely and regularly
    sampled time series
    y : np.ndarray, Observed time series

    returns : np.ndarray, ACF from 1-element offset in y
    """
    def autocorr(x, t=1):
        return np.corrcoef(np.array([x[:-t], x[t:]]))
    acf = np.empty(*y.shape)  # init
    acf[0] = np.corrcoef(np.array([y, y]))[0, 1]
    for i, lag_i in enumerate(np.arange(1, len(y))):
        acf[lag_i] = autocorr(y, lag_i)[0, 1]
    return acf

def sf_from_acf(acf):
    """Convert autocorrelation function (ACF) into structure function (SF)
    Parameters
    ----------
    acf : np.ndarray
    """
    sf = (1.0 - acf)**0.5
    return sf

def plot_ACF_and_SF(true_LC,reconstructed_LC,num,epoch,SFinf,tau):
    """
    This function plots the ACF and the SF of the true and reconstructed light curves.
    """
    num_bands = true_LC.shape[1]
    num_days = true_LC.shape[0]

    ACF = np.zeros((num_days,num_bands))
    ACF_reconstructed = np.zeros((num_days,num_bands))
    SF = np.zeros((num_days,num_bands))
    SF_reconstructed = np.zeros((num_days,num_bands))

    for band_i in range(num_bands):
        ACF[:,band_i] = get_acf(true_LC[:,band_i])
        ACF_reconstructed[:,band_i] = get_acf(reconstructed_LC[:,band_i])
        SF[:,band_i] = sf_from_acf(ACF[:,band_i])
        SF_reconstructed[:,band_i] = sf_from_acf(ACF_reconstructed[:,band_i])

    # plot the ACF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):

            axes[band_i].plot(days, ACF[:,band_i],linewidth=1.,label=f'truth')
            axes[band_i].plot(days, ACF_reconstructed[:,band_i],linewidth=1.,label=f'reconstructed')
            axes[band_i].set_ylim(-1.05,1.05)
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            if band_i == 0:
                axes[band_i].legend(fontsize=12)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band ACF',fontsize=13)
            if i == 1:
                axes[band_i].set_xlim(days.min()+1,days.max())
                axes[band_i].set_xscale('log')
            else:
                axes[band_i].set_xlim(days.min(),days.max())

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_log_time_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()
    
    #plot the SF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):
            if i == 0:
                axes[band_i].plot(days, SF[:,band_i],linewidth=1.,label=f'truth')
                axes[band_i].plot(days, SF_reconstructed[:,band_i],linewidth=1.,label=f'reconstructed')
                #plot tau as a verticle line
                axes[band_i].axvline(x=tau,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=12)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]} '+r'$\mathrm{SF}/\mathrm{SF}_\infty$',fontsize=13)
                axes[band_i].set_xlim(days.min()+1,days.max())
                axes[band_i].set_xscale('log')
            else:
                #not sure what to do for this yet

                SF_normalized = SF[:,band_i] #/SFinf
                SF_reconstructed_normalized = SF_reconstructed[:,band_i] #/SFinf
                xaxis = np.copy(days)/tau
                axes[band_i].plot(xaxis, SF_normalized,linewidth=1.,label=f'truth')
                axes[band_i].plot(xaxis, SF_reconstructed_normalized,linewidth=1.,label=f'reconstructed')
                #plot tau as a verticle line
                axes[band_i].axvline(x=1.0,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=12)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]} '+r'$\mathrm{SF}/\mathrm{SF}_\infty$',fontsize=13)
                axes[band_i].set_xlim(max((days.min()+1)/tau,0.1),min(days.max()/tau,100))
                axes[band_i].set_xscale('log')

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.xlabel('time [days]',fontsize=13)
            plt.savefig(f"{save_path}/recovery/recovery_SF_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.xlabel(r'$\Delta t/\tau$',fontsize=13)
            plt.savefig(f"{save_path}/recovery/recovery_SF_over_DRW_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()

def plot_ACF_and_SF_with_uncertainty(true_LC,combined_samples,num,epoch,SFinf,tau):
    """
    This function plots the ACF and the SF of the true and reconstructed light curves.
    """
    num_bands = true_LC.shape[1]

    ACF = np.zeros(true_LC.shape)
    ACF_reconstructed = np.zeros(combined_samples.shape)
    SF = np.zeros(true_LC.shape)
    SF_reconstructed = np.zeros(combined_samples.shape)


    print("Calculating ACF and SF with uncertainties")
    for band_i in range(num_bands):
        ACF[:,band_i] = get_acf(true_LC[:,band_i])
        SF[:,band_i] = sf_from_acf(ACF[:,band_i])
        for i in range(combined_samples.shape[2]):
            ACF_reconstructed[:,band_i,i] = get_acf(combined_samples[:,band_i,i])
            SF_reconstructed[:,band_i,i] = sf_from_acf(ACF_reconstructed[:,band_i,i])

    mean_ACF_reconstructed = np.mean(ACF_reconstructed,axis=2)
    mean_SF_reconstructed = np.mean(SF_reconstructed,axis=2)
    std_ACF_reconstructed = np.std(ACF_reconstructed,axis=2)
    std_SF_reconstructed = np.std(SF_reconstructed,axis=2)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot the ACF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):

            axes[band_i].plot(days, ACF[:,band_i],label=f'truth', linewidth=1.1,color=colors[0])
            axes[band_i].plot(days, mean_ACF_reconstructed[:,band_i],label=f'mean rec.',linewidth=1.1,color=colors[1])

            alpha_list = [0.3,0.2,0.1]
            num_sigma_list = [1,2,3]
            for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                axes[band_i].fill_between(days, 
                                mean_ACF_reconstructed[:,band_i]-num_sigma*std_ACF_reconstructed[:,band_i],
                                mean_ACF_reconstructed[:,band_i]+num_sigma*std_ACF_reconstructed[:,band_i],
                                color=colors[1], 
                                alpha=alpha, 
                                label=f"{num_sigma}$\sigma$ unc.")

            axes[band_i].set_ylim(-1.05,1.05)
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            if band_i == 0:
                axes[band_i].legend(fontsize=10,ncol=3)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band ACF',fontsize=12)
            if i == 1:
                axes[band_i].set_xlim(days.min()+1,days.max())
                axes[band_i].set_xscale('log')
            else:
                axes[band_i].set_xlim(days.min(),days.max())

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_combined_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_combined_log_time_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()
    
    #plot the SF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):
            if i == 0:
                axes[band_i].plot(days, SF[:,band_i],linewidth=1.1,label=f'truth',color=colors[0])
                axes[band_i].plot(days, mean_SF_reconstructed[:,band_i],linewidth=1.1,label=f'mean rec.',color=colors[1])
                alpha_list = [0.3,0.2,0.1]
                num_sigma_list = [1,2,3]
                for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                    axes[band_i].fill_between(days, 
                                            mean_SF_reconstructed[:,band_i]-num_sigma*std_SF_reconstructed[:,band_i],
                                            mean_SF_reconstructed[:,band_i]+num_sigma*std_SF_reconstructed[:,band_i],
                                            color=colors[1], 
                                            alpha=alpha, 
                                            label=f"{num_sigma}$\sigma$ unc.")
                
                #plot tau as a verticle line
                axes[band_i].axvline(x=tau,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=10,ncol=3)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band ACF',fontsize=12)
                axes[band_i].set_xlim(days.min()+1,days.max())

                axes[band_i].set_xscale('log')
            else:
                #not sure what to do for this yet
                xaxis = np.copy(days)/tau
                axes[band_i].plot(xaxis, SF[:,band_i],linewidth=1.1,label=f'truth',color=colors[0])
                axes[band_i].plot(xaxis, mean_SF_reconstructed[:,band_i],linewidth=1.1,label=f'mean rec.',color=colors[1])
                alpha_list = [0.3,0.2,0.1]
                num_sigma_list = [1,2,3]
                for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                    axes[band_i].fill_between(xaxis, 
                                            mean_SF_reconstructed[:,band_i]-num_sigma*std_SF_reconstructed[:,band_i],
                                            mean_SF_reconstructed[:,band_i]+num_sigma*std_SF_reconstructed[:,band_i],
                                            color=colors[1], 
                                            alpha=alpha, 
                                            label=f"{num_sigma}$\sigma$ unc.")
                
                #plot tau as a verticle line
                axes[band_i].axvline(x=1.0,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=10,ncol=3)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band '+r'$\mathrm{SF}/\mathrm{SF}_\infty$',fontsize=12)
                axes[band_i].set_xlim(max((days.min()+1)/tau,0.1),min(days.max()/tau,100))
                axes[band_i].set_xscale('log')

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.xlabel('time [days]',fontsize=13)
            plt.savefig(f"{save_path}/recovery/recovery_SF_combined_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.xlabel(r'$\Delta t/\tau$')
            plt.savefig(f"{save_path}/recovery/recovery_SF_over_DRW_combined_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()

def plot_ACF_and_SF_latent_vs_GPR(true_LC,reconstructed_LC,reconstructed_LC_GPR,num,epoch,SFinf,tau):
    """
    This function plots the ACF and the SF of the true and reconstructed light curves.
    """
    num_bands = true_LC.shape[1]
    num_days = true_LC.shape[0]

    ACF = np.zeros((num_days,num_bands))
    ACF_reconstructed = np.zeros((num_days,num_bands))
    ACF_reconstructed_GPR = np.zeros((num_days,num_bands))
    SF = np.zeros((num_days,num_bands))
    SF_reconstructed = np.zeros((num_days,num_bands))
    SF_reconstructed_GPR = np.zeros((num_days,num_bands))

    for band_i in range(num_bands):
        ACF[:,band_i] = get_acf(true_LC[:,band_i])
        ACF_reconstructed[:,band_i] = get_acf(reconstructed_LC[:,band_i])
        ACF_reconstructed_GPR[:,band_i] = get_acf(reconstructed_LC_GPR[:,band_i])
        SF[:,band_i] = sf_from_acf(ACF[:,band_i])
        SF_reconstructed[:,band_i] = sf_from_acf(ACF_reconstructed[:,band_i])
        SF_reconstructed_GPR[:,band_i] = sf_from_acf(ACF_reconstructed_GPR[:,band_i])

    # plot the ACF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):

            axes[band_i].plot(days, ACF[:,band_i],linewidth=1.,label=f'truth')
            axes[band_i].plot(days, ACF_reconstructed[:,band_i],linewidth=1.,label=f'latent SDE')
            axes[band_i].plot(days, ACF_reconstructed_GPR[:,band_i],linewidth=1.,label=f'GPR')
            axes[band_i].set_ylim(-1.05,1.05)
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            if band_i == 0:
                axes[band_i].legend(fontsize=12)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band ACF',fontsize=13)
            if i == 1:
                axes[band_i].set_xlim(days.min()+1,days.max())
                axes[band_i].set_xscale('log')
            else:
                axes[band_i].set_xlim(days.min(),days.max())

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_SDE_vs_GPR_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{save_path}/recovery/recovery_ACF_SDE_vs_GPR_log_time_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()
    
    #plot the SF
    for i in range(2):
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        for band_i in range(num_bands):
            if i == 0:
                axes[band_i].plot(days, SF[:,band_i],linewidth=1.,label=f'truth')
                axes[band_i].plot(days, SF_reconstructed[:,band_i],linewidth=1.,label=f'SDE')
                axes[band_i].plot(days, SF_reconstructed_GPR[:,band_i],linewidth=1.,label=f'GPR')
                #plot tau as a verticle line
                axes[band_i].axvline(x=tau,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=12)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]} '+r'$\mathrm{SF}/\mathrm{SF}_\infty$',fontsize=13)
                axes[band_i].set_xlim(days.min()+1,days.max())
                axes[band_i].set_xscale('log')
            else:
                #not sure what to do for this yet

                SF_normalized = SF[:,band_i] #/SFinf
                SF_reconstructed_normalized = SF_reconstructed[:,band_i] #/SFinf
                SF_reconstructed_GPR_normalized = SF_reconstructed_GPR[:,band_i] #/SFinf
                xaxis = np.copy(days)/tau
                axes[band_i].plot(xaxis, SF_normalized,linewidth=1.,label=f'truth')
                axes[band_i].plot(xaxis, np.sqrt(1-np.exp(-xaxis)),linewidth=1.,label=f'DRW') # This is the SF for the x-ray driving DRW
                axes[band_i].plot(xaxis, SF_reconstructed_normalized,linewidth=1.,label=f'SDE')
                axes[band_i].plot(xaxis, SF_reconstructed_GPR_normalized,linewidth=1.,label=f'GPR')


                #plot tau as a verticle line
                axes[band_i].axvline(x=1.0,linestyle='--',color='k',linewidth=1.)
                #plot SFinf as a horizontal line
                axes[band_i].axhline(y=1.0,linestyle='--',color='k',linewidth=1.)
                axes[band_i].set_ylim(0,np.sqrt(2))
                axes[band_i].minorticks_on()
                axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                if band_i == 0:
                    axes[band_i].legend(fontsize=11,ncol=4)
                axes[band_i].set_ylabel(f'{bandpasses[band_i]} '+r'$\mathrm{SF}/\mathrm{SF}_\infty$',fontsize=13)
                axes[band_i].set_xlim(max((days.min()+1)/tau,0.1),min(days.max()/tau,100))
                axes[band_i].set_xscale('log')

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        #plt.ylabel('auto-correlation')
        if i ==0:
            plt.xlabel('time [days]',fontsize=13)
            plt.savefig(f"{save_path}/recovery/recovery_SF_SDE_vs_GPR_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        else:
            plt.xlabel(r'$\Delta t/\tau$',fontsize=13)
            plt.savefig(f"{save_path}/recovery/recovery_SF_SDE_vs_GPR_over_DRW_num_{num}_epoch_{epoch}.pdf",bbox_inches='tight')
        plt.close()

def plot_reconstruction(ys_val_temp,true_LC_temp,recovery_mean,recovery_std,band_color,mag_mean,epoch,num_epoch,num,SFinf,tau,GPR_mean=None,GPR_std=None):
    """
    This funciton plots the true light curve and the reconstructed light curve from the SDE.
    """

    # set the ylim for plotting
    ys_val_temp[:,:num_bands] += mag_mean 
    recovery_mean = np.copy(recovery_mean) + mag_mean
    lower_lim = min(ys_val_temp[:,:num_bands].min(),true_LC_temp.min())
    upper_lim = max(ys_val_temp[:,:num_bands].max(),true_LC_temp.max())
    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    # eliminate the points that are not observed by setting them to np.nan
    ys_val_temp_obs = np.copy(ys_val_temp)[:,:num_bands]
    ys_val_temp_obs[ys_val_temp_obs == mag_mean] = np.nan 
    ys_val_temp_std = np.copy(ys_val_temp)[:,num_bands:]
    ys_val_temp_std[ys_val_temp_std == 0] = np.nan

    #set the y label

    ylabel = 'magntiude'

    # make plot for each band seperately
    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for band_i in range(num_bands):
        #axes[band_i].plot(days, recovery_mean[:,band_i],color=band_color[band_i],linewidth=1.1,label=f'{bandpasses[band_i]} band')#,label='mean')
        axes[band_i].plot(days, recovery_mean[:,band_i],color=colors[1],linewidth=1.1,label='mean pred')#,label='mean')
        alpha_list = [0.3,0.2,0.1]
        num_sigma_list = [1,2,3]
        for alpha,num_sigma in zip(alpha_list,num_sigma_list):
            axes[band_i].fill_between(days, 
                            recovery_mean[:,band_i]-num_sigma*recovery_std[:,band_i],
                            recovery_mean[:,band_i]+num_sigma*recovery_std[:,band_i],
                            color=colors[1], #band_color[band_i],
                            alpha=alpha, 
                            label=f"{num_sigma}$\sigma$ unc.")
        
        axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
                        linestyle='--',label='truth')#,label=f'{bandpasses[band_i]} true')
        error_color = colors[0] #band_color[band_i]
        axes[band_i].errorbar(days, 
                            ys_val_temp_obs[:,band_i],       
                            yerr=ys_val_temp_std[:,band_i],
                            fmt='o',
                            markersize=2,
                            mfc=error_color,
                            mec=error_color,
                            ecolor=error_color,
                            elinewidth=1,
                            capsize=2,
                            label = 'obsserved')
                            #label=f'{bandpasses[band_i]} obs.')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(days.min(),days.max())
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
        if band_i == 0:
            axes[band_i].legend(fontsize=10,loc='upper left',ncol=3)
        axes[band_i].invert_yaxis()

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    #plt.ylabel(ylabel)
    
    fig.tight_layout()
    if (epoch+1) == num_epoch:
        plt.savefig(f"{save_path}/recovery/recovery_sigma_seperate_num_{num}_epoch_{epoch}.{save_file_format}",bbox_inches='tight')
    plt.savefig(f"{save_path}/recovery/recovery_sigma_seperate_num_{num}_epoch_{epoch}.png",bbox_inches='tight')
    plt.close()

    if (epoch+1) == num_epoch:
        

        # make plot for each band seperately except now different axes for each band
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for band_i in range(num_bands):
            #axes[band_i].plot(days, recovery_mean[:,band_i],color=band_color[band_i],linewidth=1.1,label=f'{bandpasses[band_i]} band')#,label='mean')
            axes[band_i].plot(days, recovery_mean[:,band_i],color=colors[1],linewidth=1.1,label='mean pred')#,label='mean')
            alpha_list = [0.3,0.2,0.1]
            num_sigma_list = [1,2,3]
            for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                axes[band_i].fill_between(days, 
                                recovery_mean[:,band_i]-num_sigma*recovery_std[:,band_i],
                                recovery_mean[:,band_i]+num_sigma*recovery_std[:,band_i],
                                color=colors[1], #band_color[band_i],
                                alpha=alpha, 
                                label=f"{num_sigma}$\sigma$ unc.")
            
            axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
                            linestyle='--',label='truth')#,label=f'{bandpasses[band_i]} true')
            error_color = colors[0] #band_color[band_i]
            axes[band_i].errorbar(days, 
                                ys_val_temp_obs[:,band_i],       
                                yerr=ys_val_temp_std[:,band_i],
                                fmt='o',
                                markersize=2,
                                mfc=error_color,
                                mec=error_color,
                                ecolor=error_color,
                                elinewidth=1,
                                capsize=2,
                                label = 'obsserved')
                                #label=f'{bandpasses[band_i]} obs.')
            axes[band_i].set_ylim(ylim)
            axes[band_i].invert_yaxis()
            axes[band_i].set_xlim(days.min(),days.max())
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
            if band_i == 0:
                axes[band_i].legend(fontsize=10,loc='best',ncol=3)

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel(ylabel)
        
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/recovery_bands_sep_num_{num}_epoch_{epoch}.{save_file_format}",bbox_inches='tight')
        plt.close()

        # make a plot only of the context points
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for band_i in range(num_bands):
            # plot the truth
            #axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
            #                linestyle='--',label='truth')
            error_color = colors[0] #band_color[band_i]
            axes[band_i].errorbar(days, 
                                ys_val_temp_obs[:,band_i],       
                                yerr=ys_val_temp_std[:,band_i],
                                fmt='o',
                                markersize=2,
                                mfc=error_color,
                                mec=error_color,
                                ecolor=error_color,
                                elinewidth=1,
                                capsize=2,
                                label = 'obsserved')
            axes[band_i].set_ylim(ylim)
            axes[band_i].set_xlim(days.min(),days.max())
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
            if band_i == 0:
                axes[band_i].legend(fontsize=11,loc='upper left')
            axes[band_i].invert_yaxis()

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel(ylabel)
        
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/context_points_only_num_{num}_epoch_{epoch}.{save_file_format}",bbox_inches='tight')
        plt.close()

        # make a plot only of the context points
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for band_i in range(num_bands):
            # plot the truth
            axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
                            linestyle='--',label='truth')
            error_color = colors[0] #band_color[band_i]
            axes[band_i].errorbar(days, 
                                ys_val_temp_obs[:,band_i],       
                                yerr=ys_val_temp_std[:,band_i],
                                fmt='o',
                                markersize=2,
                                mfc=error_color,
                                mec=error_color,
                                ecolor=error_color,
                                elinewidth=1,
                                capsize=2,
                                label = 'obsserved')
            axes[band_i].set_ylim(ylim)
            axes[band_i].set_xlim(days.min(),days.max())
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
            if band_i == 0:
                axes[band_i].legend(fontsize=11,loc='upper left',ncol=2)
            axes[band_i].invert_yaxis()

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel(ylabel)
        
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/context_points_and_truth_num_{num}_epoch_{epoch}.{save_file_format}",bbox_inches='tight')
        plt.close()

    
        # make plot only of the recovery
        fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
        fig.add_subplot(111, frameon=False)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for band_i in range(num_bands):
            #axes[band_i].plot(days, recovery_mean[:,band_i],color=band_color[band_i],linewidth=1.1,label=f'{bandpasses[band_i]} band')#,label='mean')
            axes[band_i].plot(days, recovery_mean[:,band_i],color=colors[1],linewidth=1.1,label='mean pred')#,label='mean')
            alpha_list = [0.3,0.2,0.1]
            num_sigma_list = [1,2,3]
            for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                axes[band_i].fill_between(days, 
                                recovery_mean[:,band_i]-num_sigma*recovery_std[:,band_i],
                                recovery_mean[:,band_i]+num_sigma*recovery_std[:,band_i],
                                color=colors[1], #band_color[band_i],
                                alpha=alpha, 
                                label=f"{num_sigma}$\sigma$ unc.")
                
            axes[band_i].set_ylim(ylim)
            axes[band_i].set_xlim(days.min(),days.max())
            axes[band_i].minorticks_on()
            axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
            if band_i == 0:
                axes[band_i].legend(fontsize=11,loc='upper left',ncol=4)
            axes[band_i].invert_yaxis()

        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        #plt.ylabel(ylabel)
        
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/recovery_only_num_{num}_epoch_{epoch}.{save_file_format}",bbox_inches='tight')
        plt.close()
    
    if (epoch+1) == num_epoch:
        #plot the power spectrum of the light curve
        plot_power_spectrum(true_LC_temp,recovery_mean,num,epoch)
        # This doesn't really work how you would expect drawing from the uncertainty changes the power spectrum
        #plot_power_spectrum_with_uncertainty(true_LC_temp,combined_samples,num,epoch)

        #plot the autocorrelation function of the light curve
        plot_ACF_and_SF(true_LC_temp,recovery_mean,num,epoch,SFinf,tau)
        if GPR_mean is not None and GPR_std is not None:
            plot_ACF_and_SF_latent_vs_GPR(true_LC_temp,recovery_mean,GPR_mean,num,epoch,SFinf,tau)
        # This doesn't really work how you would expect drawing from the uncertainty changes the autocorrelation function
        #plot_ACF_and_SF_with_uncertainty(true_LC_temp,combined_samples,num,epoch,SFinf,tau)

def enable_dropout(model):
    """ 
    Function to enable the dropout layers during inference. 

    model: torch model to enable dropout
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def disable_dropout(model):
    """ 
    Function to disable the dropout layers during inference. 

    model: torch model to enable dropout
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

class LinearScheduler(object):
    def __init__(self, max_epoch, maxval=1.0, minval=1e-3):
        self._max_epoch = max_epoch
        self._maxval = maxval
        self._minval = minval if max_epoch > 0 else maxval
        self._val = self._minval
        self.epoch = 0

    def step(self):
        if self.epoch < self._max_epoch:
            self._val = max(min(self.epoch/self._max_epoch,1.0)*self._maxval, self._minval)
        else:
            self._val = self._maxval
        self.epoch += 1

    @property
    def val(self):
        return self._val

def softmax(x):
    x = np.exp(x)
    x /= np.sum(x)
    return x

@torch.enable_grad()
def gaussian_process_regression(LC,LC_std,device,mean_across_bands=False,min_std=0.01):
    """
    LC = light curve to be normalized, shape (num_time_steps,num_bands)
    LC_std = standard deviation of the light curve, shape (num_time_steps,num_bands)
    device = torch.device 
    mean_across_bands = whether to average the mean and std across bands or normalize each band separately
    min_std = minimum standard deviation to use for the GP in magnitude.
    """
    # use cpu
    device = torch.device('cpu')

    num_time_steps = LC.shape[0]

    #Mask out the zero values
    mask = (LC != 0.0).type_as(LC) 
    
    epsilon = 1e-6
    #get masked mean weighted by the photometric noise
    LC_mean_per_band = torch.sum(LC/(LC_std+epsilon)**2,dim=0)/torch.sum(mask/(LC_std+epsilon)**2,dim=0)

    #get masked std weighted by the photometric noise
    mean_diff = mask*(LC - LC_mean_per_band)**2
    LC_std_per_band = torch.sqrt(torch.sum(mean_diff/(LC_std+epsilon)**2,dim=0)/torch.sum(mask/(LC_std+epsilon)**2,dim=0))

    # get mean and std averaged over all bands
    if mean_across_bands:
        LC_mean = LC_mean_per_band.mean(dim=-1)
        LC_mean = LC_mean.unsqueeze(-1) #add a dimension to make it broadcastable
        LC_pred_std = LC_std_per_band.mean(dim=-1)
        LC_pred_std = LC_pred_std.unsqueeze(-1) #add a dimension to make it broadcastable
    else:
        LC_mean = LC_mean_per_band
        LC_pred_std = LC_std_per_band

    # Normalize the light curve to have zero mean and unit variance
    LC = mask*((LC - LC_mean)/LC_pred_std)
    LC_std = LC_std/LC_pred_std

    num_tasks = LC.shape[-1]

    x_list = []
    y_list = []
    y_var_list = []
    for i in range(num_tasks):
        x = LC[:,i].nonzero()[:,0]
        y = LC[x,i]
        
        # ADD?
        y_std = LC_std[x,i] #add a small number to avoid numerical issues

        x = x/(LC.shape[0]-1) #normalize the x values to be between 0 and 1
        x = x.type_as(LC)
        y_var_list.append(y_std**2)
        x = x[:,None] #add a dimension to make it a 2D tensor
        i = i*torch.ones(x.shape)
        x_list.append(torch.cat([x,i],-1))
        y_list.append(y)

    del LC, LC_std, mask, mean_diff, LC_mean_per_band, LC_std_per_band

    train_X = torch.cat(x_list)
    train_Y = torch.cat(y_list,-1).unsqueeze(-1)
    train_Yvar = torch.cat(y_var_list,-1).unsqueeze(-1)    
    
    train_X = train_X.to(device,dtype=torch.float64)
    train_Y = train_Y.to(device,dtype=torch.float64)
    train_Yvar = train_Yvar.to(device,dtype=torch.float64)

    #Define the model
    model = botorch.models.FixedNoiseMultiTaskGP(
        train_X, train_Y, train_Yvar, task_feature=-1,
        covar_module=gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.MaternKernel(nu=0.5,ard_num_dims=train_X.shape[-1]))
        ).to(device)

    mll = gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()

    botorch.fit.fit_gpytorch_model(mll,options={"maxiter": 10000},num_restarts=10)

    del train_X, train_Y, train_Yvar
    del x_list, y_list, y_var_list

    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        model.eval()
  
        X_test_time = torch.linspace(0.0, 1.0, num_time_steps).unsqueeze(-1).repeat(1, num_tasks).view(-1, 1).to(device, dtype=torch.float64)
        X_test_task = torch.arange(num_tasks).unsqueeze(0).repeat(num_time_steps, 1).view(-1, 1).to(device, dtype=torch.float64)
        X_test = torch.cat([X_test_time, X_test_task], dim=-1)

        eval = model(X_test)
        mean = eval.mean.view(num_time_steps, num_tasks).detach().cpu().numpy()
        std = eval.variance.view(num_time_steps, num_tasks).sqrt().detach().cpu().numpy()

        del X_test, X_test_time, X_test_task, eval

        # Unnormalize the results
        mean = mean * LC_pred_std.detach().cpu().numpy() + LC_mean.detach().cpu().numpy() 
        std = std * LC_pred_std.detach().cpu().numpy()

        #clip the std to avoid outliers when the std is very small. Minimum std is 0.01.
        std = np.clip(std,min_std,100.0)  


    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    lengthscale = lengthscale[0,0]

    # Get tau and SF_inf from the GP
    tau = lengthscale * (num_time_steps-1) * cadence #convert the lengthscale to the original scale
    
    scale = model.covar_module.outputscale.detach().cpu().numpy()

    scale = scale*LC_pred_std.detach().cpu().numpy() # convert from variance to stdev and unnormaize
    
    scale = scale.sum() # sum over all bands
    SFinf = scale
    #SFinf = scale * np.sqrt(2) 

    # k(x, x') = exp(-||x - x'|| / lengthscale) same as tau

    #scale = model.covar_module.outputscale.detach().cpu().numpy()
    #SF_infinty = np.sqrt(scale) * np.sqrt(2.0)  
 
    del model            

    return mean, std, tau, SFinf

def plot_gaussian_process_regression(mean,std,true_LC_temp,ys,mag_mean,num=0):
    # set the ylim for plotting
    mean = np.copy(mean) + mag_mean
    std = np.copy(std)
    lower_lim = min(mean.min(),true_LC_temp.min())
    upper_lim = max(mean.max(),true_LC_temp.max())
    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    # eliminate the points that are not observed by setting them to np.nan
    ys_val_temp_obs = np.copy(ys)[:,:num_bands] + mag_mean
    ys_val_temp_obs[ys_val_temp_obs == mag_mean] = np.nan 
    ys_val_temp_std = np.copy(ys)[:,num_bands:]
    ys_val_temp_std[ys_val_temp_std == 0] = np.nan
    
    #set the y label

    ylabel = 'magntiude'
    
    # make plot for each band seperately
    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for band_i in range(num_bands):
        #axes[band_i].plot(days, recovery_mean[:,band_i],color=band_color[band_i],linewidth=1.1,label=f'{bandpasses[band_i]} band')#,label='mean')
        axes[band_i].plot(days, mean[:,band_i],color=colors[1],linewidth=1.1,label='mean pred')#,label='mean')
        alpha_list = [0.3,0.2,0.1]
        num_sigma_list = [1,2,3]
        for alpha,num_sigma in zip(alpha_list,num_sigma_list):
            axes[band_i].fill_between(days, 
                            mean[:,band_i]-num_sigma*std[:,band_i],
                            mean[:,band_i]+num_sigma*std[:,band_i],
                            color=colors[1], #band_color[band_i],
                            alpha=alpha, 
                            label=f"{num_sigma}$\sigma$ unc.")
        
        axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
                        linestyle='--',label='truth')#,label=f'{bandpasses[band_i]} true')
        error_color = colors[0] #band_color[band_i]
        axes[band_i].errorbar(days, 
                              ys_val_temp_obs[:,band_i],       
                              yerr=ys_val_temp_std[:,band_i],
                              fmt='o',
                              markersize=2,
                              mfc=error_color,
                              mec=error_color,
                              ecolor=error_color,
                              elinewidth=1,
                              capsize=2,
                            label = 'obsserved')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(days.min(),days.max())
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
        if band_i == 0:
            axes[band_i].legend(fontsize=10,loc='upper left',ncol=3)
        axes[band_i].invert_yaxis()

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    #plt.ylabel(ylabel)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/GPR_recovery_{num}.pdf",bbox_inches='tight')
    plt.close()

def compare_gaussian_process_regression_and_SDE(recovery_mean,recovery_std,mean,std,true_LC_temp,ys,mag_mean,num=0):
    # set the ylim for plotting
    mean = np.copy(mean)+mag_mean
    std = np.copy(std)
    recovery_mean = np.copy(recovery_mean)+mag_mean
    recovery_std = np.copy(recovery_std)

    lower_lim = min(mean.min(),recovery_mean.min(),true_LC_temp.min())
    upper_lim = max(mean.max(),recovery_mean.max(),true_LC_temp.max())
    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    # eliminate the points that are not observed by setting them to np.nan
    ys_val_temp_obs = np.copy(ys)[:,:num_bands] + mag_mean
    ys_val_temp_obs[ys_val_temp_obs == mag_mean] = np.nan 
    ys_val_temp_std = np.copy(ys)[:,num_bands:]
    ys_val_temp_std[ys_val_temp_std == 0] = np.nan
    
    #set the y label

    ylabel = 'magntiude'
    
    # make plot for each band seperately
    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for band_i in range(num_bands):
        axes[band_i].plot(days, recovery_mean[:,band_i],color=colors[1],linewidth=1.1,label='SDE mean')
        alpha = 0.2
        num_sigma = 2
        axes[band_i].fill_between(days, 
                        recovery_mean[:,band_i]-num_sigma*recovery_std[:,band_i],
                        recovery_mean[:,band_i]+num_sigma*recovery_std[:,band_i],
                        color=colors[1], #band_color[band_i],
                        alpha=alpha, 
                        label=f"{num_sigma}$\sigma$ unc.")

        axes[band_i].plot(days, mean[:,band_i],color=colors[2],linewidth=1.1,label='GPR mean')
        axes[band_i].fill_between(days, 
                        mean[:,band_i]-num_sigma*std[:,band_i],
                        mean[:,band_i]+num_sigma*std[:,band_i],
                        color=colors[2], #band_color[band_i],
                        alpha=alpha, 
                        label=f"{num_sigma}$\sigma$ unc.")
        
        axes[band_i].plot(days,true_LC_temp[:,band_i],color='black',linewidth=0.9,
                          linestyle='--',label='truth')

        axes[band_i].errorbar(days, 
                              ys_val_temp_obs[:,band_i],       
                              yerr=ys_val_temp_std[:,band_i],
                              fmt='o',
                              markersize=2,
                              mfc=colors[0],
                              mec=colors[0],
                              ecolor=colors[0],
                              elinewidth=1,
                              capsize=2,
                            label = 'obsserved')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(days.min(),days.max())
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band mag')
        if band_i == 0:
            axes[band_i].legend(fontsize=10,loc='upper left',ncol=3)
        axes[band_i].invert_yaxis()

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/Compare_SDE_GPR_{num}.pdf",bbox_inches='tight')
    plt.close()


def main(
        create_data_before_training,                 # if True, create data before training. Have to change this in main. Creating before training is faster since we don't remake the data every epoch.
        batch_size=50,                              # batch size for training
        latent_size=8,                               # Latent size of the model. 
        context_size=64,                             # Context size in encoder. This takes up a lot of gpu memory so I am keeping it at 32
        hidden_size=128,                              # Hidden size of NN
        dt=1.0e-3,                                   # This is the integration time step with t-axis in the range of 0 to 1. Probably want to get this to 1e-3 but it takes a long time to train.
        anneal_dt=False,                             # Whether to anneal the dt during training. Just to speed up training.
        lr_init=2.5e-3,                             # Initial learning rate. 
        lr_gamma=0.97,                               # learning rate decay factor. Probably want to change depending on how many epochs we are training for.
        num_epoch=100,                                # number of epochs to train! CHANGE TO DESIRED NUMBER.
        uncertainty_output=True,                     # Whether to obtain additional gaussian uncertainty of the light curve. No longer self supervised though.
        self_supervised=False,                        # If true then we only use the LSST cadences to train for LC reconstruction. If false we supervise using the full LC.
        kl_anneal_iters=15,                          # number of iterations to anneal KL term
        adjoint=False,                               # Use adjoint method of SDE
        include_prior_drift=True,                    # Include the neural SDE in the parameter estimation.
        method="euler",                              # SDE solver method, either "euler", "milstein", or "srk".
        param_weight=1.0,                            # Weighting factor of the parameter estimation in the loss
        param_w_iters=15,                            # Anneal the param weight from 0 to 1 over this many iterations.
        normalize=True,                              # Normalize all the bands to have mean 0 and std 1
        num_recovery_LC=25,                           # This is just the number of LC to visualize during training
        mag_mean=mag_mean,                           # This is the mean of the magnitude. We subtract this off to unnormalize the data.
        dropout_rate=0.0,                            # Dropout rate
        MC_dropout=False,                            # Enable MC dropout during inference
        num_RNN_layers=3,                            # Number of RNN layers for encoder.
        RNN_decoder=True,                            # I added the ability to use a combined SDE/RNN decoder. Could work better than just SDE alone.
        sample_from_latent=False,                     # Sample from the latent space or not. If False then it will just use one sample.
        num_RNN_layers_decoder=3,                    # Number of RNN layers for decoder, only used if RNN_decoder is True!
        evaluate_LC_performance=True,                # This will calculate -log(P), MSE, MAE for the test set using the latent SDE
        evaluate_LC_performance_GPR=True,            # This will calculate -log(P), MSE, MAE for the test set using a GPR 
        grad_clip_value=0.5,                         # Applies an L2 norm clip to the gradients to prevent exploding gradients
        use_SDE=True,                                # Turn on or off the SDE. If False then it will just do parameter estimation.
        make_animation=True,                         # Make an animation of the SDE during training
        use_logit=True,                              # Parameter reconstruction on logit space or not
        min_std_recovery=0.01,                       # Minimum standard deviation for the recovery. This is for the GPR to avoid outliers from tiny uncertainty.
        load_model=False,                            # Load a model from a previous training session
):  
    assert type(create_data_before_training) == bool, 'create_data_before_training must be a boolean'
    assert type(batch_size) == int and batch_size > 0, 'batch_size must be a positive integer'
    assert type(latent_size) == int and latent_size > 0, 'latent_size must be a positive integer'
    assert type(context_size) == int and context_size > 0, 'context_size must be a positive integer'
    assert type(hidden_size) == int and hidden_size > 0, 'hidden_size must be a positive integer'
    assert type(dt) == float and 0.0 < dt < 1.0, 'dt must be a float between 0 and 1'
    if dt > 0.01:
        logging.warning('dt is large. Consider reducing it to 2e-3 or smaller.')
    #assert type(adaptive_dt) == bool, 'adaptive_dt must be a boolean'
    assert type(anneal_dt) == bool, 'anneal_dt must be a boolean'
    assert type(lr_init) == float and lr_init > 0.0, 'lr_init must be a positive float'
    if lr_init > 0.01:
        logging.warning('lr_init is large. Consider reducing it to 1e-2 or smaller to avoid exploding gradient and numerical issues.')
    assert type(lr_gamma) == float and 0.0 < lr_gamma <= 1.0, 'lr_gamma must be a float between 0 and 1'
    assert type(num_epoch) == int and num_epoch >= 0, 'num_epoch must be a positive integer'
    if num_epoch == 0:
        logging.warning('num_epoch is 0. No training will be performed!')
    assert type(uncertainty_output) == bool, 'uncertainty_output must be a boolean'
    if uncertainty_output == False:
        logging.warning('uncertainty_output is False. Uncertainty will only come from the latent space if sample_from_latent is True.')
    assert type(self_supervised) == bool, 'self_supervised must be a boolean'
    assert type(kl_anneal_iters) == int and kl_anneal_iters >= 0, 'kl_anneal_iters must be a positive integer'
    assert type(adjoint) == bool, 'adjoint must be a boolean'
    if adjoint == True:
        logging.warning('Adjoint is True. This will be much slower.')
    assert type(include_prior_drift) == bool, 'include_prior_drift must be a boolean'
    assert method in ['euler','milstein','srk'], 'Not a valid method. Method must be one of "euler", "milstein", or "srk"'
    assert type(param_weight) == float and param_weight >= 0.0, 'param_weight must be a non-negative float'
    if param_weight == 0.0:
        logging.warning('param_weight is 0. No parameter estimation will be learned!')
    #assert type(param_w_iters) == int and param_w_iters >= 0, 'param_w_iters must be a positive integer'
    assert type(param_w_iters) == int, 'param_w_iters must be an integer'
    assert type(normalize) == bool, 'normalize must be a boolean'
    if normalize == False:
        logging.warning('normalize is False. The data will not be normalized during training!')
    assert type(num_recovery_LC) == int and num_recovery_LC >= 0, 'num_recovery_LC must be a positive integer'
    if num_recovery_LC == 0:
        logging.warning('num_recovery_LC is 0. No light curves will be plotted during training!')
    assert type(use_logit) == bool, 'use_logit must be a boolean'
    assert type(min_std_recovery) == float and min_std_recovery >= 0.0, 'min_std_recovery must be a positive float'
    if min_std_recovery == 0.0:
        logging.warning('min_std_recovery is 0.0 which can cause numerical issues evaluating -log(P) for the GPR!')
    if min_std_recovery > 0.02:
        logging.warning('min_std_recovery is large. Consider reducing it to 0.01 or so.')
    if min_std_recovery < 0.005:
        logging.warning('min_std_recovery is small. Consider increasing it to 0.01 or so.')
    assert type(mag_mean) == float, 'mag_mean must be a float'
    assert type(dropout_rate) == float and 0.0 <= dropout_rate < 1.0, 'dropout_rate must be a float between 0 and 1'
    if dropout_rate > 0.5:
        logging.warning('Dropout_rate is too large. Consider reducing it to 0.5 or smaller.')
    assert type(MC_dropout) == bool, 'MC_dropout must be a boolean'
    assert type(num_RNN_layers) == int and num_RNN_layers > 0, 'num_RNN_layers must be a positive integer'
    assert type(RNN_decoder) == bool, 'RNN_decoder must be a boolean'
    if RNN_decoder == False and uncertainty_output == True:
        logging.warning('RNN_decoder is False and uncertainty_output is True. This esentially means the SDE needs to also model the uncertainty.')
    assert (type(num_RNN_layers_decoder) == int and num_RNN_layers_decoder > 0) or RNN_decoder == False, 'num_RNN_layers_decoder must be a positive integer'
    assert type(evaluate_LC_performance) == bool, 'evaluate_LC_performance, must be a boolean'
    assert type(evaluate_LC_performance_GPR) == bool, 'evaluate_LC_performance_GPR, must be a boolean'
    assert type(grad_clip_value) == float and grad_clip_value > 0.0, 'grad_clip_value must be a positive float'
    assert type(make_animation) == bool, 'make_animation must be sa boolean'
    assert type(use_SDE) == bool, 'use_SDE must be a boolean'
    if use_SDE == False:
        logging.warning('use_SDE is False. The model will not reconstruct the light curve, and will only estimate parameters.')
    assert type(load_model) == bool, 'load_model must be a boolean'
    if load_model == True:
        logging.warning('load_model is True. The model will be loaded from the specified path.')
    if torch.cuda.is_available():
        device_to_use = 'cuda' 
    else:
        device_to_use = 'cpu' 

    print(f'using device type: {device_to_use}')
    logging.warning(f'using device type: {device_to_use}')

    if device_to_use == 'cpu':
        logging.warning('Using cpu for training. This will be slow.')    

    device = torch.device(device_to_use)
    num_cpus_use = os.cpu_count() 
    print(f'using {num_cpus_use} cpus')
    logging.warning(f'using {num_cpus_use} cpus')

    if create_data_before_training:
        new_training_files = glob(f'{save_path}/train_data/*.npy')
        new_validation_files = glob(f'{save_path}/val_data/*.npy')
        new_test_files = glob(f'{save_path}/test_data/*.npy')

        assert(len(new_training_files) > 0), 'No training data found'
        assert(len(new_validation_files) > 0), 'No validation data found'
        assert(len(new_test_files) > 0),     'No test data found'

        train_dataset = Dataset_Loader(new_training_files,cadence,mag_mean,augment=True,save_true_LC=True,load_before=create_data_before_training)
        val_dataset = Dataset_Loader(new_validation_files,cadence,mag_mean,augment=False,save_true_LC=True,load_before=create_data_before_training)
        test_dataset = Dataset_Loader(new_test_files,cadence,mag_mean,augment=False,save_true_LC=True,load_before=create_data_before_training)
    else:
        #create a new training set each epoch on the fly. The validation and test sets are fixed by their random seed.
        train_dataset = Dataset_Loader(file_names_train,cadence,mag_mean,augment=True,save_true_LC=True,load_before=create_data_before_training)
        val_dataset = Dataset_Loader(file_names_val,cadence,mag_mean,augment=False,save_true_LC=True,load_before=create_data_before_training)
        test_dataset = Dataset_Loader(file_names_test,cadence,mag_mean,augment=False,save_true_LC=True,load_before=create_data_before_training)

    #train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,num_workers=max(int((1.-validation_rate)*num_cpus_use),1),pin_memory=True)
    #val_loader = DataLoader(val_dataset, shuffle=False,batch_size=batch_size,num_workers=max(int(validation_rate*num_cpus_use),1),pin_memory=True)
    #test_loader = DataLoader(test_dataset, shuffle=False,batch_size=batch_size,num_workers=max(int(validation_rate*num_cpus_use),1),pin_memory=True)

    train_loader = DataLoader(train_dataset, shuffle=True,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True)

    # create the model or load an existing models
    if load_model:
        logging.warning(f'loading model existing model from {save_path}/latent_sde.pth')
        logging.warning(f'model parameters not used')

        transfer_learning = True
        if transfer_learning:
            latent_sde = LatentSDE(
                                data_size=output_dim,  
                                latent_size=latent_size,
                                context_size=context_size,
                                hidden_size=hidden_size,
                                n_params=n_params,
                                device=device,
                                include_prior_drift=include_prior_drift,
                                dt=dt,
                                uncertainty_output=uncertainty_output,
                                anneal_dt=anneal_dt,
                                num_epoch=num_epoch,
                                normalize=normalize,
                                dropout_rate=dropout_rate,
                                num_RNN_layers=num_RNN_layers,
                                RNN_decoder=RNN_decoder,
                                num_RNN_layers_decoder=num_RNN_layers_decoder,
                                use_SDE=use_SDE,
                                give_redshift=give_redshift, 
                                self_supervised=self_supervised,
                                ).to(device)
            
            old_model = torch.load(f'{save_path}/latent_sde.pth').to(device)

            # apply transfer learning to match the new model weights to the old model weights where possible
            for name, param in old_model.named_parameters():
                if name in latent_sde.state_dict():
                    if latent_sde.state_dict()[name].shape == param.shape:

                        latent_sde.state_dict()[name].copy_(param)
                    else:
                        
                        # use only the first part of the tensor if the new model is smaller than the old model otherwise use only the part that fits
                        if latent_sde.state_dict()[name].shape[0] < param.shape[0]:
                            latent_sde.state_dict()[name].copy_(param[:latent_sde.state_dict()[name].shape[0]])
                        else:
                            latent_sde.state_dict()[name].copy_(param)
                                                                
            del old_model
        
        else:
            latent_sde = torch.load(f'{save_path}/latent_sde.pth').to(device)


    else:
        latent_sde = LatentSDE(
            data_size=output_dim,  
            latent_size=latent_size,
            context_size=context_size,
            hidden_size=hidden_size,
            n_params=n_params,
            device=device,
            include_prior_drift=include_prior_drift,
            dt=dt,
            uncertainty_output=uncertainty_output,
            anneal_dt=anneal_dt,
            num_epoch=num_epoch,
            normalize=normalize,
            dropout_rate=dropout_rate,
            num_RNN_layers=num_RNN_layers,
            RNN_decoder=RNN_decoder,
            num_RNN_layers_decoder=num_RNN_layers_decoder,
            use_SDE=use_SDE,
            give_redshift=give_redshift, 
            self_supervised=self_supervised,
        ).to(device)

    n_params_NN = sum(p.numel() for p in latent_sde.parameters() if p.requires_grad)
    print(f"Number of params: {n_params_NN}")
    logging.warning(f"Number of params: {n_params_NN}")
    
    optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,lr_gamma)

    param_loss_fn = FullRankGaussianNLL(n_params,device)
    param_mse_metric = nn.MSELoss(reduction='mean')
    param_ngll_metric = FullRankGaussianNLL(n_params,device)

    kl_scheduler = LinearScheduler(max_epoch=kl_anneal_iters,maxval=1.0)
    param_w_scheduler = LinearScheduler(max_epoch=param_w_iters,maxval=1.0)

    recovery_figure_dict = dict()
    recovery_figure_seperate_dict = dict()
    recovery_figure_sigma_dict = dict()
    recovery_figure_sigma_seperate_dict = dict()
    recovery_figure_mean_dict = dict()
    
    mse_list = []
    val_mse_list = []
    ngll_list = []
    val_ngll_list = []
    recon_loss_list = []
    val_recon_loss_list = []

    use_SDE_original = use_SDE
    n_batches = len(train_loader)
    for epoch in tqdm(range(num_epoch)):
        latent_sde.set_epoch(epoch)
        latent_sde.train()

        # Do parameter prediction before the LC reconstruction to save training time
        if param_w_iters + epoch < 0 and use_SDE_original:
            latent_sde.toggle_SDE(False)
            use_SDE = False
        elif use_SDE_original:
            latent_sde.toggle_SDE(True)
            use_SDE = True

        mse_epoch = 0 
        ngll_epoch = 0
        log_pxs_epoch = 0
        recon_loss_epoch = 0
        loss_epoch = 0
        for i, batch in enumerate(train_loader):
            ys_batch = batch['y'].transpose(0, 1)  # [T, B, out_dim]
            true_LC = batch['true_LC'].transpose(0, 1)  # [T, B, out_dim]
            if give_redshift:
                redshift = batch['redshift'].float().to(device)
            else:
                redshift = None
            #delta = batch['delta'].transpose(0, 1)  # [T, B, out_dim]
            ts = batch['x'][-1, :]  # [T]
            param_labels = batch['params'].float().to(device)  # [B, n_params]
            latent_sde.zero_grad()
            log_pxs, log_ratio, param_pred, _xs = latent_sde(ys_batch.to(device),
                                                             true_LC.to(device)-mag_mean,
                                                             ts.to(device),
                                                             redshift=redshift,
                                                             adjoint=adjoint,
                                                             method=method)

            if use_logit:
                param_loss = param_loss_fn(param_pred, logit(param_labels))
                ngll = param_ngll_metric(expit(param_pred),param_labels)
                mse = param_mse_metric(expit(param_pred[:,:n_params]), param_labels)
            else:
                param_loss = param_loss_fn(param_pred, param_labels)
                ngll = param_ngll_metric(param_pred,param_labels)
                mse = param_mse_metric(param_pred[:,:n_params], param_labels)

            if use_SDE:
                recon_loss = log_pxs + log_ratio * kl_scheduler.val
                log_pxs_epoch += log_pxs.detach().cpu().item()
                recon_loss_epoch += (log_pxs + log_ratio).detach().cpu().item()
                loss = recon_loss + param_weight*param_w_scheduler.val*param_loss
            else:
                loss = param_loss
            ngll_epoch += ngll.detach().cpu().item()
            mse_epoch += mse.detach().cpu().item()
            
            loss.backward()
            #torch.nn.utils.clip_grad_value_(latent_sde.parameters(), grad_clip_value)
            torch.nn.utils.clip_grad_norm_(latent_sde.parameters(),grad_clip_value)  # clip gradients to avoid exploding gradients problem in RNNs
            optimizer.step()
        if use_SDE:
            # Only update the kl annealing if the SDE is used
            kl_scheduler.step()
        param_w_scheduler.step()
        latent_sde.eval()
        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step()

        if use_SDE:
            logging.warning(
                f'epoch: {epoch}, lr: {lr_now:.5f}, kl_coeff: {kl_scheduler.val:.4f}'
            )
        else:
            logging.warning(
                f'epoch: {epoch}, lr: {lr_now:.5f}'
            )

        torch.save(latent_sde, f"{save_path}/latent_sde.pth")
        # Visualization during training
        with torch.no_grad():
            #average across each batch
            mse_epoch /= n_batches
            ngll_epoch /= n_batches
            recon_loss_epoch /= n_batches
            log_pxs_epoch /= n_batches
            loss_epoch /= n_batches

            val_mse_epoch = 0
            val_param_ngll_epoch = 0
            val_recon_loss_epoch = 0

            val_loss_epoch = 0
            val_neg_log_pxs_epoch = 0
            val_kl_term_epcoh = 0
            val_param_ngll_epoch = 0

            for i, val_batch in enumerate(val_loader):
                ys_batch = val_batch['y'].transpose(0, 1)  
                #delta = val_batch['delta'].transpose(0, 1)
                ts = batch['x'][-1, :]  # [T]
                param_labels = val_batch['params'].float().to(device)  # [B, n_params]
                true_LC = val_batch['true_LC'].transpose(0,1) #true LC with no noise or missing values
                if give_redshift:
                    redshift = val_batch['redshift'].float().to(device)
                else:
                    redshift = None
                log_pxs, log_ratio, param_pred, _xs = latent_sde(ys_batch.to(device),
                                                                 true_LC.to(device)-mag_mean,
                                                                 ts.to(device),
                                                                 redshift=redshift,
                                                                 adjoint=adjoint,
                                                                 method=method)

                if use_logit:
                    param_pred = expit(param_pred)    
                val_mse = param_mse_metric(param_pred[:,:n_params], param_labels)            

                
                param_loss = param_loss_fn(param_pred, param_labels)
                if use_SDE:
                    recon_loss = log_pxs + log_ratio * kl_scheduler.val
                    val_loss = recon_loss + param_weight*param_loss    
                    unw_val_loss = (recon_loss + param_weight*param_loss).detach().cpu().item()
                    val_loss=float(val_loss.detach().cpu().item())
                    val_neg_log_pxs=float((log_pxs).detach().cpu().item())
                    val_kl_term=float((log_ratio*kl_scheduler.val).detach().cpu().item())
                    val_param_ngll=float((param_ngll_metric(param_pred, param_labels)).detach().cpu().item())
                    val_mse=float((val_mse).detach().cpu().item())
                    val_recon_loss = ((log_pxs + log_ratio).detach().cpu().item())
                else:
                    val_loss = param_loss
                    unw_val_loss = param_loss.detach().cpu().item()
                    val_loss=float(val_loss.detach().cpu().item())
                    val_param_ngll=float((param_ngll_metric(param_pred, param_labels)).detach().cpu().item())
                    val_mse=float((val_mse).detach().cpu().item())
                    val_neg_log_pxs = 0
                    val_kl_term = 0
                    val_recon_loss = 0

                val_mse_epoch += val_mse
                val_param_ngll_epoch += val_param_ngll
                val_recon_loss_epoch += val_recon_loss
                
                val_loss_epoch += val_loss
                val_neg_log_pxs_epoch += val_neg_log_pxs
                val_kl_term_epcoh += val_kl_term
                val_param_ngll_epoch += val_param_ngll

            #average across each batch
            val_mse_epoch /= n_batches
            val_param_ngll_epoch /= n_batches
            val_recon_loss_epoch /= n_batches

            val_loss_epoch /= n_batches
            val_neg_log_pxs_epoch /= n_batches
            val_kl_term_epcoh /= n_batches
            val_param_ngll_epoch /= n_batches

            #print training metrics
            print(f"mse: {round(mse_epoch,4)},  ngll: {round(ngll_epoch,4)}, neg_log_pxs: {round(log_pxs_epoch,4)}, recon_loss: {round(recon_loss_epoch,4)}")
            print(f"val_mse: {round(val_mse,4)},  val_ngll: {round(val_param_ngll,4)}, val_neg_log_pxs: {round(val_neg_log_pxs,4)},  val_recon_loss: {round(val_recon_loss,4)}")
            print()

            logging.warning(f"mse: {round(mse_epoch,4)},  ngll: {round(ngll_epoch,4)}, neg_log_pxs: {round(log_pxs_epoch,4)}, recon_loss: {round(recon_loss_epoch,4)}")
            logging.warning(f"val_mse: {round(val_mse,4)},  val_ngll: {round(val_param_ngll,4)}, val_neg_log_pxs: {round(val_neg_log_pxs,4)},  val_recon_loss: {round(val_recon_loss,4)}")
            

            #for plotting the metrics vs epoch
            mse_list.append(mse_epoch)
            ngll_list.append(ngll_epoch)
            recon_loss_list.append(recon_loss_epoch)

            val_mse_list.append(val_mse)
            val_ngll_list.append(val_param_ngll)
            val_recon_loss_list.append(val_recon_loss)

            #just gets the first batch from the test_loader. So same as corner plots in test set!
            for val_batch in test_loader:
                ys_val = val_batch['y'].transpose(0,1)
                #delta_val = val_batch['delta'].transpose(0,1) 
                ts_val =  val_batch['x'][-1, :]  # [T]
                true_LC = val_batch['true_LC'].transpose(0,1)

                parameters_val = val_batch['params'].float().cpu().numpy()

                if give_redshift:
                    redshift = val_batch['redshift'].float().to(device)
                else:
                    redshift = None

                break

            true_LC = true_LC.cpu().numpy()
            
            # set dropout layers to train mode
            if MC_dropout:
                enable_dropout(latent_sde)

            band_color = ['violet', 'green', 'red' , 'orange', 'blue', 'purple']

            if (make_animation or epoch == num_epoch-1) and use_SDE:
                for num in range(min(num_recovery_LC,true_LC.shape[1])): # plot num_recovery_LC reconstructions (unless the batch size is smaller than num_recovery_LC)
                    ys_val_num = torch.clone(ys_val)[:,num:num+1,:] #This is the observed LC with noise and missing values
                    if give_redshift:                   
                        redshift_num = torch.clone(redshift)[num:num+1]
                    else:
                        redshift_num = None
                    true_LC_temp = np.copy(true_LC)[:,num] #This is the true LC without noise and with all points
                    
                    #sample 100 reconstructed LCs from the posteriors
                    if epoch == num_epoch-1:
                        num_samples = 250 
                    else:
                        num_samples = 60
                    # if we don't sample from the latent then we can just use one sample with mean and std
                    if sample_from_latent == True or uncertainty_output == False:
                        for i in range(num_samples):
                            #sample from the posterior
                            sample = latent_sde.sample_posterior(ys_val_num.to(device),
                                                                ts_val.to(device),
                                                                redshift=redshift_num,
                                                                adjoint=adjoint,
                                                                method='euler')
                            if uncertainty_output:
                                recovery_mean = sample[:,0,:num_bands].detach().cpu().numpy() 
                                recovery_std = np.sqrt(np.exp(sample[:,0,num_bands:].detach().cpu().numpy())) # convert log variance to std
                                sample = np.random.normal(recovery_mean, recovery_std) # [T_vis, n_bandpasses]
                            else:
                                sample = sample[:,0,:num_bands].detach().cpu().numpy()
                            sample = np.expand_dims(sample,axis=2) # [T_vis, n_bandpasses, 1]

                            # combined_samples has shape [T_vis, n_bandpasses, n_sample]
                            if i == 0: 
                                combined_samples = np.copy(sample) 
                            else:
                                combined_samples = np.concatenate((combined_samples,sample),axis=2) 
                        del sample
                        recovery_mean = np.mean(combined_samples,axis=2)
                        recovery_std = np.std(combined_samples,axis=2)
                    else:
                        #sample from the posterior
                        sample = latent_sde.sample_posterior(ys_val_num.to(device),
                                                            ts_val.to(device),
                                                            redshift=redshift_num,
                                                            adjoint=adjoint,
                                                            method='euler')
                        recovery_mean = sample[:,0,:num_bands].detach().cpu().numpy() 
                        recovery_std = np.sqrt(np.exp(sample[:,0,num_bands:].detach().cpu().numpy())) # convert log variance to std
                        del sample
                    # The observed LC as a numpy array
                    ys_val_temp = np.copy(ys_val.detach().cpu().numpy())[:,num,:]

                    #get SFinf and tau from parameters_val

                    for i,par in enumerate(parameters_keys):
                        if par == 'SFinf':
                            SFinf = parameters_val[num,i]
                        elif par == 'log_tau':
                            log_tau = parameters_val[num,i]

                    #convert the SFinf and tau to mag_mean and mag_std

                    SFinf = min_max_dict['SFinf'][0] + SFinf*(min_max_dict['SFinf'][1]-min_max_dict['SFinf'][0])
                    log_tau = min_max_dict['log_tau'][0] + log_tau*(min_max_dict['log_tau'][1]-min_max_dict['log_tau'][0])
                    
                    tau = 10.0**log_tau #convert from log10(tau) to tau

                    if epoch == num_epoch-1:
                        mean, std, tau_pred, SFinf_pred = gaussian_process_regression(torch.clone(ys_val[:,num,:num_bands]),
                                                                                    torch.clone(ys_val[:,num,num_bands:]),
                                                                                    device=device,
                                                                                    min_std=min_std_recovery)         

                        ys = ys_val[:,num].detach().cpu().numpy()
                        plot_gaussian_process_regression(mean,std,true_LC_temp,ys,mag_mean,num=num)
                        compare_gaussian_process_regression_and_SDE(recovery_mean,recovery_std,mean,std,true_LC_temp,ys,mag_mean,num=num)
                    else:
                        # then don't plot the GP regression SF compared to the SDE SF
                        mean = None
                        std = None
                    # plots recovery
                    # if epoch == num_epoch-1: then also plots SF, ACF, power spectrum
                    plot_reconstruction(ys_val_temp,true_LC_temp,recovery_mean,recovery_std,band_color,mag_mean,epoch,num_epoch,num,SFinf,tau,GPR_mean=mean,GPR_std=std)

    #save the trained torch model
    latent_sde.eval()
    torch.save(latent_sde, f"{save_path}/latent_sde.pth")

    with torch.no_grad():
        # plots of metrics vs epochs
        plot_metric_vs_epoch(mse_list,val_mse_list,"MSE")
        plot_metric_vs_epoch(ngll_list,val_ngll_list,"NGLL")
        plot_metric_vs_epoch(recon_loss_list,val_recon_loss_list,"reconstructed loss")

        # makes gifs of the LC reconstructions
        if make_animation and use_SDE:
            for num in range(num_recovery_LC):
                try:
                    make_gifs(f'recovery_sigma_seperate_num_{num}',num_epoch)
                except:
                    pass

        val_mse_recovery_epoch = []
        val_mae_recovery_epoch = []
        val_log_P_epoch = []
        LC_reconstruction_mean_diff = [] #list of the mean of the LC reconstructions for each LC in the validation set

        val_mse_recovery_GPR_epoch = []
        val_mae_recovery_GPR_epoch = []
        val_log_P_GPR_epoch =  []

        num_sigma_1_recovery_epoch = 0
        num_sigma_2_recovery_epoch = 0
        num_sigma_3_recovery_epoch = 0

        num_sigma_1_recovery_GPR_epoch = 0
        num_sigma_2_recovery_GPR_epoch = 0
        num_sigma_3_recovery_GPR_epoch = 0

        #metric for parameter predicitons uncertainty i.e. coverage probabilities
        coverage_prob_dict = dict()
        quantiles = [0.638,0.955,0.997]
        for quantile in quantiles:
            coverage_prob_dict[quantile] = dict()
            for par in parameters_keys:
                coverage_prob_dict[quantile][par] = 0

        all_coverage_prob_dict = dict()
        num_quantiles = 50 #resolution of the quantiles for graphing the coverage probabilities
        all_quantiles = np.linspace(0.0,1.0,num_quantiles+1)
        for quantile in all_quantiles:
            all_coverage_prob_dict[quantile] = dict()
            for par in parameters_keys:
                all_coverage_prob_dict[quantile][par] = 0
        
            all_coverage_prob_dict[quantile]['recovery'] = 0
            all_coverage_prob_dict[quantile]['recovery_GPR'] = 0

        # set dropout layers to train mode
        if MC_dropout:
            enable_dropout(latent_sde)

        #calculate metrics for the recovery LCs and parameters
        i = 0
        total_val = 0
        print("Calculating metrics for the recovery LCs and parameters")
        pbar = tqdm(total=len(test_loader))

        tau_pred_list = []
        SFinf_pred_list = []

        for val_batch in tqdm(test_loader):

            ys_batch = val_batch['y'].transpose(0,1) # [T, B, n_band]
            ts_batch = val_batch['x'][-1,:] # T
            true_LC = val_batch['true_LC'].transpose(0,1) # [T, B, n_band]
            truth = val_batch['params'].float().cpu().numpy()
            if give_redshift:
                redshift = val_batch['redshift'].float().to(device)
            else:
                redshift = None
            if MC_dropout:
                num_MC = 40
                samples = 10_000
            else:
                num_MC = 1
                samples = 400_000

            mean_list = []
            for j in range(num_MC):
                
                log_pxs, log_ratio, param_pred, _xs = latent_sde(ys_batch.to(device),
                                                                 true_LC.to(device)-mag_mean,
                                                                 ts_batch.to(device),
                                                                 redshift=redshift,
                                                                 adjoint=adjoint,
                                                                 method=method)
                mu = param_pred[:,:n_params]
                tril_elements = param_pred[:,n_params:]
                tril = torch.zeros([tril_elements.shape[0], n_params,n_params]).to(device).type_as(tril_elements)
                tril_idx = torch.tril_indices(n_params, n_params, offset=0).to(device)
                tril[:,tril_idx[0],tril_idx[1]] = tril_elements
                log_diag_tril = torch.diagonal(tril, offset=0, dim1=1, dim2=2)  # [batch_size, Y_dim]
                tril[:, torch.eye(n_params, dtype=bool)] = torch.exp(log_diag_tril)
                prec_mat = torch.bmm(tril, torch.transpose(tril, 1, 2)) 
                cov = torch.inverse(prec_mat).detach().cpu().numpy()

                for k in range(len(truth)):
                    if k == 0:
                        sample_val = np.random.multivariate_normal(mu[k].detach().cpu().numpy(),cov[k],size=samples)
                        sample_val = np.expand_dims(sample_val,axis=0)
                    else:
                        _sample_val = np.random.multivariate_normal(mu[k].detach().cpu().numpy(),cov[k],size=samples)
                        _sample_val = np.expand_dims(_sample_val,axis=0)
                        sample_val = np.concatenate((sample_val,_sample_val),axis=0)
                if use_logit:
                    sample_val = expit(sample_val,numpy=True)

                if j == 0:
                    sample = np.copy(sample_val)
                else:
                    sample = np.concatenate((sample,sample_val),axis=1)
            
            for k in range(len(truth)):
                mean = np.mean(sample[k],axis=0)
                mean_list.append(mean)            
            
            if i == 0:
                param_labels = np.copy(truth)
            else:
                param_labels = np.concatenate((param_labels,np.copy(truth)),axis=0)

            num_corner_plot = max(20,num_recovery_LC)
            num_corner_plot = min(num_corner_plot,len(truth))
            if i == 0:
                for num in range(num_corner_plot):
                    corner_plot(sample[num],truth[num],num)


            #save the mean prediction for all
            mean_pred = np.array(mean_list)
            if i == 0:
                predictions = np.copy(mean_pred)
            else:
                predictions = np.concatenate((predictions,np.copy(mean_pred)),axis=0)
            
            # Sample has shape [N_test, samples, n_params]

            # Get the 1sigma left and right bounds and median
            if i == 0:
                eval_lower_bound = np.quantile(sample,0.5-0.68/2.,axis=1)
                eval_upper_bound = np.quantile(sample,0.5+0.68/2.,axis=1)
                eval_median = np.quantile(sample,0.5,axis=1)
            else:
                eval_lower_bound = np.concatenate((eval_lower_bound,np.quantile(sample,0.5-0.68/2.,axis=1)),axis=0)
                eval_upper_bound = np.concatenate((eval_upper_bound,np.quantile(sample,0.5+0.68/2.,axis=1)),axis=0)
                eval_median = np.concatenate((eval_median,np.quantile(sample,0.5,axis=1)),axis=0)
            

            print("evaluating the coverage probabilities")
            for k in range(len(truth)):
                for j,par in enumerate(parameters_keys): 
                    true_mean = truth[k,j]
                    sample_par = sample[k,:,j]
                    for quantile in quantiles:
                        if np.quantile(sample_par,0.5-quantile/2.) <= true_mean <= np.quantile(sample_par,0.5+quantile/2.):
                            coverage_prob_dict[quantile][par] += 1

                    for quantile in all_quantiles:
                        if np.quantile(sample_par,0.5-quantile/2.) <= true_mean <= np.quantile(sample_par,0.5+quantile/2.):
                            all_coverage_prob_dict[quantile][par] += 1

            total_val += len(truth)

            true_LC = true_LC.detach().cpu().numpy()
            #metric for evaluating the performance of the LC reconstruction
            if evaluate_LC_performance and use_SDE:

                if sample_from_latent == True or uncertainty_output == False:

                    num_samples = 60 #sample 100 reconstructed LCs from the posteriors
                    print('sampling to evaluate LC reconstruction')
                    logging.warning('sampling to evaluate LC reconstruction')

                    for i in range(num_samples):
                        #sample from the posterior
                        sample = latent_sde.sample_posterior(ys_batch.to(device),
                                                            ts_batch.to(device),
                                                            redshift=redshift,
                                                            adjoint=adjoint,
                                                            method=method,
                                                            )
                        if uncertainty_output:
                            recovery_mean = sample[:,:,:num_bands].detach().cpu().numpy() 
                            recovery_std = np.sqrt(np.exp(sample[:,:,num_bands:].detach().cpu().numpy())) # convert log variance to std
                            sample = np.random.normal(recovery_mean, recovery_std) # [T_vis, n_bandpasses]
                        else:
                            sample = sample.detach().cpu().numpy() # [T_vis, n_bandpasses]
                        sample = np.expand_dims(sample,axis=3) # [T_vis, n_bandpasses, 1]

                        # combined_samples has shape [T_vis, n_bandpasses, n_sample]
                        if i == 0: 
                            combined_samples = np.copy(sample) 
                        else:
                            combined_samples = np.concatenate((combined_samples,sample),axis=3) 
                    recovery_mean = np.mean(combined_samples,axis=3)
                    recovery_std = np.std(combined_samples,axis=3)
                    del combined_samples
                    del sample
                else:
                    #sample from the posterior
                    sample = latent_sde.sample_posterior(ys_batch.to(device),
                                                        ts_batch.to(device),
                                                        redshift=redshift,
                                                        adjoint=adjoint,
                                                        method=method,
                                                        )
                    recovery_mean = sample[:,:,:num_bands].detach().cpu().numpy() 
                    recovery_std = np.sqrt(np.exp(sample[:,:,num_bands:].detach().cpu().numpy())) # convert log variance to std
                    del sample
                recovery_mean += mag_mean
                #clip the std to avoid outliers when the std is very small. Minimum std is 0.01. The same is done to for the GPR.
                recovery_std = np.clip(recovery_std,min_std_recovery,100.0) 

                #sum across time, batch, and bandpass for the average mse for each data point in our reconstruction
                val_mse_recovery_epoch.append((np.abs(recovery_mean-true_LC)**2).mean(axis=(0,2)))
                val_mae_recovery_epoch.append((np.abs(recovery_mean-true_LC)).mean(axis=(0,2)))
                # Get the mean difference between the true and recovered light curve for each LC in the batch
                for j in range(recovery_mean.shape[1]):
                    LC_reconstruction_mean_diff.append(np.abs(recovery_mean[:,j]-true_LC[:,j]).mean())
                    
                #log prob of reconstruction
                #xs_dist = Normal(loc=torch.from_numpy(recovery_mean),scale=torch.from_numpy(recovery_std))
                #log_P = xs_dist.log_prob(torch.from_numpy(true_LC)).mean(dim=0).mean()
                #val_log_P_epoch += log_P.mean().cpu().numpy()
                NGLL = 0.5*(np.log(2*np.pi)+np.log(recovery_std**2)+((recovery_mean-true_LC)**2)/recovery_std**2)
                val_log_P_epoch.append(NGLL.mean(axis=(0,2)))
                
                #evaluate uncertainty of reconstruction
                num_sigma_recovery = np.abs((recovery_mean-true_LC)/recovery_std)

                num_sigma_1_recovery_epoch += (num_sigma_recovery <= 1.0).astype(float).mean()
                num_sigma_2_recovery_epoch += (num_sigma_recovery <= 2.0).astype(float).mean()
                num_sigma_3_recovery_epoch += (num_sigma_recovery <= 3.0).astype(float).mean()

                for quantile in all_quantiles:
                    all_coverage_prob_dict[quantile]['recovery'] += (num_sigma_recovery <= stats.norm.ppf(1-(1-quantile)/2)).astype(float).mean()

            if evaluate_LC_performance_GPR:
                
                for j in range(ys_batch.shape[1]):

                    #GPR
                    mean, std, tau_pred, SFinf_pred = gaussian_process_regression(torch.clone(ys_batch[:,j,:num_bands]),
                                                                                torch.clone(ys_batch[:,j,num_bands:]),
                                                                                device=device,
                                                                                min_std=min_std_recovery)
                                    
                    # Save the predicted tau and SFinf of the GPR
                    tau_pred_list.append(tau_pred)
                    SFinf_pred_list.append(SFinf_pred)
                    
                    mean += mag_mean
                    num_sigma_recovery = np.abs((mean-true_LC[:,j,:])/std)
                    num_sigma_1_recovery_GPR_epoch += (num_sigma_recovery <= 1.0).astype(float).mean()/ys_batch.shape[1]
                    num_sigma_2_recovery_GPR_epoch += (num_sigma_recovery <= 2.0).astype(float).mean()/ys_batch.shape[1]
                    num_sigma_3_recovery_GPR_epoch += (num_sigma_recovery <= 3.0).astype(float).mean()/ys_batch.shape[1]

                    for quantile in all_quantiles:
                        all_coverage_prob_dict[quantile]['recovery_GPR'] += (num_sigma_recovery <= stats.norm.ppf(1-(1-quantile)/2)).astype(float).mean()/ys_batch.shape[1]

                    #sum across time, batch, and bandpass for the average mse for each data point in our reconstruction
                    val_mse_recovery_GPR_epoch.append(np.abs((mean-true_LC[:,j,:])**2).mean(axis=(0,1)))
                    val_mae_recovery_GPR_epoch.append(np.abs(mean-true_LC[:,j,:]).mean(axis=(0,1)))

                    #log prob of reconstruction
                    #xs_dist = Normal(loc=torch.from_numpy(mean),scale=torch.from_numpy(std))
                    #log_P = xs_dist.log_prob(torch.from_numpy(true_LC[:,j,:])).mean(dim=0)
                    #val_log_P_GPR_epoch += log_P.mean().cpu().numpy()/ys_batch.shape[1]
                    NGLL = 0.5*(np.log(2*np.pi)+np.log(std**2)+((mean-true_LC[:,j,:])**2)/std**2)
                    # Clip incase we have some very large values for one particular LC. This really only happens with the GPR.
                    val_log_P_GPR_epoch.append(NGLL.mean(axis=(0,1)))

            pbar.update(1)
            i += 1
        pbar.close()
        
        for quantile in quantiles:
            for par in parameters_keys:
                coverage_prob_dict[quantile][par] /= total_val

        for quantile in all_quantiles:
            for par in parameters_keys:
                all_coverage_prob_dict[quantile][par] /= total_val


        print()
        print('coverage prob dict SDE')
        print(coverage_prob_dict)

        if evaluate_LC_performance and use_SDE:
            
            np.save(f"{save_path}/val_mse_recovery_epoch.npy",val_mse_recovery_epoch)
            np.save(f"{save_path}/val_mae_recovery_epoch.npy",val_mae_recovery_epoch)
            np.save(f"{save_path}/val_log_P_epoch.npy",val_log_P_epoch)

            # get mean and median of the metrics
            val_mse_recovery_epoch = np.concatenate(val_mse_recovery_epoch,axis=0)
            val_mae_recovery_epoch = np.concatenate(val_mae_recovery_epoch,axis=0)
            val_log_P_epoch = np.concatenate(val_log_P_epoch,axis=0)

            val_rmse_recovery_epoch_median = np.median(np.sqrt(val_mse_recovery_epoch))
            val_rmse_recovery_epoch_mean = np.mean(np.sqrt(val_mse_recovery_epoch))

            # divide by sqrt(N) to get the standard error of the mean
            val_rmse_recovery_epoch_median_absolute_deviation = np.median(np.abs(np.sqrt(val_mse_recovery_epoch) - val_rmse_recovery_epoch_median))/np.sqrt(val_mse_recovery_epoch.shape[0])
            val_rmse_recovery_epoch_mean_standard_deviation = np.std(np.sqrt(val_mse_recovery_epoch)-val_rmse_recovery_epoch_mean)/np.sqrt(val_mse_recovery_epoch.shape[0])


            val_mae_recovery_epoch_median = np.median(val_mae_recovery_epoch)
            val_mae_recovery_epoch_mean = np.mean(val_mae_recovery_epoch)

            # divide by sqrt(N) to get the standard error of the mean
            val_mae_recovery_epoch_median_absolute_deviation = np.median(np.abs(val_mae_recovery_epoch - val_mae_recovery_epoch_median))/np.sqrt(val_mae_recovery_epoch.shape[0])
            val_mae_recovery_epoch_mean_standard_deviation = np.std(val_mae_recovery_epoch-val_mae_recovery_epoch_mean)/np.sqrt(val_mae_recovery_epoch.shape[0])

            val_log_P_epoch_median = np.median(val_log_P_epoch)
            val_log_P_epoch_mean = np.mean(val_log_P_epoch)

            # divide by sqrt(N) to get the standard error of the mean
            val_log_P_epoch_median_absolute_deviation = np.median(np.abs(val_log_P_epoch - val_log_P_epoch_median))/np.sqrt(val_log_P_epoch.shape[0])
            val_log_P_epoch_mean_standard_deviation = np.std(val_log_P_epoch-val_log_P_epoch_mean)/np.sqrt(val_log_P_epoch.shape[0])

            #average over the number of batches
            num_sigma_1_recovery_epoch /= len(test_loader)
            num_sigma_2_recovery_epoch /= len(test_loader)
            num_sigma_3_recovery_epoch /= len(test_loader)

            for quantile in all_quantiles:
                all_coverage_prob_dict[quantile]['recovery'] /= len(test_loader)

            # SDE
            # median
            print()
            print('median LC reconstruction SDE:')
            logging.warning('median LC reconstruction SDE')
            print()
            print(f"median LC reconstruction RMSE: {val_rmse_recovery_epoch_median} +/- {val_rmse_recovery_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction RMSE: {val_rmse_recovery_epoch_median} +/- {val_rmse_recovery_epoch_median_absolute_deviation}")
            print(f"median LC reconstruction MAE: {val_mae_recovery_epoch_median} +/- {val_mae_recovery_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction MAE: {val_mae_recovery_epoch_median} +/- {val_mae_recovery_epoch_median_absolute_deviation}")
            print(f"median LC reconstruction: -log(P): {val_log_P_epoch_median} +/- {val_log_P_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction: -log(P): {val_log_P_epoch_median} +/- {val_log_P_epoch_median_absolute_deviation}")

            # mean
            print()
            logging.warning('mean LC reconstruction SDE:')

            print(f"mean LC reconstruction root MSE: {val_rmse_recovery_epoch_mean} +/- {val_rmse_recovery_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction root MSE: {val_rmse_recovery_epoch_mean} +/- {val_rmse_recovery_epoch_mean_standard_deviation}")
            print(f"mean LC reconstruction MAE: {val_mae_recovery_epoch_mean} +/- {val_mae_recovery_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction MAE: {val_mae_recovery_epoch_mean} +/- {val_mae_recovery_epoch_mean_standard_deviation}")
            print(f"mean LC reconstruction: -log(P): {val_log_P_epoch_mean} +/- {val_log_P_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction: -log(P): {val_log_P_epoch_mean} +/- {val_log_P_epoch_mean_standard_deviation}")
            print()

            #Plots a histogram of the LC reconstruction coverage probability
            plot_LC_reconstruction_sigma_eval(num_sigma_1_recovery_epoch,
                                            num_sigma_2_recovery_epoch,
                                            num_sigma_3_recovery_epoch,
                                            name="LC_reconstruction_coverage_prob_hist_SDE")

            print(f"1 sigma coverage: {round(100*num_sigma_1_recovery_epoch,1)}% compared to 68.3%")
            print(f"2 sigma coverage: {round(100*num_sigma_2_recovery_epoch,1)}% compared to 95.5%")
            print(f"3 sigma coverage: {round(100*num_sigma_3_recovery_epoch,1)}% compared to 99.7%")

            logging.warning(
                f"1 sigma coverage: {round(100*num_sigma_1_recovery_epoch,1)}% compared to 68.3%,  "
                f"2 sigma coverage: {round(100*num_sigma_2_recovery_epoch,1)}% compared to 95.5%,  "
                f"3 sigma coverage: {round(100*num_sigma_3_recovery_epoch,1)}% compared to 99.7%"
            )

            print()
            print()

        if evaluate_LC_performance_GPR:
            
            # get mean and median of the metrics

            val_mse_recovery_GPR_epoch = np.array(val_mse_recovery_GPR_epoch)
            val_mae_recovery_GPR_epoch = np.array(val_mae_recovery_GPR_epoch)
            val_log_P_GPR_epoch = np.array(val_log_P_GPR_epoch)

            np.save(f"{save_path}/val_mse_recovery_GPR_epoch.npy", val_mse_recovery_GPR_epoch)
            np.save(f"{save_path}/val_mae_recovery_GPR_epoch.npy", val_mae_recovery_GPR_epoch)
            np.save(f"{save_path}/val_log_P_GPR_epoch.npy", val_log_P_GPR_epoch)
            
            val_rmse_recovery_GPR_epoch_median = np.median(np.sqrt(val_mse_recovery_GPR_epoch))
            val_rmse_recovery_GPR_epoch_mean = np.mean(np.sqrt(val_mse_recovery_GPR_epoch))

            val_rmse_recovery_GPR_epoch_median_absolute_deviation = np.median(np.abs(np.sqrt(val_mse_recovery_GPR_epoch) - val_rmse_recovery_GPR_epoch_median))/np.sqrt(len(val_mse_recovery_GPR_epoch))
            val_rmse_recovery_GPR_epoch_mean_standard_deviation = np.std(np.sqrt(val_mse_recovery_GPR_epoch) - val_rmse_recovery_GPR_epoch_mean)/np.sqrt(len(val_mse_recovery_GPR_epoch))
            
            val_mae_recovery_GPR_epoch_median = np.median(val_mae_recovery_GPR_epoch)
            val_mae_recovery_GPR_epoch_mean = np.mean(val_mae_recovery_GPR_epoch)

            val_mae_recovery_GPR_epoch_median_absolute_deviation = np.median(np.abs(val_mae_recovery_GPR_epoch - val_mae_recovery_GPR_epoch_median))
            val_mae_recovery_GPR_epoch_mean_standard_deviation = np.std(val_mae_recovery_GPR_epoch - val_mae_recovery_GPR_epoch_mean)/np.sqrt(len(val_mae_recovery_GPR_epoch))
            
            val_log_P_GPR_epoch_median = np.median(val_log_P_GPR_epoch)
            val_log_P_GPR_epoch_mean = np.mean(val_log_P_GPR_epoch)

            val_log_P_GPR_epoch_median_absolute_deviation = np.median(np.abs(val_log_P_GPR_epoch - val_log_P_GPR_epoch_median))
            val_log_P_GPR_epoch_mean_standard_deviation = np.std(val_log_P_GPR_epoch - val_log_P_GPR_epoch_mean)/np.sqrt(len(val_log_P_GPR_epoch))

            #average over the number of batches
            num_sigma_1_recovery_GPR_epoch /= len(test_loader)
            num_sigma_2_recovery_GPR_epoch /= len(test_loader)
            num_sigma_3_recovery_GPR_epoch /= len(test_loader)

            for quantile in all_quantiles:
                all_coverage_prob_dict[quantile]['recovery_GPR'] /= len(test_loader)

            # GPR
            # median
            print('median LC reconstruction GPR')
            logging.warning('median LC reconstruction GPR')
            print()

            print(f"median LC reconstruction RMSE: {val_rmse_recovery_GPR_epoch_median} +/- {val_rmse_recovery_GPR_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction RMSE: {val_rmse_recovery_GPR_epoch_median} +/- {val_rmse_recovery_GPR_epoch_median_absolute_deviation}")
            print(f"median LC reconstruction MAE: {val_mae_recovery_GPR_epoch_median} +/- {val_mae_recovery_GPR_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction MAE: {val_mae_recovery_GPR_epoch_median} +/- {val_mae_recovery_GPR_epoch_median_absolute_deviation}")
            print(f"median LC reconstruction log_P: {val_log_P_GPR_epoch_median} +/- {val_log_P_GPR_epoch_median_absolute_deviation}")
            logging.warning(f"median LC reconstruction log_P: {val_log_P_GPR_epoch_median} +/- {val_log_P_GPR_epoch_median_absolute_deviation}")

            print()
            print()

            # mean
            print('mean LC reconstruction GPR')
            logging.warning('mean LC reconstruction GPR')
            print()

            print(f"mean LC reconstruction RMSE: {val_rmse_recovery_GPR_epoch_mean} +/- {val_rmse_recovery_GPR_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction RMSE: {val_rmse_recovery_GPR_epoch_mean} +/- {val_rmse_recovery_GPR_epoch_mean_standard_deviation}")
            print(f"mean LC reconstruction MAE: {val_mae_recovery_GPR_epoch_mean} +/- {val_mae_recovery_GPR_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction MAE: {val_mae_recovery_GPR_epoch_mean} +/- {val_mae_recovery_GPR_epoch_mean_standard_deviation}")
            print(f"mean LC reconstruction log_P: {val_log_P_GPR_epoch_mean} +/- {val_log_P_GPR_epoch_mean_standard_deviation}")
            logging.warning(f"mean LC reconstruction log_P: {val_log_P_GPR_epoch_mean} +/- {val_log_P_GPR_epoch_mean_standard_deviation}")

            print()
            print()


            #Plots a histogram of the LC reconstruction coverage probability
            plot_LC_reconstruction_sigma_eval(num_sigma_1_recovery_GPR_epoch, 
                                            num_sigma_2_recovery_GPR_epoch,
                                            num_sigma_3_recovery_GPR_epoch,
                                            name="LC_reconstruction_coverage_prob_hist_GPR")

            print(f"1 sigma coverage: {round(100*num_sigma_1_recovery_GPR_epoch,1)}% compared to 68.3%")
            print(f"2 sigma coverage: {round(100*num_sigma_2_recovery_GPR_epoch,1)}% compared to 95.5%")
            print(f"3 sigma coverage: {round(100*num_sigma_3_recovery_GPR_epoch,1)}% compared to 99.7%")

            logging.warning(
                f"1 sigma coverage: {round(100*num_sigma_1_recovery_GPR_epoch,1)}% compared to 68.3%,  "
                f"2 sigma coverage: {round(100*num_sigma_2_recovery_GPR_epoch,1)}% compared to 95.5%,  "
                f"3 sigma coverage: {round(100*num_sigma_3_recovery_GPR_epoch,1)}% compared to 99.7%"
            )
            print()
            print()

        print()


        mean_values_dict,test_label_dict = dict(),dict()
        for j,par in enumerate(parameters_keys):
            mean_values_dict[par] = predictions[:,j]
            test_label_dict[par] = param_labels[:,j]  

            MSE = np.mean((predictions[:,j]-param_labels[:,j])**2)
            MAE = np.mean(np.abs(predictions[:,j]-param_labels[:,j]))
            percent_error = 100*np.sqrt(MSE)
            print(f"{par} MSE: {round(MSE,3)}")
            print(f"{par} MAE: {round(MAE,3)}")
            print(f"{par} Percent error: {round(percent_error,3)} %")

            logging.warning(f"{par} MSE: {round(MSE,3)}")
            logging.warning(f"{par} MAE: {round(MAE,3)}")
            logging.warning(f"{par} Percent error: {round(percent_error,3)} %")            
            print()

        plot_sigma_eval(coverage_prob_dict,quantiles)
        plot_sigma_eval_all(all_coverage_prob_dict,all_quantiles)
        plot_confusion_matrices(mean_values_dict,test_label_dict,num_bins=20)
        plot_confusion_matrices(mean_values_dict,test_label_dict,num_bins=15)
        try:
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=15,log_mass_limit=8.0)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=15,log_mass_limit=8.5)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=15,log_mass_limit=9.0)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=15,log_mass_limit=9.5) 

            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=20,log_mass_limit=8.0)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=20,log_mass_limit=8.5)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=20,log_mass_limit=9.0)
            plot_confusion_matrices_large_mass(mean_values_dict,test_label_dict,num_bins=20,log_mass_limit=9.5) 
        except:
            pass
        plot_residual_histograms(mean_values_dict,test_label_dict)


        # Plot scatter plots of the predictions vs the true values (like confusion matrices but with scatter plots and uncertainty bars)

        log_mass_limit_list = [None, 8.0, 8.5, 9.0, 9.5]
        max_num_samples_list = [50, 100, 200, 300, 400, 500]
        for max_num_samples in max_num_samples_list:
            for log_mass_limit in log_mass_limit_list:
                try:
                    plot_scatter_plots(param_labels,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=log_mass_limit,max_num_samples=max_num_samples)
                except:
                    pass
        if evaluate_LC_performance_GPR:
            tau_pred_list = np.log10(np.array(tau_pred_list))
            SFinf_pred_list = np.array(SFinf_pred_list)

            # Clip the predictions to the min and max values of the training set for plotting purposes
            tau_pred = np.clip(tau_pred,min_max_dict['log_tau'][0],min_max_dict['log_tau'][1])
            SFinf_pred = np.clip(SFinf_pred,min_max_dict['SFinf'][0],min_max_dict['SFinf'][1])

            plot_scatter_DRW_params(param_labels,eval_median,eval_lower_bound,eval_upper_bound, tau_pred_list, SFinf_pred_list, max_num_samples=10)
            plot_scatter_DRW_params(param_labels,eval_median,eval_lower_bound,eval_upper_bound, tau_pred_list, SFinf_pred_list, max_num_samples=15)
            plot_scatter_DRW_params(param_labels,eval_median,eval_lower_bound,eval_upper_bound, tau_pred_list, SFinf_pred_list, max_num_samples=100)


        if evaluate_LC_performance:
            LC_reconstruction_mean_diff = np.array(LC_reconstruction_mean_diff)
        else:
            LC_reconstruction_mean_diff = np.array([])

        try:
            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=False,zoom_range=1.0)
            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=False,zoom_range=2.0)
            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=True)

            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=False,zoom_range=1.0,plot_truth=True)
            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=False,zoom_range=2.0,plot_truth=True)
            plot_residual_corner(np.copy(predictions),np.copy(param_labels),np.copy(LC_reconstruction_mean_diff),zoom=True,plot_truth=True)
        except:
            logging.warning('Corner plot failed because you have too few samples in the validation set (or there was an error).')

def build_training_set(i):
    file = file_names_train[i]
    sample = build_data_set(file,mag_mean=mag_mean,cadence=cadence,augment=True,save_true_LC=False)
    save_name = f'{save_path}/train_data/training_{i}.npy'
    np.save(save_name,sample)
    
def build_validation_set(i):
    file = file_names_val[i]
    sample = build_data_set(file,mag_mean=mag_mean,cadence=cadence,augment=False,save_true_LC=True)
    save_name = f'{save_path}/val_data/val_{i}.npy'
    np.save(save_name,sample)

def build_test_set(i):
    file = file_names_test[i]
    sample = build_data_set(file,mag_mean=mag_mean,cadence=cadence,augment=True,save_true_LC=True)
    save_name = f'{save_path}/test_data/test_{i}.npy'
    np.save(save_name,sample)

if __name__ == '__main__':
    # Use this to build the data set before training. Otherwise the training set is build on the fly each epoch.
    create_data_before_training = False

    assert(type(create_data_before_training) == bool), "create_data_before_training must be a boolean"
    
    if create_data_before_training:
        start_time = time.time()

        # Build the data set. We use multiprocessing to speed up the process.

        # Build the training set in directory 'train_data'
        training_file_path = f'{save_path}/train_data'
        if os.path.exists(training_file_path):
            shutil.rmtree(training_file_path, ignore_errors=False, onerror=None)
        os.makedirs(training_file_path, exist_ok=True)
        
        # Build the validation set in directory 'val_data'
        val_file_path = f'{save_path}/val_data'
        if os.path.exists(val_file_path):
            shutil.rmtree(val_file_path, ignore_errors=False, onerror=None)
        os.makedirs(val_file_path, exist_ok=True)

        # Build the test set in directory 'test_data'
        test_file_path = f'{save_path}/test_data'
        if os.path.exists(test_file_path):
            shutil.rmtree(test_file_path, ignore_errors=False, onerror=None)
        os.makedirs(test_file_path, exist_ok=True)

        #use multiprocessing to speed up the data generation using the build_data_set function and Pool
        num_cpus = os.cpu_count()
        print(f"Number of CPU cores: {num_cpus}")

        #create training data
        print("Creating training data")
        with Pool(num_cpus) as p:
            sample = p.map(build_training_set,range(len(file_names_train)))
        
        # create validation data
        print("Creating validation data")
        with Pool(num_cpus) as p:
            sample = p.map(build_validation_set,range(len(file_names_val)))

        #create test data
        print("Creating test data")
        with Pool(num_cpus) as p:
            sample = p.map(build_test_set,range(len(file_names_test)))

        # print the time it took to build the data set in hours
        print(f"Time to build data set: {round((time.time()-start_time)/3600,2)} hours")

    # train and evaluate the model
    start_time = time.time()
    main(create_data_before_training) 
    print()
    # print the time it took to train and evaluate the model in minues
    print(f"Time to train and evaluate model: {round((time.time()-start_time)/60,2)} minutes")
    # print the time it took to train and evaluate the model in hours
    print(f"Time to train and evaluate model: {round((time.time()-start_time)/(60*60),2)} hours")
    # print the time it took to train and evaluate the model in days
    print(f"Time to train and evaluate model: {round((time.time()-start_time)/(60*60*24),2)} days")


