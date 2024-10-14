#!/usr/bin/env python
# coding: utf-8

# In[3]:


import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[107]:


#Create jackknife bins
def jackknife_bin_creation(values):
    n = len(values)
    jackknife_bins = np.empty((n, n-1))
    jackknife_averages = np.empty(n)
    
    for i in range(n):
        jackknife_bins[i] = np.delete(values, i) #remove a value for each bin
        jackknife_averages[i] = np.mean(jackknife_bins[i]) #calculate average for remaining values
        
    return jackknife_bins, jackknife_averages


# In[111]:


#Extract values and create bins
def extract_values(zip_path, time_slots):    
    extracted_values = [[] for _ in range(time_slots)]  # For storing values from the first 64 rows
    jackknife_bins = [[] for _ in range(time_slots)]    # For storing jackknife bins
    jackknife_averages = [[] for _ in range(time_slots)]
    num_files = 0  # Count the number of files processed  
    
    with zipfile.ZipFile(zip_path, 'r') as z:        
        data_files = [f for f in z.namelist() if f.endswith('.dat') and '__MACOSX' not in f and '1196' not in f] #go into pion data folder, and remove file 1196 since it has NANs
        
        for file_path in data_files: #for each measure (folder) in the overall data file. file_id is 0,1,...,num_files-1
            num_files += 1
            #print(f"Processing file: {file_path}")
            try:
                with z.open(file_path) as file:
                    for i in range(time_slots):
                        line = file.readline().decode('utf-8').strip()
                        if not line:
                            print(f"File {file_path} has less than {time_slots} lines")
                            break;
                        parts = line.split()
                        if len(parts) >= 5:
                            extracted_value = parts[4]
                            try:
                                extracted_values[i].append(float(extracted_value))
                            except ValueError:
                                print(f"Error converting value to float in file {file_path}, line {i+1}: {extracted_value}")
                        else:
                            print(f"Unexpected format in file {file_path}, line {i+1}: {line}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    #now create jackknife bins
    for i in range(time_slots):
        if len(extracted_values[i]) > 1:
            jackknife_bins[i], jackknife_averages[i] = jackknife_bin_creation(np.array(extracted_values[i]))
        else:
            jackknife_bins[i], jackknife_averages[i] = np.array([])

    return extracted_values, jackknife_bins, jackknife_averages, num_files


# In[112]:


root_directory = 'pion_P0 1.zip'
time_slots = 64

extracted_values, jackknife_bins, jackknife_averages, num_files = extract_values(root_directory, time_slots)


# In[122]:


#Checking that everything above is good

#print("Extracted Values for Each Time Slot:")
#for i, values in enumerate(extracted_values):
    #print(f"Time Slot {i}: {values}")

#print("\nJackknife Averages for Each Time Slot:")
#for i, bins in enumerate(jackknife_bins):
    #print(f"Time Slot {i}: {bins}")
    
#len(extracted_values)
#len(jackknife_bins)
#len(jackknife_averages)
#len(jackknife_bins[0])
#len(jackknife_averages[0])
#num_files
#jackknife_averages
#jackknife_bins


# In[133]:


#Folding
def fold_values(jackknife_averages):
    fold_range = 33
    folded_values = np.zeros((fold_range, len(jackknife_averages[0])))
    
    for t in range(fold_range):
        if t==0 or t==32:
            folded_values[t] = jackknife_averages[t]
        else:
            folded_values[t] = (jackknife_averages[t] + jackknife_averages[63-t])/2
                
    return folded_values


# In[136]:


folded_values = fold_values(jackknife_averages)
#len(folded_values)
folded_values


# In[139]:


#Average 1997 values for each time slot to get one time slot average each
def calculate_overall_averages(folded_values):
    averages = np.zeros(len(folded_values))

    for t in range(len(folded_values)):
        averages[t] = np.mean(folded_values[t])
    
    return averages


# In[154]:


folded_averages = calculate_overall_averages(folded_values)

#Plot averages
times = np.arange(0, 33)

plt.errorbar(times, folded_averages, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Average')
plt.show()


# In[155]:


unfolded_averages = calculate_overall_averages(jackknife_bins)

#Plot unfolded averages
times = np.arange(0, 64)

plt.errorbar(times, unfolded_averages, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Average')
plt.show()


# In[146]:


#Calculate the log ratios for every t and t+1
def calculate_logs(folded_values):
    i_ratios = np.empty(len(folded_values[0]))
    ratios = np.empty(len(folded_values)-1) #no ratio for the last t since there's no t+1
    
    for t in range(len(folded_values)-1): #for every time slice except the last one
        for i in range(len(folded_values[t])): #for every bin in t
            this_value = folded_values[t][i]
            other_value = folded_values[t+1][i]
            i_ratios[i] = np.log(this_value / other_value)
        ratios[t] = np.mean(i_ratios)
    
    return ratios


# In[150]:


logs = calculate_logs(folded_values)
logs
#len(logs)


# In[178]:


#Plot logs
times = np.arange(0, 32)

plt.errorbar(times, logs, fmt='o', capsize=5)
plt.axhline(y=0.1253,color='black') #test to see if it plateaus at the correct spot
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Energy Plot (without errors)')
plt.show()


# In[179]:


def calculate_errors(bins):
    errors = np.empty(len(bins)-1)
    
    n = len(bins[0])
    
    for t in range(len(bins)-1):
        std_dev = np.std(bins[t])
        errors[t] = std_dev * np.sqrt(n-1) #error is std_dev * sqrt(n-1)
    
    return errors


# In[180]:


errors = calculate_errors(folded_values)
errors
#len(errors)


# In[181]:


#Plot errors and averages (log ratios)
times = np.arange(0, 32)

plt.errorbar(times, logs, yerr=errors, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Log')
plt.title('Logs with errors')
plt.show()


# **All stuff below is old, not working code**

# In[4]:


#Read the 5th column of the first 64 rows from a data file
def process_data_file(file):
    df = pd.read_csv(file, delim_whitespace=True)
    
    all_data = df.iloc[:64, 4].values #select first 64 rows and 5th column
    
    return all_data


# In[5]:


def fold(all_data):
    m = len(all_data) #equals 64
    n = m//2 #equals 32

    folded_data = np.empty(n+1) #length of 33 (index 0 to 32)
    
    folded_data[0] = all_data[0] #keep t=0 the same
    
    for i in range(1, n):
        #print(f"Folding: i={i}, m-i={m-i}, all_data[i]={all_data[i]}, all_data[m-i]={all_data[m-i]}")
        folded_data[i] = (all_data[i]+all_data[m-i]) / 2
        
    folded_data[n] = all_data[n] #keep t=32 the same
    
    return folded_data


# In[6]:


#Create jackknife bins
def jackknife_bin_creation(data):
    n = len(data)
    jackknife_bins = np.empty((n, n-1))
    jackknife_averages = np.empty(n)
    
    for i in range(n):
        jackknife_bins[i] = np.delete(data, i) #remove a value for each bin
        #print(f"Jackknife bin {i}: {jackknife_bins[i]}") #see that jackknife bins are being created correctly
            
        jackknife_averages[i] = np.mean(jackknife_bins[i]) #calculate average for remaining values (excluding the 1)
        
    return jackknife_averages


# In[144]:


#Main processing function
def process_zipfile(zip_path, num_files):    
    all_bins = np.zeros((33, num_files)) #we want 33 sets of bins
            
    with zipfile.ZipFile(zip_path, 'r') as z:        
        data_files = [f for f in z.namelist() if 'pion_P0' in f and f.endswith('.dat') and '__MACOSX' not in f and '1196' not in f] #go into pion data folder, and remove file 1196 since it has NANs
        
        for file_path in data_files: #for each measure (folder) in the overall data file. file_id is 0,1,...,num_files-1
            num_files += 1
            print(f"Processing file: {file_path}")
            try:
                with z.open(file_path) as file:
                    for i in range(time_slots):
                        line = file.readline().decode('utf-8').strip()
                        if not line:
                            print(f"File {file_path} has less than {time_slots} lines")
                            break;
                        parts = line.split()
                        if len(parts) >= 5:
                            extracted_value = parts[4]
                            try:
                                extracted_values[i].append(float(extracted_value))
                            except ValueError:
                                print(f"Error converting value to float in file {file_path}, line {i+1}: {extracted_value}")
                        else:
                            print(f"Unexpected format in file {file_path}, line {i+1}: {line}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    #now create jackknife bins
    for i in range(time_slots):
        if len(extracted_values[i]) > 1:
            jackknife_bins[i] = jackknife_bin_creation(np.array(extracted_values[i]))
        else:
            jackknife_bins[i] = np.array([])

    return extracted_values, jackknife_bins, num_files


# In[145]:


zip_file_path = 'pion_P0 1.zip' #momentum = 0

num_files = 1198
jackknife_bins = process_zipfile(zip_file_path, int(num_files))


# In[58]:


#WITHOUT FOLDING
jackknife_bins_no_fold = process_zipfile(zip_file_path, int(num_files))


# In[69]:


jackknife_bins_no_fold[61][1]


# In[70]:


jackknife_bins_no_fold[62][1]


# In[71]:


jackknife_bins_no_fold[63][1] #why does it get so big???


# In[82]:


len(jackknife_bins_no_fold)


# In[92]:


jackknife_bins[0][1]


# In[93]:


jackknife_bins[1][1] #<--- we see an issue here. It should not be a value larger than at t=2


# In[94]:


jackknife_bins[2][1]


# In[83]:


len(jackknife_bins)


# In[87]:


def get_averages(jackkknife_bins, n):
    averages1 = np.empty(n)

    this = np.empty(len(jackknife_bins[0]))

    for t in range(n):
        for i in range(len(jackknife_bins[t])):
            this[i] = jackknife_bins[t][i]
        averages1[t] = np.mean(this)
        
    return averages1


# In[88]:


averages1 = get_averages(jackknife_bins, len(jackknife_bins))


# In[89]:


averages1


# In[226]:


len(averages1)


# In[228]:


#Plot bin averages
times = np.arange(0, 33)

plt.errorbar(times, averages1, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.show()


# In[189]:


#Plot bin averages (before folding)
times = np.arange(0, 64)###

plt.errorbar(times, averages1, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.show()


# try doing jackknife bins before folding

# In[229]:


#Calculate the log ratios for every t and t+1
def calculate_ratios(jackknife_bins):
    i_ratios = np.empty(len(jackknife_bins[0])) #size 1199
    ratios = np.empty(len(jackknife_bins)-1) #size 32. no ratio for the last t since there's no t+1
    
    for t in range(len(jackknife_bins)-1): #for every time slice except the last one
        for i in range(len(jackknife_bins[t])): #for every bin in t
            this_value = jackknife_bins[t][i]
            other_value = jackknife_bins[t+1][i]
            i_ratios[i] = np.log(this_value / other_value)
        ratios[t] = np.mean(i_ratios)
    
    return ratios


# In[230]:


ratios = calculate_ratios(jackknife_bins) #these are the effective masses to plot


# In[231]:


print(ratios)


# In[232]:


#Calculate error for every t
def calculate_error(jackknife_bins):
    errors = np.empty(len(jackknife_bins)-1) #32 errors

    for t in range(len(jackknife_bins)-1):
        part_two = np.sqrt((len(jackknife_bins[t]) - 1) / len(jackknife_bins[t]))
    
        differences = np.empty(len(jackknife_bins[t]))
        for i in range(len(jackknife_bins[t])):
            differences[i] = (jackknife_bins[t][i] - ratios[t])**2
        part_one = np.sqrt(np.sum(differences))
    
        errors[t] = part_one * part_two
    return errors


# take std * sqrt(n-1) instead!!

# In[245]:


errors = calculate_error(jackknife_bins)
errors


# In[244]:


#Plot errors and averages (log ratios)
times = np.arange(0, len(jackknife_bins)-1)

plt.errorbar(times, ratios, fmt='o', capsize=5)
plt.xlabel('Time slice')
plt.ylabel('Effective mass')
plt.title('Effective masses with errors')
plt.show()

