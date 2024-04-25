# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:48:39 2024

@author: Luis Miguel Moreno Saavedra
"""

import pandas as pd
import numpy as np
import random
import os

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)

from GRASP import GRASPClass

# Set the scenario to run the experiment
Num_scenario = 2
Scenario = f"Scenario {Num_scenario}"
Folder = os.path.join(parent_dir, "data", Scenario)
df_scenario = pd.read_csv(os.path.join(Folder,"Points.csv"), encoding="utf-8", sep=";")
df_devices = pd.read_csv(os.path.join(Folder,"Devices.csv"), encoding="utf-8", sep=";")

# Set the seeds
Seeds = np.arange(2023, 2053, 1)

# Set the algorithm hyper-parameters
iter_greedy=500
iter_local_search=200
verbose = True

for seed in Seeds:
    random.seed(int(seed))
    print(f"Start in seed {seed}, Scenario {Num_scenario}")
    GRASP = GRASPClass(df_escenario = df_scenario,
                       df_dispositivos = df_devices,
                       iter_greedy=iter_greedy,
                       iter_local_search=iter_local_search,
                       seed=seed, 
                       verbose=verbose)
    
    GRASP.start_algorithm()

    # Save the solutions
    np.save(os.path.join(Folder,f"Seed_{seed}_Best_Individual_Dev.npy"),
            GRASP.Mejor_Individuo_Disp)
    np.save(os.path.join(Folder,f"Seed_{seed}_Best_Individual_Ang.npy"),
            GRASP.Mejor_Individuo_Ang)
    np.save(os.path.join(Folder,f"Seed_{seed}_Fitness.npy"),
            GRASP.cost)