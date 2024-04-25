# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:50:46 2024

@author: Luis Miguel Moreno Saavedra
"""

import pandas as pd
import numpy as np
import random
import os

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)

from Evolutivo_MCCE import EvolutiveClass


# Set the scenario to run the experiment
Num_scenario = 2
Scenario = f"Scenario {Num_scenario}"
Folder = os.path.join(parent_dir, "data", Scenario)
df_scenario = pd.read_csv(os.path.join(Folder,"Points.csv"), encoding="utf-8", sep=";")
df_devices = pd.read_csv(os.path.join(Folder,"Devices.csv"), encoding="utf-8", sep=";")

# Set the seeds
Seeds = np.arange(2023, 2053, 1)

# Set the algorithm hyper-parameters
Num_Individuos = 100
Num_Generaciones = 1000
Prob_Padres = 0.5
Prob_Mutacion = 0.7
Prob_Cruce = 0.5
verbose = True

for seed in Seeds:
    random.seed(int(seed))
    print(f"Start in seed {seed}, Scenario {Num_scenario}")
    Ev1 = EvolutiveClass(df_escenario = df_scenario, 
                         df_dispositivos = df_devices,
                         Num_Individuos = Num_Individuos, 
                         Num_Generaciones = Num_Generaciones, 
                         Tam_Individuos = np.where(df_scenario["Punto de Vigilancia"] == 1)[0].shape[0], 
                         Prob_Padres = Prob_Padres, 
                         Prob_Mutacion = Prob_Mutacion, 
                         Prob_Cruce = Prob_Cruce,
                         seed = seed,
                         verbose = verbose)
    
    Ev1.InicioAlgoritmo()

    # Save the solutions
    np.save(os.path.join(Folder,f"Seed_{seed}_Best_Individual_Dev.npy"),
            Ev1.Mejor_Individuo_Disp)
    np.save(os.path.join(Folder,f"Seed_{seed}_Best_Individual_Ang.npy"),
            Ev1.Mejor_Individuo_Ang)
    np.save(os.path.join(Folder,f"Seed_{seed}_Fitness.npy"),
            Ev1.Fitness_Grafica) 
    