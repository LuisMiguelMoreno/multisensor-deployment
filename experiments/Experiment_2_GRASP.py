# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:48:39 2024

@author: LuisMi-ISDEFE
"""

import pandas as pd
import numpy as np
import random
from GRASP import GRASPClass


Escenario = "\Escenario 4"
Carpeta = r"" + Escenario # Añadir carpeta
df_escenario = pd.read_csv(Carpeta + "\Puntos.csv", encoding="utf-8", sep=";")
df_dispositivos = pd.read_csv(Carpeta + "\Dispositivos.csv", encoding="utf-8", sep=";")

# Seeds = np.arange(2023, 2033, 1)
# Seeds = np.arange(2033, 2043, 1)
Seeds = np.arange(2023, 2053, 1)



iter_greedy=500
iter_local_search=200
verbose = True

for seed in Seeds:
    random.seed(int(seed))
    print(f"Inicio seed {seed} Escenario 1")
    GRASP = GRASPClass(df_escenario = df_escenario,
                       df_dispositivos = df_dispositivos,
                       iter_greedy=iter_greedy,
                       iter_local_search=iter_local_search,
                       seed=seed, 
                       verbose=verbose)
    
    GRASP.start_algorithm()
    # Representacion_Fitness()
    np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Disp.npy",
            GRASP.Mejor_Individuo_Disp)
    np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Ang.npy",
            GRASP.Mejor_Individuo_Ang)
    np.save(Carpeta + f"\Seed_{seed}_Fitness_Grafica.npy",
            GRASP.cost)    