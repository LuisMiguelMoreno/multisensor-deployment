# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:48:39 2024

@author: LuisMi-ISDEFE
"""

import pandas as pd
import numpy as np
import random
from GRASP import GRASPClass


Escenario = "\Escenario 3"
Carpeta = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (GRASP)" + Escenario
df_escenario = pd.read_csv(Carpeta + "\Puntos.csv", encoding="utf-8", sep=";")
df_dispositivos = pd.read_csv(Carpeta + "\Dispositivos.csv", encoding="utf-8", sep=";")
# df_solucion = pd.read_csv(Carpeta + "\Solucion CPLEX.csv").to_numpy()
# individuo = 2**4*df_solucion[:,3]+2**3*df_solucion[:,4]+2**2*df_solucion[:,5]+2**1*df_solucion[:,6]+2**0*df_solucion[:,7]
# individuo = individuo.astype(int)
# Seeds = np.arange(2023, 2033, 1)
# Seeds = np.arange(2033, 2043, 1)
Seeds = np.arange(2023, 2053, 1)
# Seeds = [2023]


iter_greedy=100
iter_local_search=1000
verbose = False

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