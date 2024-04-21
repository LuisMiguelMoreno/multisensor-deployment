# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:50:46 2024

@author: LuisMi-ISDEFE
"""

import pandas as pd
import numpy as np
import random
from Evolutivo_MCCE_inicializacion import EvolutiveClass


#%% Escenario 6 (Completo)
# Parámetros
Escenario = "\Escenario 3"
Carpeta = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (CPLEX)" + Escenario
df_escenario = pd.read_csv(Carpeta + "\Puntos.csv", encoding="utf-8", sep=";")
df_dispositivos = pd.read_csv(Carpeta + "\Dispositivos.csv", encoding="utf-8", sep=";")
# df_solucion = pd.read_csv(Carpeta + "\Solucion CPLEX.csv").to_numpy()
# individuo = 2**4*df_solucion[:,3]+2**3*df_solucion[:,4]+2**2*df_solucion[:,5]+2**1*df_solucion[:,6]+2**0*df_solucion[:,7]
# individuo = individuo.astype(int)
# Seeds = np.arange(2023, 2033, 1)
# Seeds = np.arange(2033, 2043, 1)
# Seeds = np.arange(2023, 2053, 1)
Seeds = [2023]



Num_Individuos = 100
Num_Generaciones = 1000
Tam_Individuos = 100
Prob_Padres = 0.5
Prob_Mutacion = 0.7
Prob_Cruce = 0.5
for seed in Seeds:
    random.seed(int(seed))
    print(f"Inicio seed {seed} Escenario 3")
    Ev1 = EvolutiveClass(df_escenario = df_escenario, 
                              df_dispositivos = df_dispositivos,
                              Num_Individuos = Num_Individuos, 
                              Num_Generaciones = Num_Generaciones, 
                              Tam_Individuos = np.where(df_escenario["Punto de Vigilancia"] == 1)[0].shape[0], 
                              Prob_Padres = Prob_Padres, 
                              Prob_Mutacion = Prob_Mutacion, 
                              Prob_Cruce = Prob_Cruce,
                              seed = seed,
                              verbose = True)
    
    Ev1.InicioAlgoritmo()
    # Representacion_Fitness()
    # np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Disp.npy",
    #         Ev1.Mejor_Individuo_Disp)
    # np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Ang.npy",
    #         Ev1.Mejor_Individuo_Ang)
    # np.save(Carpeta + f"\Seed_{seed}_Fitness_Grafica.npy",
    #         Ev1.Fitness_Grafica)    
    
#%% Escenario 7 (Completo)
# Parámetros
# Escenario = "\Escenario 7"
# Carpeta = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (CPLEX)" + Escenario
# df_escenario = pd.read_csv(Carpeta + "\Puntos.csv", encoding="utf-8", sep=";")
# df_dispositivos = pd.read_csv(Carpeta + "\Dispositivos.csv", encoding="utf-8", sep=";")
# # df_solucion = pd.read_csv(Carpeta + "\Solucion CPLEX.csv").to_numpy()
# # individuo = 2**4*df_solucion[:,3]+2**3*df_solucion[:,4]+2**2*df_solucion[:,5]+2**1*df_solucion[:,6]+2**0*df_solucion[:,7]
# # individuo = individuo.astype(int)
# # Seeds = np.arange(2023, 2033, 1)


# Num_Individuos = 100
# Num_Generaciones = 1000
# Tam_Individuos = 100
# Prob_Padres = 0.5
# Prob_Mutacion = 0.7
# Prob_Cruce = 0.5
# for seed in Seeds:
#     random.seed(int(seed))
#     print(f"Inicio seed {seed} Escenario 7")
#     Ev1 = EvolutiveClass(df_escenario = df_escenario, 
#                               df_dispositivos = df_dispositivos,
#                               Num_Individuos = Num_Individuos, 
#                               Num_Generaciones = Num_Generaciones, 
#                               Tam_Individuos = np.where(df_escenario["Punto de Vigilancia"] == 1)[0].shape[0], 
#                               Prob_Padres = Prob_Padres, 
#                               Prob_Mutacion = Prob_Mutacion, 
#                               Prob_Cruce = Prob_Cruce,
#                               seed = seed,
#                               verbose = False)
    
#     Ev1.InicioAlgoritmo()
#     # Representacion_Fitness()
#     np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Disp.npy",
#             Ev1.Mejor_Individuo_Disp)
#     np.save(Carpeta + f"\Seed_{seed}_Mejor_Individuo_Ang.npy",
#             Ev1.Mejor_Individuo_Ang)
#     np.save(Carpeta + f"\Seed_{seed}_Fitness_Grafica.npy",
#             Ev1.Fitness_Grafica)    