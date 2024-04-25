# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np
import random
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from Aux_Func import CalcularDistanciaAngulosVisionPuntos

from Fitness_Function import Fitness_ang_vision, Ptos_Cubiertos_Dispositivo

class GRASPClass:
    def __init__(self, 
                 df_escenario=None, 
                 df_dispositivos=None,
                 iter_greedy=100,
                 iter_local_search=100,
                 seed=2024, 
                 verbose = False):
        self.df_escenario = df_escenario
        self.df_dispositivos = df_dispositivos
        self.iter_greedy = iter_greedy
        self.iter_local_search = iter_local_search
        self.Num_Disp = self.df_dispositivos.shape[0]
        self.CoordPtosVigilancia = np.array(self.df_escenario[["Coord_x (m)","Coord_y (m)"]])
        self.CoordPtosDispositivos = np.array(self.CoordPtosVigilancia[np.where(self.df_escenario["Punto de Vigilancia"] == 1)[0],:])
        self.Tam_Individuos = np.where(self.df_escenario["Punto de Vigilancia"] == 1)[0].shape[0]
        self.Num_Max = 2**self.Num_Disp
        np.random.seed(seed)
        self.verbose = verbose
    
    
    
    
    def greedy(self):
        Individuo_Inicializacion_Disp = np.zeros((self.Tam_Individuos,self.Num_Disp))
        Individuo_Inicializacion_Ang = np.random.randint(0,360,size=(self.Tam_Individuos))
        Ptos_Cubiertos = 1 - self.df_escenario.to_numpy()[:,3:]
        for disp in range(self.Num_Disp):
            list_disp_1 = []
            list_disp_2 = []
            # ptos_cubiertos_disp = Ptos_Cubiertos[:,disp]
            ptos_cubiertos_disp = np.zeros((Ptos_Cubiertos.shape[0]))
            while int(sum(ptos_cubiertos_disp)) != len(ptos_cubiertos_disp):
                num_ptos_cubiertos_antes = int(sum(ptos_cubiertos_disp))
                ind_ptos_no_cubiertos = np.where(ptos_cubiertos_disp == 0)[0]
                ind_punto = np.random.choice(ind_ptos_no_cubiertos,size=1)[0]
                ind_dispositivo = np.random.choice(np.where(self.Matriz_Distancia[:,ind_punto] < self.df_dispositivos["Distancia (m)"].iloc[disp])[0],1)[0]
                list_disp_1.append(ind_dispositivo)
                Individuo_Inicializacion_Disp[ind_dispositivo,disp] = 1
                ptos_cubiertos_disp = Ptos_Cubiertos_Dispositivo(Individuo_Inicializacion_Disp[:,disp],
                                                                 Individuo_Inicializacion_Ang,
                                                                 ptos_cubiertos_disp,
                                                                 self.Matriz_Distancia,
                                                                 self.Matriz_Angulo,
                                                                 self.Matriz_Vision,
                                                                 self.df_dispositivos["Distancia (m)"].iloc[disp], 
                                                                 self.df_dispositivos["Angulo (grados)"].iloc[disp])
                num_ptos_cubiertos_despues = int(sum(ptos_cubiertos_disp))

                if len(list_disp_1) > 3000:
                    individuo = 2**4*Individuo_Inicializacion_Disp[:,0]+2**3*Individuo_Inicializacion_Disp[:,1]+2**2*Individuo_Inicializacion_Disp[:,2]+2**1*Individuo_Inicializacion_Disp[:,3]+2**0*Individuo_Inicializacion_Disp[:,4]
                    individuo = individuo.astype(int)
                    Coste = Fitness_ang_vision(individuo,
                                                      Individuo_Inicializacion_Ang,
                                                      self.CoordPtosDispositivos, 
                                                      self.CoordPtosVigilancia, 
                                                      self.Matriz_Distancia,
                                                      self.Matriz_Angulo,
                                                      self.Matriz_Vision,
                                                      self.df_escenario, 
                                                      self.df_dispositivos)
                    return individuo, Individuo_Inicializacion_Ang, Coste
                if num_ptos_cubiertos_despues != len(ptos_cubiertos_disp):
                    if (num_ptos_cubiertos_antes == num_ptos_cubiertos_despues) and (ind_dispositivo not in list_disp_2):
                        Individuo_Inicializacion_Disp[ind_dispositivo,disp] = 0
                    else:
                        list_disp_2.append(ind_dispositivo)
                        
        individuo = 2**4*Individuo_Inicializacion_Disp[:,0]+2**3*Individuo_Inicializacion_Disp[:,1]+2**2*Individuo_Inicializacion_Disp[:,2]+2**1*Individuo_Inicializacion_Disp[:,3]+2**0*Individuo_Inicializacion_Disp[:,4]
        individuo = individuo.astype(int)
        Coste = Fitness_ang_vision(individuo,
                                          Individuo_Inicializacion_Ang,
                                          self.CoordPtosDispositivos, 
                                          self.CoordPtosVigilancia, 
                                          self.Matriz_Distancia,
                                          self.Matriz_Angulo,
                                          self.Matriz_Vision,
                                          self.df_escenario, 
                                          self.df_dispositivos)
        return individuo, Individuo_Inicializacion_Ang, Coste


    
    def local_search_sensor(self, individual, Num_Max=32):      
        """
        Types of local search (in sensors):
            0: Take a gene and increase the value by 1 unit
            1: Take a gene and decrease the value by 1 unit
            2: Take a gene and replace the value with a random number
            3: Take a gene and set the value to 0.
        """
        Type = np.random.randint(0,4)
        position = np.random.randint(0, individual.shape[0])
        new_individual = np.copy(individual)
        if Type == 0:
            new_individual[position] += 1
        elif Type == 1:
            new_individual[position] -= 1
        elif Type == 2:
            new_individual[position] = np.random.randint(0,Num_Max) 
        elif Type == 3:
            new_individual[position] = 0
                
        return np.clip(new_individual,0,Num_Max-1)
    
    def local_search_angle(self, individual, Num_Max=360):      
        """
        Types of local search (in angles):
            0: Take a gene and increase the value by 30 unit
            1: Take a gene and decrease the value by 30 unit
            2: Take a gene and replace the value with a random number
            3: Take a gene and set the value to 0.
        """
        Type = np.random.randint(0,4)
        position = np.random.randint(0, individual.shape[0])
        new_individual = np.copy(individual)
        if Type == 0:
            new_individual[position] += 30
        elif Type == 1:
            new_individual[position] -= 30
        elif Type == 2:
            new_individual[position] = np.random.randint(0,Num_Max) 
        elif Type == 3:
            new_individual[position] = 0
                
        return np.clip(new_individual,0,Num_Max-1)   
    
    
    def start_algorithm(self):
        self.cost = np.zeros(self.iter_greedy)
        self.Matriz_Distancia,self.Matriz_Angulo,self.Matriz_Vision = CalcularDistanciaAngulosVisionPuntos(self.CoordPtosDispositivos, 
                                                                  self.CoordPtosVigilancia)

        for i in range (self.iter_greedy):
            individual_sensor, individual_angle, self.cost[i] = self.greedy()
            # print (f"i: {i}")
            t_inicio = time.process_time()
            for j in range (self.iter_local_search):
                # print (f"j: {j}")
                individual_sensor_new = self.local_search_sensor(individual_sensor, self.Num_Max)
                individual_angle_new = self.local_search_angle(individual_angle, 360)
                cost_new = Fitness_ang_vision(individual_sensor_new,
                                              individual_angle_new,
                                              self.CoordPtosDispositivos, 
                                              self.CoordPtosVigilancia, 
                                              self.Matriz_Distancia,
                                              self.Matriz_Angulo,
                                              self.Matriz_Vision,
                                              self.df_escenario, 
                                              self.df_dispositivos)
                # print(cost_new)
                if cost_new < self.cost[i]: # If minimization is taken into account
                    self.cost[i] = cost_new
                    individual_sensor = individual_sensor_new
                    individual_angle = individual_angle_new
                
            t_gen = time.process_time()
            if self.verbose:
                print(f"Tiempo en generación {i}: {t_gen-t_inicio}s. Coste = {self.cost[i]}€")
                
        self.Mejor_Individuo_Disp = individual_sensor
        self.Mejor_Individuo_Ang = individual_angle
                    
                    
                    
if __name__ == "__main__":
    # Definicion de los parámetros del genético
    iter_greedy=10
    iter_local_search=10
    seed=2023
    verbose = True
    ruta_df_escenario = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (CPLEX)\Escenario 1\Puntos.csv"
    ruta_df_dispositivos = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (CPLEX)\Escenario 1\Dispositivos.csv"
    print("Escenario 1")
    GRASP = GRASPClass(df_escenario = pd.read_csv(ruta_df_escenario, encoding="utf-8", sep=";"),
                       df_dispositivos = pd.read_csv(ruta_df_dispositivos, encoding="utf-8", sep=";"),
                       iter_greedy=iter_greedy,
                       iter_local_search=iter_local_search,
                       seed=seed, 
                       verbose=verbose)
    GRASP.start_algorithm()
