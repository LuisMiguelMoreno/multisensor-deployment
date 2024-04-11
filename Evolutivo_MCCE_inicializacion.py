# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:51:44 2023

@author: LuisMi-ISDEFE
"""

import numpy as np
import random
import pandas as pd
import time

from Aux_Func import CalcularDistanciaAngulosVisionPuntos
from Fitness_Function import Fitness, Fitness_ang, Fitness_ang_vision, Ptos_Cubiertos_Dispositivo

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def imprimir_escenario_disp(coords_escenario,
                            coords_dispositivos,
                            coords_puntos_cubrir,
                            individuo_disp,
                            individuo_ang, 
                            radio,
                            angulo):
     
    fig, ax = plt.subplots()
    ax.scatter(coords_escenario[:,0],coords_escenario[:,1])
    ind_cubrir = np.where(coords_puntos_cubrir == 1)[0]
    ind_sol = np.where(individuo_disp == 1)[0]
    ax.scatter(coords_escenario[ind_cubrir,0],coords_escenario[ind_cubrir,1], c="r")
    plt.xlim([min(coords_escenario[:,0])-0.5, max(coords_escenario[:,0])+0.5])
    plt.ylim([min(coords_escenario[:,1])-0.5, max(coords_escenario[:,1])+0.5])
    for j in ind_sol:
        ang_1 = individuo_ang[j] - angulo / 2
        ang_2 = individuo_ang[j] + angulo / 2
        circle = Wedge((coords_dispositivos[j,0], coords_dispositivos[j,1]), 
                    2*radio, ang_1, ang_2, color='y', alpha=0.1)
        ax.add_patch(circle)
        # circle = Wedge((coords_dispositivos[j,0], coords_dispositivos[j,1]), 
        #            radio, ang_1, ang_2, color='y', alpha=0.2)
        # ax.add_patch(circle)


class EvolutiveClass:
    def __init__(self, 
                 df_escenario=None, 
                 df_dispositivos=None, 
                 Num_Individuos=200, 
                 Num_Generaciones=10, 
                 Tam_Individuos=10, 
                 Prob_Padres=0.5, 
                 Prob_Mutacion=0.02, 
                 Prob_Cruce=0.5,
                 seed=2024, 
                 verbose = False):
        self.df_escenario = df_escenario
        self.df_dispositivos = df_dispositivos
        self.Num_Disp = self.df_dispositivos.shape[0]
        self.CoordPtosVigilancia = np.array(self.df_escenario[["Coord_x (m)","Coord_y (m)"]])
        self.CoordPtosDispositivos = np.array(self.CoordPtosVigilancia[np.where(self.df_escenario["Punto de Vigilancia"] == 1)[0],:])
        self.Num_Individuos = Num_Individuos
        self.Num_Generaciones = Num_Generaciones
        self.Tam_Individuos = np.where(self.df_escenario["Punto de Vigilancia"] == 1)[0].shape[0]
        self.Num_Max = 2**self.Num_Disp
        self.Prob_Padres = Prob_Padres
        self.Num_Padres = round(self.Num_Individuos * self.Prob_Padres)
        self.Prob_Mutacion = Prob_Mutacion
        self.Prob_Cruce = Prob_Cruce
        np.random.seed(seed)
        self.verbose = verbose
        

    def ImprimirInformacion(self):
        print("Los parámetros del algoritmo genético son los siguientes:")
        print("Número de individuos: " + str(self.Num_Individuos))
        print("Número de generaciones: " + str(self.Num_Generaciones))
        print("Probabilidad de padres que sobreviven: " + str(self.Prob_Padres))
        print("Número de padres: " + str(self.Num_Padres))
        print("Probabilidad de mutación: " + str(self.Prob_Mutacion))
        print("Probabilidad de cruce: " + str(self.Prob_Cruce))
    
    def IndividuoInicial(self):
        Individuo_Inicializacion_Disp = np.zeros((self.Tam_Individuos,self.Num_Disp))
        # Individuo_Inicializacion_Ang = np.zeros((self.Tam_Individuos,self.Num_Disp))
        # Individuo_Inicializacion_Ang = 0*np.ones((self.Tam_Individuos,self.Num_Disp))
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
                # ind_dispositivo = np.where(self.Matriz_Distancia[:,ind_punto] == min(self.Matriz_Distancia[:,ind_punto]))[0][0]
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
                    return individuo, Individuo_Inicializacion_Ang
                if num_ptos_cubiertos_despues != len(ptos_cubiertos_disp):
                    if (num_ptos_cubiertos_antes == num_ptos_cubiertos_despues) and (ind_dispositivo not in list_disp_2):
                        Individuo_Inicializacion_Disp[ind_dispositivo,disp] = 0
                    else:
                        # print(ind_dispositivo)
                        list_disp_2.append(ind_dispositivo)
                        # imprimir_escenario_disp(self.CoordPtosVigilancia,
                        #                             self.CoordPtosDispositivos,
                        #                             ptos_cubiertos_disp,
                        #                             Individuo_Inicializacion_Disp[:,disp],
                        #                             Individuo_Inicializacion_Ang[:,disp], 
                        #                             self.df_dispositivos["Distancia (m)"].iloc[disp],
                        #                             self.df_dispositivos["Angulo (grados)"].iloc[disp])
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
        # self.Mutar_Individuo_Inicial(individuo,
        #                              Individuo_Inicializacion_Ang[:,0])
        return individuo, Individuo_Inicializacion_Ang
    
    def Mutar_Individuo_Inicial(self, 
                                individuo_disp,
                                individuo_ang):
        
        Coste = Fitness_ang_vision(individuo_disp,
                                          individuo_ang,
                                          self.CoordPtosDispositivos, 
                                          self.CoordPtosVigilancia, 
                                          self.Matriz_Distancia,
                                          self.Matriz_Angulo,
                                          self.Matriz_Vision,
                                          self.df_escenario, 
                                          self.df_dispositivos)
        
        
    def PoblacionInicial(self, Fil=None, Col=None, Num_Max=None):
        if Fil == None:
            Fil = self.Num_Individuos
        if Col == None:
            Col = self.Tam_Individuos
        if Num_Max == None:
            Num_Max = self.Num_Max
        Mult_Dos = [2**i for i in range(self.Num_Disp)]
        Pob_Ini_Disp = np.zeros((Fil,Col))
        Pob_Ini_Ang = np.zeros((Fil,Col))
        Pob_Ini_Disp = np.random.randint(0, Num_Max, size=(Fil,Col))
        Pob_Ini_Ang = np.random.randint(0,360, size=(Fil,Col))
        
        
        # Inicializacón con potencias de 2, que indica que solo se utiliza 1 dispositivo en cada punto
        for i_ind, i_val in enumerate(Pob_Ini_Disp):
            num_rand = np.random.rand()
            if num_rand < 0.33:
                # Inicialización completamente random
                continue
            elif num_rand > 0.33 and num_rand < 0.67:
                # Inicializacion de buenas soluciones:
                    Pob_Ini_Disp[i_ind,:], Pob_Ini_Ang[i_ind,:] = self.IndividuoInicial()
                    # Pob_Ini_Ang[i_ind,:] = np.zeros(Pob_Ini_Ang.shape[1])
            else:
                # Inicializacion potencias 2
                for j_ind, j_val in enumerate(i_val):
                    if np.random.rand() < 0.75:
                        Pob_Ini_Disp [i_ind, j_ind] = np.random.choice(Mult_Dos, 1)[0]
            

                    
        return Pob_Ini_Disp, Pob_Ini_Ang

    def Seleccion(self, poblacion_inicial_disp, poblacion_inicial_ang, coste):
        index = np.argsort(coste)
        coste_ordenado = np.sort(coste)
        # coste_ordenado = coste_ordenado[0:self.Num_Padres]
        poblacion_actual_disp = poblacion_inicial_disp[index,:]
        poblacion_actual_disp = poblacion_actual_disp[0:self.Num_Padres,:]
        poblacion_actual_ang = poblacion_inicial_ang[index,:]
        poblacion_actual_ang = poblacion_actual_ang[0:self.Num_Padres,:]
        return poblacion_actual_disp, poblacion_actual_ang, coste_ordenado

    def Cruce (self, poblacion_disp, poblacion_ang, Num_Max = None):
        if Num_Max == None:
            Num_Max = self.Num_Max
        for i in range (self.Num_Individuos - self.Num_Padres):
            Indice_Padres = random.sample(range(self.Num_Padres), 2)            # Se elige aleatoriamente el indice de los padres
            
            # Cruce del individuo en los dispositivos
            Padre1 = poblacion_disp[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion_disp[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo_disp = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo_disp[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                if np.random.rand() < 0.5:
                    Hijo_disp = self.Mutacion(Hijo_disp, Num_Max)
                else:
                    Hijo_disp = self.Mutacion_Gaussiana(Hijo_disp, Num_Max/2)
            poblacion_disp = np.insert(poblacion_disp,self.Num_Padres+i,Hijo_disp, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado


            # Cruce del individuo en los angulos
            Padre1 = poblacion_ang[Indice_Padres[0],:]                              # Se coge el padre 1
            Padre2 = poblacion_ang[Indice_Padres[1],:]                              # Se coge el padre 2
            Hijo_ang = np.copy(Padre1)                                              # El hijo va a ser una copia del padre 1
            vector = 1*(np.random.rand(self.Tam_Individuos) > self.Prob_Cruce)  # Se genera un vector para seleccionar los genes del padre 2
            Hijo_ang[np.where(vector==1)[0]] = Padre2[np.where(vector==1)[0]]       # Los genes seleccionados del padre 2 pasan al hijo
            if np.random.rand() < self.Prob_Mutacion:                           # Se comprueba si el hijo va a mutar
                if np.random.rand() < 0.5:
                    Hijo_ang = self.Mutacion(Hijo_ang, 360)
                else:
                    Hijo_ang = self.Mutacion_Gaussiana(Hijo_ang, 360/2)
            poblacion_ang = np.insert(poblacion_ang,self.Num_Padres+i,Hijo_ang, axis = 0)   # Se añade a la población una vez que ha mutado y se ha reparado
        return poblacion_disp, poblacion_ang

    def Mutacion (self, individuo, Num_Max=None):                                
        # aux1 = np.random.randint(0, individuo.shape[0])                         # Se genera número aleatorio para ver la posición que muta
        try:
            aux1 = int(np.random.choice(np.where(individuo != 0)[0],1))
        except:
            return individuo
        # aux2 = np.random.randint(0,Num_Max)                                   # Se genera el número a modificar
        # individuo[aux1] = aux2
        individuo[aux1] = 0


        # A veces se hace una mutación dura, haciendo que el individuo sea nuevo
        # if np.random.rand() < 0.3: # 30% de probabilidad
        #     individuo = np.random.randint(0,Num_Max,size=individuo.shape)     
        return individuo

    def Mutacion_Gaussiana(self, individuo, Media):
        Vec_Mut = Media * np.random.randn(individuo.shape[0])
        
        individuo = np.clip(np.array(Vec_Mut + individuo, dtype = np.int32),0,2*Media)
        return individuo

    def Reparacion(self):
        pass


    def InicioAlgoritmo(self):
        t_1 = time.time()
        self.Fitness_Grafica = []
        self.Matriz_Distancia , self.Matriz_Angulo, self.Matriz_Vision = CalcularDistanciaAngulosVisionPuntos(self.CoordPtosDispositivos, self.CoordPtosVigilancia)
        # self.IndividuoInicial()
        self.Pob_Ini_Disp, self.Pob_Ini_Ang = self.PoblacionInicial()
        self.Coste_Pob = np.zeros((self.Num_Individuos))
        for indice, individuo in enumerate(self.Pob_Ini_Disp):
            # self.Coste_Pob[indice] = self.Fitness(individuo)
            # self.Coste_Pob[indice] = Fitness_ang(individuo,
            #                                   self.Pob_Ini_Ang[indice],
            #                                   self.CoordPtosDispositivos, 
            #                                   self.CoordPtosVigilancia, 
            #                                   self.Matriz_Distancia,
            #                                   self.Matriz_Angulo,
            #                                   self.df_escenario, 
            #                                   self.df_dispositivos)
            self.Coste_Pob[indice] = Fitness_ang_vision(individuo,
                                              self.Pob_Ini_Ang[indice],
                                              self.CoordPtosDispositivos, 
                                              self.CoordPtosVigilancia, 
                                              self.Matriz_Distancia,
                                              self.Matriz_Angulo,
                                              self.Matriz_Vision,
                                              self.df_escenario, 
                                              self.df_dispositivos)
            # self.Coste_Pob[indice] = Fitness(individuo,
            #                                   self.CoordPtosDispositivos, 
            #                                   self.CoordPtosVigilancia, 
            #                                   self.Matriz_Distancia,
            #                                   self.df_escenario, 
            #                                   self.df_dispositivos)
        self.Pob_Act_Disp = np.copy(self.Pob_Ini_Disp)
        self.Pob_Act_Ang = np.copy(self.Pob_Ini_Ang)
        t_2 = time.time()
        # print(f"Tiempo sin paralelización: {t_2-t_1}s.")

        
        t_inicio = time.process_time()
        for generacion in range(self.Num_Generaciones):
            self.Pob_Act_Disp, self.Pob_Act_Ang, self.Coste_Pob = self.Seleccion(self.Pob_Act_Disp, self.Pob_Act_Ang, self.Coste_Pob)
            self.Pob_Act_Disp, self.Pob_Act_Ang = self.Cruce(self.Pob_Act_Disp, self.Pob_Act_Ang)
            for indice, individuo in enumerate(self.Pob_Act_Disp):
                if indice < self.Num_Padres:
                    continue
                # self.Coste_Pob[indice] = self.Fitness(individuo)
                # self.Coste_Pob[indice] = Fitness_ang(individuo,
                #                                   self.Pob_Act_Ang[indice],
                #                                   self.CoordPtosDispositivos, 
                #                                   self.CoordPtosVigilancia, 
                #                                   self.Matriz_Distancia, 
                #                                   self.Matriz_Angulo,
                #                                   self.df_escenario, 
                #                                   self.df_dispositivos)
                self.Coste_Pob[indice] = Fitness_ang_vision(individuo,
                                                  self.Pob_Act_Ang[indice],
                                                  self.CoordPtosDispositivos, 
                                                  self.CoordPtosVigilancia, 
                                                  self.Matriz_Distancia,
                                                  self.Matriz_Angulo,
                                                  self.Matriz_Vision,
                                                  self.df_escenario, 
                                                  self.df_dispositivos)
                # self.Coste_Pob[indice] = Fitness(individuo,
                #                                   self.CoordPtosDispositivos, 
                #                                   self.CoordPtosVigilancia, 
                #                                   self.Matriz_Distancia,
                #                                   self.df_escenario, 
                #                                   self.df_dispositivos)
            self.Fitness_Grafica.append(self.Coste_Pob[0])
            t_gen = time.process_time()
            if self.verbose:
                print(f"Tiempo en generación {generacion}: {t_gen-t_inicio}s. Coste = {self.Coste_Pob[0]}€")
        self.Mejor_Individuo_Disp = self.Pob_Act_Disp[0,:]
        self.Mejor_Individuo_Ang = self.Pob_Act_Ang[0,:]

if __name__ == "__main__":
    # Definicion de los parámetros del genético
    Num_Individuos = 100
    Num_Generaciones = 300
    Tam_Individuos = 100
    Num_Max = 2**5
    Prob_Padres = 0.5
    Prob_Mutacion = 0.3
    Prob_Cruce = 0.5
    df_escenario = r"C:\Users\LuisMi-ISDEFE\OneDrive - Universidad de Alcala\Universidad\Doctorado\Papers\Paper 2 MCCE\Resultados\Experimento 2 (CPLEX)\Escenario 4\Puntos.csv"
    Ev1 = EvolutiveClass(df_escenario = pd.read_csv(df_escenario, encoding="utf-8", sep=";"),
                         df_dispositivos = pd.read_csv(r"C:/Users/LuisMi-ISDEFE/Documents/GitHub/Seguridad_MCCE/Dispositivos.csv", encoding="utf-8", sep=";"),
                         Num_Individuos = Num_Individuos, 
                         Num_Generaciones = Num_Generaciones, 
                         Tam_Individuos = Tam_Individuos, 
                         Prob_Padres = Prob_Padres, 
                         Prob_Mutacion = Prob_Mutacion, 
                         Prob_Cruce = Prob_Cruce,
                         verbose=True)
    Ev1.ImprimirInformacion()
    Ev1.InicioAlgoritmo()
    a = Ev1.Fitness_Grafica