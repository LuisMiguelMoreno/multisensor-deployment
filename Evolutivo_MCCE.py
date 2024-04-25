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
from Fitness_Function import Fitness, Fitness_ang, Fitness_ang_vision

class EvolutiveClass:
    def __init__(self, df_escenario=None, df_dispositivos=None, Num_Individuos=200, Num_Generaciones=10, Tam_Individuos=10, Prob_Padres=0.5, Prob_Mutacion=0.02, Prob_Cruce=0.5, seed=2024, verbose = False):
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
        Pob_Ini_Disp = np.random.randint(0,Num_Max, size=(Fil,Col))
        
        # Inicializacón con petencias de 2, que indica que solo se utiliza 1 dispositivo en cada punto
        for i_ind, i_val in enumerate(Pob_Ini_Disp):
            for j_ind, j_val in enumerate(i_val):
                if np.random.rand() < 0.75:
                    Pob_Ini_Disp [i_ind, j_ind] = np.random.choice(Mult_Dos, 1)[0]
        Pob_Ini_Ang = np.random.randint(0,360, size=(Fil,Col))
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
        aux1 = int(np.random.choice(np.where(individuo != 0)[0],1))
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
        self.Pob_Ini_Disp, self.Pob_Ini_Ang = self.PoblacionInicial()
        self.Coste_Pob = np.zeros((self.Num_Individuos))
        self.Matriz_Distancia , self.Matriz_Angulo, self.Matriz_Vision = CalcularDistanciaAngulosVisionPuntos(self.CoordPtosDispositivos, self.CoordPtosVigilancia)
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
    Num_Generaciones = 100
    Tam_Individuos = 100
    Num_Max = 2**5
    Prob_Padres = 0.5
    Prob_Mutacion = 0.3
    Prob_Cruce = 0.5
    
    Ev1 = EvolutiveClass(df_escenario = pd.read_csv(r"C:/Users/LuisMi-ISDEFE/Documents/GitHub/Seguridad_MCCE/Puntos.csv", encoding="utf-8", sep=";"),
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