# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra

"""
import numpy as np
import pandas as pd


def Fitness(individuo: np.array, 
            CoordPtosDispositivos: np.array, 
            CoordPtosVigilancia: np.array, 
            Matriz_Distancia: np.array, 
            df_escenario: pd.DataFrame, 
            df_dispositivos: pd.DataFrame):

    # Inicialización de variables necesarias
    Num_Disp = df_dispositivos.shape[0] # Número de dispositivos total
    Num_Disp_Usados = np.zeros((Num_Disp)) # Array para contar el número de dispositivos de cada tipo usados
    Coste_Disp = df_dispositivos["Coste (euros)"] # Array con los costes de cada dispositivo
    
    # Array de puntos cubiertos.
    # Las dos primeras columnas son las coordenadas de puntos de vigilancia
    # El resto de columnas es un booleano si está cubierto por cada dispositivo
    Ptos_Cubiertos = np.zeros((CoordPtosVigilancia.shape[0],
                               CoordPtosVigilancia.shape[1]+Num_Disp)) 
    Ptos_Cubiertos[:,:2] = CoordPtosVigilancia
    # Ptos_Cubiertos[:,0] = CoordPtosVigilancia[:,1]
    # Ptos_Cubiertos[:,1] = CoordPtosVigilancia[:,0]
    
    # Bucle principal para ver los puntos cubiertos, recorre el individuo.
    for i_ind, i_val in enumerate(individuo):
        if i_val == 0:
            continue
        # Decodificacion del valor decimal a binario
        aux = list(bin(i_val)[2:].zfill(Num_Disp))
        # Bucle del valor en binario
        for j_ind, j_val in enumerate(aux):
            # Si el dispositivo está puesto (1)
            if int(j_val) == 1:
                # Distancia que cubre dicho dispositivo
                Dist = df_dispositivos.iloc[j_ind]["Distancia (m)"]
                # Puntos cubiertos con ese dispositivo, es decir, su distancia es menor a la del dispositivo
                Ptos_Cubiertos[:,2+j_ind] = 1 * np.logical_or(Ptos_Cubiertos[:,2+j_ind],(Matriz_Distancia[i_ind,:] < Dist))
                # Se ha utilizado un dispositivo y se añade al array
                Num_Disp_Usados[j_ind] += 1
    # A: Puntos en los que es necesario estar cubierto por un dispositivo
    # B: Puntos cubiertos por cada dispositivo           
    A = np.array(df_escenario)[:,3:]
    B = Ptos_Cubiertos[:,2:]
    
    # Resultado = not(A) + B
    Result = np.logical_or(np.logical_not(A),B)
    # El número de fallos viene determinado por los puntos que necesitan
    # ser cubiertos y no se cubren
    Num_Fallos = np.sum(np.logical_not(Result))
    # El coste del individuo consta de una penalización grande por cada punto
    # no cubierto y por el coste en euros de los dispositivos
    Coste_Individuo = 1000000*Num_Fallos + np.sum(Coste_Disp*Num_Disp_Usados)

    return Coste_Individuo

def Fitness_ang(individuo_dist: np.array,
                individuo_ang: np.array,
                CoordPtosDispositivos: np.array, 
                CoordPtosVigilancia: np.array, 
                Matriz_Distancia: np.array,
                Matriz_Angulo: np.array,
                df_escenario: pd.DataFrame, 
                df_dispositivos: pd.DataFrame):

    # Inicialización de variables necesarias
    Num_Disp = df_dispositivos.shape[0] # Número de dispositivos total
    Num_Disp_Usados = np.zeros((Num_Disp)) # Array para contar el número de dispositivos de cada tipo usados
    Coste_Disp = df_dispositivos["Coste (euros)"] # Array con los costes de cada dispositivo
    
    # Array de puntos cubiertos.
    # Las dos primeras columnas son las coordenadas de puntos de vigilancia
    # El resto de columnas es un booleano si está cubierto por cada dispositivo
    Ptos_Cubiertos = np.zeros((CoordPtosVigilancia.shape[0],
                               CoordPtosVigilancia.shape[1]+Num_Disp)) 
    Ptos_Cubiertos[:,:2] = CoordPtosVigilancia
    
    # Bucle principal para ver los puntos cubiertos, recorre el individuo.
    for i_ind, i_val in enumerate(individuo_dist):
        # Decodificacion del valor decimal a binario
        if i_val == 0:
            continue
        aux = list(bin(i_val)[2:].zfill(Num_Disp))
        # Bucle del valor en binario
        for j_ind, j_val in enumerate(aux):
            # Si el dispositivo está puesto (1)
            if int(j_val) == 1:
                # Distancia que cubre dicho dispositivo
                Dist = df_dispositivos.iloc[j_ind]["Distancia (m)"]
                Ang = df_dispositivos.iloc[j_ind]["Angulo (grados)"]
                # Puntos cubiertos con ese dispositivo, es decir, su distancia es menor a la del dispositivo
                # Puntos cubiertos por distancia
                Ptos_Cub_Dist = (Matriz_Distancia[i_ind,:] < Dist)
                # Puntos cubiertos por ángulo
                Ang_1 = individuo_ang[i_ind] - Ang/2
                Ang_2 = individuo_ang[i_ind] + Ang/2
                if Ang_1 < 0:
                    Ang_1 = 360 + Ang_1
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                elif Ang_2 > 360:
                    Ang_2 = Ang_2 % 360 # Si la suma está por encima de 360, se convierte en el módulo
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                else:
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_and(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                Ptos_Cubiertos_Dist_Ang = np.logical_and(Ptos_Cubiertos_Ang,Ptos_Cub_Dist)
                Ptos_Cubiertos[:,2+j_ind] = 1 * np.logical_or(Ptos_Cubiertos[:,2+j_ind],Ptos_Cubiertos_Dist_Ang)
                # Se ha utilizado un dispositivo y se añade al array
                Num_Disp_Usados[j_ind] += 1
    # A: Puntos en los que es necesario estar cubierto por un dispositivo
    # B: Puntos cubiertos por cada dispositivo           
    A = np.array(df_escenario)[:,3:]
    B = Ptos_Cubiertos[:,2:]
    
    # Resultado = not(A) + B
    Result = np.logical_or(np.logical_not(A),B)
    # El número de fallos viene determinado por los puntos que necesitan
    # ser cubiertos y no se cubren
    Num_Fallos = np.sum(np.logical_not(Result))
    # El coste del individuo consta de una penalización grande por cada punto
    # no cubierto y por el coste en euros de los dispositivos
    Coste_Individuo = 1000000*Num_Fallos + np.sum(Coste_Disp*Num_Disp_Usados)

    return Coste_Individuo



def Fitness_ang_vision(individuo_dist: np.array,
                       individuo_ang: np.array,
                       CoordPtosDispositivos: np.array, 
                       CoordPtosVigilancia: np.array, 
                       Matriz_Distancia: np.array,
                       Matriz_Angulo: np.array,
                       Matriz_Vision: np.array,
                       df_escenario: pd.DataFrame, 
                       df_dispositivos: pd.DataFrame):

    # Inicialización de variables necesarias
    Num_Disp = df_dispositivos.shape[0] # Número de dispositivos total
    Num_Disp_Usados = np.zeros((Num_Disp)) # Array para contar el número de dispositivos de cada tipo usados
    Coste_Disp = df_dispositivos["Coste (euros)"] # Array con los costes de cada dispositivo
    Matriz_Distancia_Probabilidad = np.zeros((Num_Disp,Matriz_Distancia.shape[0],Matriz_Distancia.shape[1]))
    
    # Se convierten las distancias a probabilidad de detección en función de los dispositivos
    for i, dist in enumerate(df_dispositivos["Distancia (m)"]):
        Matriz_Distancia_Probabilidad[i,:,:] = -1/(1+np.exp(-0.5*(Matriz_Distancia-dist)))+1
        
    # Array de puntos cubiertos.
    # Las dos primeras columnas son las coordenadas de puntos de vigilancia
    # El resto de columnas es un booleano si está cubierto por cada dispositivo
    Ptos_Cubiertos = np.zeros((CoordPtosVigilancia.shape[0],
                               CoordPtosVigilancia.shape[1]+Num_Disp))
    # Ptos_Cubiertos_Dist_aux = np.zeros((Num_Disp,CoordPtosVigilancia.shape[0],CoordPtosDispositivos.shape[0]))
    Ptos_Cubiertos_Dist_aux = np.ones((CoordPtosVigilancia.shape[0],Num_Disp))
    Ptos_Cubiertos[:,:2] = CoordPtosVigilancia
    
    # Bucle principal para ver los puntos cubiertos, recorre el individuo.
    for i_ind, i_val in enumerate(individuo_dist):
        if i_val == 0:
            continue
        # Decodificacion del valor decimal a binario
        aux = list(bin(i_val)[2:].zfill(Num_Disp))
        # Bucle del valor en binario
        for j_ind, j_val in enumerate(aux):
            # Si el dispositivo está puesto (1)
            if int(j_val) == 1:
                # Distancia que cubre dicho dispositivo
                Dist = df_dispositivos.iloc[j_ind]["Distancia (m)"]
                Ang = df_dispositivos.iloc[j_ind]["Angulo (grados)"]
                # Puntos cubiertos con ese dispositivo, es decir, su distancia es menor a la del dispositivo
                # Puntos cubiertos por distancia
                Ptos_Cubiertos_Dist_aux[:,j_ind] = Ptos_Cubiertos_Dist_aux[:,j_ind] * (1-Matriz_Distancia_Probabilidad[j_ind,i_ind,:])
                # Puntos cubiertos por ángulo
                Ang_1 = individuo_ang[i_ind] - Ang/2
                Ang_2 = individuo_ang[i_ind] + Ang/2
                if Ang_1 < 0:
                    Ang_1 = 360 + Ang_1
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                elif Ang_2 > 360:
                    Ang_2 = Ang_2 % 360 # Si la suma está por encima de 360, se convierte en el módulo
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                else:
                    Ptos_Cubiertos_Ang_min = (Matriz_Angulo[i_ind,:] >= Ang_1)
                    Ptos_Cubiertos_Ang_max = (Matriz_Angulo[i_ind,:] < Ang_2)
                    Ptos_Cubiertos_Ang = np.logical_and(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
                Ptos_Cubiertos_Vision = Matriz_Vision[i_ind,:]
                Ptos_Cubiertos_Ang_Vision = np.logical_and(Ptos_Cubiertos_Ang,Ptos_Cubiertos_Vision)
                # Ptos_Cubiertos_Dist_Ang = np.logical_and(Ptos_Cubiertos_Ang,Ptos_Cub_Dist)
                # Ptos_Cubiertos_Dist_Ang_Vision = np.logical_and(Ptos_Cubiertos_Dist_Ang,Ptos_Cubiertos_Vision)
                Ptos_Cubiertos[:,2+j_ind] = 1 * np.logical_or(Ptos_Cubiertos[:,2+j_ind],Ptos_Cubiertos_Ang_Vision)
                # Se ha utilizado un dispositivo y se añade al array
                Num_Disp_Usados[j_ind] += 1
    # A: Puntos en los que es necesario estar cubierto por un dispositivo
    # B: Puntos cubiertos por cada dispositivo
    Ptos_Cubiertos_Dist = 1*((1- Ptos_Cubiertos_Dist_aux) > 0.5)
    Ptos_Cubiertos[:,2:] = np.logical_and(Ptos_Cubiertos[:,2:],Ptos_Cubiertos_Dist)
    A = np.array(df_escenario)[:,3:]
    B = Ptos_Cubiertos[:,2:]
    
    # Resultado = not(A) + B
    Result = np.logical_or(np.logical_not(A),B)
    # El número de fallos viene determinado por los puntos que necesitan
    # ser cubiertos y no se cubren
    Num_Fallos = np.sum(np.logical_not(Result))
    # El coste del individuo consta de una penalización grande por cada punto
    # no cubierto y por el coste en euros de los dispositivos
    Coste_Individuo = 1000000*Num_Fallos + np.sum(Coste_Disp*Num_Disp_Usados)

    return Coste_Individuo

def Ptos_Cubiertos_Dispositivo(individuo_dist: np.array,
                               individuo_ang: np.array,
                               array_vision: np.array,
                               MD: np.array,
                               MA: np.array,
                               MV: np.array,
                               Dist, 
                               Ang):
    for i_ind, i_val in enumerate(individuo_dist):
        if int(i_val) == 1:
            Ptos_Cub_Dist = (MD[i_ind,:] < Dist)
            # Ang_1 = individuo_ang[i_ind] - Ang/2
            # Ang_2 = individuo_ang[i_ind] + Ang/2

            # Ang_1 = 360 + Ang_1
            # Ptos_Cubiertos_Ang_min = (MA[i_ind,:] >= Ang_1)
            # Ptos_Cubiertos_Ang_max = (MA[i_ind,:] < Ang_2)
            # Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
            
            # Puntos cubiertos por ángulo
            Ang_1 = individuo_ang[i_ind] - Ang/2
            Ang_2 = individuo_ang[i_ind] + Ang/2
            if Ang_1 < 0:
                Ang_1 = 360 + Ang_1
                Ptos_Cubiertos_Ang_min = (MA[i_ind,:] >= Ang_1)
                Ptos_Cubiertos_Ang_max = (MA[i_ind,:] < Ang_2)
                Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
            elif Ang_2 > 360:
                Ang_2 = Ang_2 % 360 # Si la suma está por encima de 360, se convierte en el módulo
                Ptos_Cubiertos_Ang_min = (MA[i_ind,:] >= Ang_1)
                Ptos_Cubiertos_Ang_max = (MA[i_ind,:] < Ang_2)
                Ptos_Cubiertos_Ang = np.logical_or(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)
            else:
                Ptos_Cubiertos_Ang_min = (MA[i_ind,:] >= Ang_1)
                Ptos_Cubiertos_Ang_max = (MA[i_ind,:] < Ang_2)
                Ptos_Cubiertos_Ang = np.logical_and(Ptos_Cubiertos_Ang_min,Ptos_Cubiertos_Ang_max)

            Ptos_Cubiertos_Vision = MV[i_ind,:]
            Ptos_Cubiertos_Ang_Vision = np.logical_and(Ptos_Cubiertos_Ang,Ptos_Cubiertos_Vision)
            Ptos_Cubiertos = np.logical_and(Ptos_Cubiertos_Ang_Vision,Ptos_Cub_Dist)
            array_vision=np.logical_or(Ptos_Cubiertos,array_vision)

    return array_vision