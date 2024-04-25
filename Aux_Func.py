# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np
from scipy.signal import convolve2d

def Distancia (Coord1, Coord2):
    return np.sqrt((Coord2[0]-Coord1[0])**2 + (Coord2[1]-Coord1[1])**2)

def Angulo (Coord1, Coord2):
    return np.arctan2(Coord2[1]-Coord1[1], Coord2[0]-Coord1[0]) * 180 / np.pi

# Comprobar_vision es una función que comprueba si entre dos puntos existe 
# visión directa o no. Para ello recibe como datos de entrada:
# - Lista/tupla/array de la Coordenada 1 -> (x,y)
# - Lista/tupla/array de la Coordenada 2 -> (x,y)
# - Matriz de obstáculos: 1 pertenece a la habitación, 0 está fuera.ç
# Devuelve el valor 1 en caso de existir visión directa entre los puntos y 0
# en caso contrario.
def Comprobar_vision(Coord_1, Coord_2, Matriz_global):
    Desp_Fil = np.abs(Coord_1[0]-Coord_2[0])
    Desp_Col = np.abs(Coord_1[1]-Coord_2[1])
    
    Avanzar_Filas = (Coord_2[0] > Coord_1[0])
    # Siempre se avanza hacia la derecha
    if Avanzar_Filas:
        Coord_Ini = Coord_1
        Coord_Fin = Coord_2
    else:
        Coord_Ini = Coord_2
        Coord_Fin = Coord_1
    Avanzar_Columnas = (Coord_Fin[1] > Coord_Ini[1])

    if (Desp_Fil == 0) or (Desp_Col == 0): # Hay visión
        if Desp_Fil == 0: # Están en la misma fila
            aux = np.sum(Matriz_global[Coord_Ini[0],np.min([Coord_Ini[1],Coord_Fin[1]]):np.max([Coord_Ini[1],Coord_Fin[1]])])
            if aux < Desp_Col:
                return 0
        if Desp_Col == 0: # Están en la misma columna
            aux = np.sum(Matriz_global[np.min([Coord_Ini[0],Coord_Fin[0]]):np.max([Coord_Ini[0],Coord_Fin[0]]),Coord_Ini[1]])
            if aux < Desp_Fil:
                return 0
        return 1
    
    # relacion = int(np.round(np.max([Desp_Fil,Desp_Col])/np.min([Desp_Fil,Desp_Col])))
    # relacion = int(np.floor(np.max([Desp_Fil,Desp_Col])/np.min([Desp_Fil,Desp_Col])))
    # relacion = int(np.ceil(np.max([Desp_Fil,Desp_Col])/np.min([Desp_Fil,Desp_Col])))
    aux = np.max([Desp_Fil,Desp_Col])/np.min([Desp_Fil,Desp_Col])
    if aux - np.trunc(aux) == 0.5:
        relacion = int(aux + 0.5)
    else:
        relacion = int(np.round(np.max([Desp_Fil,Desp_Col])/np.min([Desp_Fil,Desp_Col])))

    if Desp_Fil >= Desp_Col:
        Desp_Mayor_Fil = True
        Desp_Mayor_Col = False
    else:
        Desp_Mayor_Fil = False
        Desp_Mayor_Col = True
    
    Ultima_Fila = False
    Ultima_Columna = False
    
    if Desp_Mayor_Fil:
        Desplazamiento = "Fila"
    else:
        Desplazamiento = "Columna"

    pos_siguiente = Coord_Ini
    avance = 0
    for _ in range(Desp_Fil + Desp_Col):
        if Desplazamiento == "Fila" and not(Ultima_Fila):
            pos_siguiente = [pos_siguiente[0]+1,pos_siguiente[1]]
            avance += 1
            if pos_siguiente[0] == Coord_Fin[0]:
                Ultima_Fila = True
            if Desp_Mayor_Fil:
                if (avance == relacion):
                    Desplazamiento = "Columna"
                    avance = 0
            else:
                if avance == 1:
                    Desplazamiento = "Columna"
                    avance = 0
            
            
        elif Desplazamiento == "Columna" and not (Ultima_Columna):
            if Avanzar_Columnas:
                pos_siguiente = [pos_siguiente[0],pos_siguiente[1]+1]
            else:
                pos_siguiente = [pos_siguiente[0],pos_siguiente[1]-1]
            avance += 1
            if pos_siguiente[1] == Coord_Fin[1]:
                Ultima_Columna = True
            if Desp_Mayor_Col:
                if (avance == relacion):
                    Desplazamiento = "Fila"
                    avance = 0
            else:
                if avance == 1:
                    Desplazamiento = "Fila"
                    avance = 0
        if (Matriz_global[pos_siguiente[0],pos_siguiente[1]]) == 0:
            return 0
            
    return 1


# Repareación es una función que repara los errores cometidos en el algoritmo
# de búsqueda de la visión directa.
# Los valores de entrada son:
# - Arra_Vision: Vector de dimensión (Np_azules) donde 0 indica que no hay
#   visión directa y 1 indica que hay visión directa.
# - Coords: Array de dimensiones (Np_azules, 2) que indica las coordenadas de
#   cada uno de los puntos.
# - Matriz_global: Matriz que indica con el valor 1 si el punto pertenece a la 
#   habitación y 0 indica que no.
def Reparacion(Array_Vision, Coords, Matriz_global):
    Pto_Min_x = np.min(Coords[:,0])
    Pto_Min_y = np.min(Coords[:,1])
    Pto_Max_x = np.max(Coords[:,0])
    Pto_Max_y = np.max(Coords[:,1])

    Dist_horizontal = np.unique(Coords[:,0])[1]-np.unique(Coords[:,0])[0]
    Dist_vertical = np.unique(Coords[:,1])[1]-np.unique(Coords[:,1])[0]
    
    Num_Ptos_x = int((Pto_Max_x-Pto_Min_x) / Dist_horizontal + 1)
    Num_Ptos_y = int((Pto_Max_y-Pto_Min_y) / Dist_vertical + 1 )
    
    Matriz_vision = np.zeros((Num_Ptos_x,Num_Ptos_y))
    Indices_vision = np.where(Array_Vision == 1)[0]
    Indices_global = np.where(Matriz_global.flatten() == 1)[0]
    
    Coords_vision = Coords[Indices_vision]
    for pto in Coords_vision:
        Matriz_vision[int(pto[0]/Dist_horizontal),int(pto[1]/Dist_vertical)] = 1

    kernel = 1/9 * np.ones((3,3))
    Matriz_vision = np.round(convolve2d(Matriz_vision, kernel, mode="same", boundary="symm"))
    Array_Vision = Matriz_vision.flatten()[Indices_global]
    return Array_Vision

# Función que devuelve las matrices de distancia y ángulos, ambas con dimensión
# (Np_rojos, Np_azules). Como entrada necesita las coordenadas de los puntos
# rojos (Np_rojos, 2) y las coordenadas de los puntos azules (Np_azules,2)
def CalcularDistanciaAngulosVisionPuntos(CoordPtosDispositivos: np.array,
                                   CoordPtosVigilancia: np.array):

    Pto_Min_x = np.min(CoordPtosVigilancia[:,0])
    Pto_Min_y = np.min(CoordPtosVigilancia[:,1])
    Pto_Max_x = np.max(CoordPtosVigilancia[:,0])
    Pto_Max_y = np.max(CoordPtosVigilancia[:,1])

    Dist_horizontal = np.unique(CoordPtosVigilancia[:,0])[1]-np.unique(CoordPtosVigilancia[:,0])[0]
    Dist_vertical = np.unique(CoordPtosVigilancia[:,1])[1]-np.unique(CoordPtosVigilancia[:,1])[0]

    Num_Ptos_x = int((Pto_Max_x-Pto_Min_x) / Dist_horizontal + 1)
    Num_Ptos_y = int((Pto_Max_y-Pto_Min_y) / Dist_vertical + 1 )


    Matriz_global = np.zeros((Num_Ptos_x,Num_Ptos_y))

    for pto in CoordPtosVigilancia:
        Matriz_global[int(pto[0]/Dist_horizontal),int(pto[1]/Dist_vertical)] = 1

    Matriz_Distancia = np.zeros((CoordPtosDispositivos.shape[0],CoordPtosVigilancia.shape[0]))
    Matriz_Angulo = np.zeros((CoordPtosDispositivos.shape[0],CoordPtosVigilancia.shape[0]))
    Matriz_Vision = np.ones((CoordPtosDispositivos.shape[0],CoordPtosVigilancia.shape[0]))
        
    
    for ind_disp, val_disp in enumerate(CoordPtosDispositivos):
        fil_ini, col_ini = int(val_disp[0]/Dist_horizontal), int(val_disp[1]/Dist_vertical)

        for ind_vig, val_vig in enumerate(CoordPtosVigilancia):
            fil_end, col_end = int(val_vig[0]/Dist_horizontal), int(val_vig[1]/Dist_vertical)

            Matriz_Distancia[ind_disp, ind_vig] = Distancia(val_disp, val_vig)
            aux = Angulo(val_disp, val_vig)
            if aux < 0:
                Matriz_Angulo[ind_disp, ind_vig] = 360 + aux
            else:
                Matriz_Angulo[ind_disp, ind_vig] = aux
                
            Matriz_Vision[ind_disp,ind_vig] = Comprobar_vision([fil_ini, col_ini],[fil_end, col_end], Matriz_global)
            
    # Reparación
    for i, fila in enumerate(Matriz_Vision):
        Matriz_Vision[i,:] = Reparacion(Matriz_Vision[i,:], CoordPtosVigilancia, Matriz_global)
        Matriz_Vision[i,:] = Reparacion(Matriz_Vision[i,:], CoordPtosVigilancia, Matriz_global)
        Matriz_Vision[i,:] = Reparacion(Matriz_Vision[i,:], CoordPtosVigilancia, Matriz_global)
    return Matriz_Distancia, Matriz_Angulo, Matriz_Vision
