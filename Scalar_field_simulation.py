# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 22:53:55 2021

@author: Tobias Wolflehner
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate

# General Information:
# This algorithm simulates scalar fields with a quadratic interaction term.
# The theoretic basics can be found in the thesis "Untersuchung des Einflusses 
# des Koordinatensystems in einer numerischen Simulation reeller Skalarfelder 
# in Lichtkegel- und Raum-Zeit Koordinaten". The sources of this thesis can
# also be referenced, since this publication is written in German.

# How to use:
# There is no UI for this simulation. Parameters have are currently hard coded
# and included as global variables. How to set these parameters is explained below.
# Install the needed libraries such as numpy, matplotlib and also the modified scipy
# library from "https://github.com/TobiasWolflehner/scipy". If the official scipy
# library is used, slight deviations occur.

# First the parameters of the simulation are chosen; two coordinatesystems are
# compared, lightcone coordinates and normal time-space Minkowski coordinates.
# These parameters include the lenght of the arrays, which governs with the
# Broadening parameter the discreteness of the lattice. This Broadening governs
# the broadness of the Gaussian, which is used as a starting condition for the 
# simulation. x0, x1 refers to normal coordinates, while p and n refers to x+
# and x- in lightcone coordinates.

# For ease of use, modify only Lenght_x0 (this needs to be uneven, this way the 
# origin is always included in the lattice), BroadeningQuotient as well as the 
# InteractionQuotient to desired values. If unclear, refer to the source thesis, 
# the additional parameters serve to fine tune the algorithm, but do not usually 
# have to be modified so long as the code is working.

# The standard output of this simulation is a graph of the difference of the two 
# simulations in lightcone and normal coordinates. A heatmap or a graph as 2D
# slices is available and the mode can be chosen with the Graph_as_Heatmap 
# parameter set to 0 or 1. A graph of the scalar field in light-cone coordinates
# on its own can be obtained by setting the parameter PlotDifference to 0.

"Size of array in normal (minkowski) coordinates in x0 and x1 direction"
Lenght_x0=201
Lenght_x1=2*Lenght_x0+1

"Size of array in lightcone coordinates in x+ (marked as p) direction"
Lenght_p=int(np.ceil(np.sqrt(2)*Lenght_x0))+20
if Lenght_p%2 == 0 :
    Lenght_p=Lenght_p-1
Lenght_m=2*Lenght_p

PictureSize=20

"Broadness of Gaussian that is set as initial condition"
BroadeningQuotient=10
Broadening=(Lenght_x0-1)/BroadeningQuotient

"Setting of the interaction constant, in relation to lenght of arrays"
InteractionQuotient=0.02
Interaction_constant=(100/(Lenght_x0-1))*(100/(Lenght_x0-1))*InteractionQuotient

"Choose the method of graphing"
Graph_as_Heatmap=1
Graph_as_2D_slices=not(Graph_as_Heatmap)
PlotDifference=1

def initialize_scalar_field_LightCone(lenght_p,lenght_m,broadening):
    "Initializes the arrays for the simulation in lightcone coordinates"
    broadening_Lightcone=broadening/np.sqrt(2)
    scalar_field1 = np.zeros((lenght_p,lenght_m))
    scalar_field2 = np.zeros((lenght_p,lenght_m))
    xp= np.arange(0,lenght_p)
    xm= np.arange(0,lenght_m)
    xp_start=int((lenght_p-1)/2)
    xm_start=int((lenght_p-1)/2)
    
    "Setting of initial conditions"
    scalar_field1[0,:]=np.exp(-np.multiply((xm-xm_start)/broadening_Lightcone,(xm-xm_start)/broadening_Lightcone))
    scalar_field1[1,:]=np.exp(-np.multiply((xm-xm_start)/broadening_Lightcone,(xm-xm_start)/broadening_Lightcone))
        
    scalar_field2[:,0]=np.exp(-np.multiply((xp-xp_start)/broadening_Lightcone,(xp-xp_start)/broadening_Lightcone))
    scalar_field2[:,1]=np.exp(-np.multiply((xp-xp_start)/broadening_Lightcone,(xp-xp_start)/broadening_Lightcone))
    
    return(scalar_field1,scalar_field2)

def initialize_scalar_field_Normal_Coordinates(lenght_x0,lenght_x1,broadening):
    "Initializes the arrays for the simulation in normal (minkowski) coordinates"
    scalar_field3 = np.zeros((lenght_x0,lenght_x1))
    scalar_field4 = np.zeros((lenght_x0,lenght_x1))
    x= np.arange(0,lenght_x1)
    start_val_field3=int((lenght_x1-1)/2)-int((lenght_x0-1)/2)
    start_val_field4=int((lenght_x1-1)/2)+int((lenght_x0-1)/2)
    
    "Setting of initial conditions"
    scalar_field3[0,:]=np.exp(-np.multiply((x-start_val_field3)/broadening,(x-start_val_field3)/broadening))
    scalar_field3[1,:]=np.exp(-np.multiply((x-start_val_field3-1)/broadening,(x-start_val_field3-1)/broadening))
    scalar_field4[0,:]=np.exp(-np.multiply((x-start_val_field4)/broadening,(x-start_val_field4)/broadening))
    scalar_field4[1,:]=np.exp(-np.multiply((x-start_val_field4+1)/broadening,(x-start_val_field4+1)/broadening))
    
    return(scalar_field3,scalar_field4)

def calculate_simulation_Lightcone(scalar_field1,scalar_field2,lenght_p,lenght_m,interaction_constant):
    
    for l in range(lenght_p-2):
        i=l+2
        for k in range(lenght_m-2):
            j=k+1
            try:
                scalar_field1[i][j]=2*scalar_field1[i-1][j]+scalar_field1[i-2][j+1]+scalar_field1[i][j-1]-scalar_field1[i-1][j+1]-scalar_field1[i-2][j]-scalar_field1[i-1][j-1]-interaction_constant*scalar_field1[i-1][j]*scalar_field2[i-1][j]*scalar_field2[i-1][j]
                scalar_field2[i][j]=2*scalar_field2[i-1][j]+scalar_field2[i-2][j+1]+scalar_field2[i][j-1]-scalar_field2[i-1][j+1]-scalar_field2[i-2][j]-scalar_field2[i-1][j-1]-interaction_constant*scalar_field1[i-1][j]*scalar_field1[i-1][j]*scalar_field2[i-1][j]
            except:
                break
            
    "Cutting away unneeded parts of the array (used to eliminate influence of boundary conditions)"
    resized_scalar_field = np.zeros((lenght_p,lenght_p))
    for i in range(lenght_p):
        for j in range(lenght_p):
            resized_scalar_field[i][j]=scalar_field1[i][j]+scalar_field2[i][j]
    return(resized_scalar_field)
    
def calculate_simulation_Normal_Coordinates(scalar_field3,scalar_field4,lenght_x0,lenght_x1,interaction_constant):

    for l in range(np.size(scalar_field3,0)-2):
        i=l+2
        for k in range(np.size(scalar_field3,1)-2):
            j=k+1
            try:
                scalar_field3[i][j]=scalar_field3[i-1][j-1]+scalar_field3[i-1][j+1]-scalar_field3[i-2][j]-interaction_constant*scalar_field3[i-1][j]*scalar_field4[i-1][j]*scalar_field4[i-1][j]
                scalar_field4[i][j]=scalar_field4[i-1][j-1]+scalar_field4[i-1][j+1]-scalar_field4[i-2][j]-interaction_constant*scalar_field4[i-1][j]*scalar_field3[i-1][j]*scalar_field3[i-1][j]
            except:
                break

    "Cutting away unneeded parts of the array (used to eliminate influence of boundary conditions)"
    resized_scalar_field = np.zeros((lenght_x0,lenght_x0))
    for i in range(lenght_x0):
        for j in range(lenght_x0):
            resized_scalar_field[i][j]=scalar_field3[i][j+int((lenght_x1-1)/2)-int((lenght_x0-1)/2)]+scalar_field4[i][j+int((lenght_x1-1)/2)-int((lenght_x0-1)/2)]
        
    return(resized_scalar_field)

def transform_scalarfield(scalar_field):
    
    "Rotate by 45° and transpose"
    re_transformed_scalar_field=rotate(scalar_field,45,axes=(1, 0))
    re_transformed_scalar_field=np.transpose(re_transformed_scalar_field)
    
    return(re_transformed_scalar_field)

def resize_scalarfield(scalar_field,target_lenght,origin):
    resized_scalar_field = np.zeros((target_lenght,target_lenght))
    if np.size(scalar_field,0) % 2 == 0:
        start_val=int(origin-(target_lenght-1)/2)
    else:
        start_val=int(origin-(target_lenght-1)/2)
    
    for i in range(target_lenght):
        for j in range(target_lenght):
            resized_scalar_field[i][j]=scalar_field[start_val+i][start_val+j]
        
    return(resized_scalar_field)

def draw_differencemap_imshow(scalar_field1,scalar_field2):
    "Draws the difference between two scalarfields as a heatmap"
    scalar_field_difference=scalar_field1-scalar_field2
    
    "Creating a custom colormap"
    colors1 = plt.cm.Reds(np.linspace(0., 1, 128))
    colors2 = plt.cm.Blues(np.flip(np.linspace(0, 1, 128)))
    colors = np.vstack((colors2, colors1))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    "Configuration of color values for colormap"
    z_min, z_max = -np.abs(scalar_field_difference).max(), np.abs(scalar_field_difference).max()
    h=plt.ylabel('$x^0$')
    plt.xlabel('$x^1$')
    h.set_rotation(0)
    plt.title('')
    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 17
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    "Drawing the difference map"
    plt.imshow(scalar_field_difference, cmap =mymap, vmin = z_min, vmax = z_max,origin='lower')
    plt.colorbar() 
    plt.show()
    print(z_max)

def draw_differencemap_as2D_slices(scalar_field1, scalar_field2):
    
    difference=scalar_field1-scalar_field2
    
    "Choosing the position of the slices"
    x0_1=int((np.size(difference,0)-1)/4)
    x0_2=int((np.size(difference,0)-1)/2)
    x0_3=int((np.size(difference,0)-1)*3/4)
    
    fig= plt.figure()
    axes=fig.subplots()
    x= np.arange(0,np.size(difference,0))
    axes.set_xlim([0,np.size(difference,0)-1])
    axes.set_ylim([-np.abs(difference).max(),np.abs(difference).max()])
    h = plt.ylabel('Δφ')
    plt.xlabel('$x^1$')
    plt.title('')
    h.set_rotation(0)
    SMALL_SIZE = 13
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 17
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    Magnification=100  # controls the magnification for first part of the plot
    plt.plot(x,Magnification*difference[x0_1,:], label=str(Magnification)+"x Δφ, $x^0$="+str(x0_1))
    plt.plot(x,difference[x0_2,:], label="$x^0$="+str(x0_2))
    plt.plot(x,difference[x0_3,:], label="$x^0$="+str(x0_3))
    plt.legend(loc="lower right")  
    
    plt.show()

"Simulate a scalarfield in Lightcone coordinates"
Scalar_field1,Scalar_field2=initialize_scalar_field_LightCone(Lenght_p,Lenght_m,Broadening)
Scalar_field_Lightcone=calculate_simulation_Lightcone(Scalar_field1,Scalar_field2,Lenght_p,Lenght_m,Interaction_constant)
Origin=int((np.size(Scalar_field_Lightcone,0)+1)/2)
Scalar_field_Lightcone=transform_scalarfield(Scalar_field_Lightcone)
Scalar_field_Lightcone=resize_scalarfield(Scalar_field_Lightcone,Lenght_x0,Origin)

"Simulate a scalarfield in normal (minkoswki) coordinates"
scalar_field3,scalar_field4=initialize_scalar_field_Normal_Coordinates(Lenght_x0,Lenght_x1,Broadening)
Scalar_field_Normal=calculate_simulation_Normal_Coordinates(scalar_field3,scalar_field4,Lenght_x0,Lenght_x1,Interaction_constant)

"Draw a difference-map as the subtraction of the two simulated fields or a single scalar field"
if Graph_as_Heatmap == 1:
    if PlotDifference == 1:
        draw_differencemap_imshow(Scalar_field_Lightcone,Scalar_field_Normal)
    else:
        draw_differencemap_imshow(Scalar_field_Lightcone,0)
else:
    if PlotDifference == 1:
        draw_differencemap_as2D_slices(Scalar_field_Lightcone,Scalar_field_Normal)
    else:
        draw_differencemap_as2D_slices(Scalar_field_Lightcone,0)
