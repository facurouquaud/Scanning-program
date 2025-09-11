# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:25:44 2025

@author: Luis1
"""

"""
Rutina para controlar el voltaje que se entrega al PMT para comenzar a medir.
El voltaje tipico de medicion es 0.3V.

La placa programable que entrega el voltaje es una LabJack U12, desde su puerto AO0.
Implementar el codigo desde el directorio de trabajo: C:/Users/mdborde/Desktop/uso_LabJack(PMT-808)

El voltaje del LabJack esta montado sobre un offset de 0.04 V aprox. 
Para proteger el PMT se impone como maximo el valor 1.2V.

"""

import tkinter as tk

# Activa la comunicación con el LabJack
import u12 #Rutina con todas las funciones del LabJack (en C:/Users/mdborde/Desktop/uso_LabJack(PMT-808))
d = u12.U12() 
  
def select():  
   sel = "Voltaje PMT = " + str(v.get())  
   label.config(text = sel)  
   n = float(str(v.get()))
   print('Voltaje PMT = ', n ,'V')
   d.eAnalogOut(n,0)
     
top = tk.Tk()
top.title('Controlador voltaje PMT')  
top.geometry("250x150")   

msg = tk.Label(top, text='Seleccione el voltaje necesario')
msg.grid(row=0)

msg2 = tk.Label(top, text='Rango de trabajo PMT: 0 - 1.25 V. Usar en 0.3 V')
msg2.grid(row=1)

v = tk.DoubleVar()  
scale = tk.Scale(top, variable = v, from_ = 0, to = 1.20, digits = 4, resolution = 0.001, orient = tk.HORIZONTAL)  
scale.grid(row=3)  
  
btn = tk.Button(top, text="OK", command=select)  
btn.grid(row=4)  
  
label = tk.Label(top)  
label.grid(row=5) 


  
top.mainloop()  