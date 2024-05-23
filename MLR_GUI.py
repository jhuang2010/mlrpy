#!/usr/bin/python
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as fd
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import numpy as np
import pandas as pd
import random
import time
import joblib

class ML(object):
    def __init__(self):
        if getattr(sys, 'frozen', False):
            self.dirDefault = os.path.dirname(sys.executable)
        else:
            self.dirDefault = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        self.dirFile = self.dirDefault+os.sep+'Dataset.csv' # directory to data file
        self.reglst = ['MLR','PLR','SVR','DTR','RFR'] # machine learning regression models
        self.split = 0.2 # testing and training dataset split, e.g. 0.2 means 20% for testing and 80% for training
        self.inputs = ['Vel','Atten fft'] # input parameters
        self.output = ['Dens'] # output parameters
        self.numdata = 1000 # number of data in total
        self.numsample = 1000 # number of data to be used as samples
        self.numresult = 5 # number of results to predict
        self.mlr = []
        self.plr = []
        self.svr = []
        self.dtr = []
        self.rfr = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.poly_reg = []
        self.sc_X = []
        self.sc_y = []
        self.loaded = False
    
    # get the size of a file with unit
    def getfilesize(self, pathfile):
        filesize = os.path.getsize(pathfile)
        unit = 'B'
        if not filesize < 1024:
            filesize = filesize/1024
            unit = 'KB'
            if not filesize < 1024:
                filesize = filesize/1024
                unit = 'MB'
                if not filesize < 1024:
                    filesize = filesize/1024
                    unit = 'GB'
        return str(int(filesize))+unit
    
    # train machine learning regression model with data and deploy model
    def regression(self, X_train, y_train, regressor):
        if regressor == 'MLR':
            ## Multi Linear Regression
            # Training the Multiple Linear Regression model on the Training set
            from sklearn.linear_model import LinearRegression
            self.mlr = LinearRegression()
            self.mlr.fit(X_train, y_train)
            filefmt = '.gz'
            dumpfile = self.dirDefault+os.sep+regressor+filefmt
            joblib.dump(self.mlr,dumpfile)
            print('Multiple Linear Regression - Trained ~'+self.getfilesize(dumpfile))

        if regressor == 'PLR':
            ## Polynomial Regression
            # Training the Polynomial Regression model on the whole dataset
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            self.poly_reg = PolynomialFeatures(degree = 2)
            X_poly = self.poly_reg.fit_transform(X_train)
            self.plr = LinearRegression()
            self.plr.fit(X_poly, y_train)
            filefmt = '.gz'
            dumpfile = self.dirDefault+os.sep+regressor+filefmt
            joblib.dump(self.plr,dumpfile)
            dumpfile = self.dirDefault+os.sep+regressor+'_X'+filefmt
            joblib.dump(self.poly_reg,dumpfile)
            print('Polynomial Linear Regression - Trained ~'+self.getfilesize(dumpfile))
        
        if regressor == 'SVR':
            ## Support Vector Regression
            from sklearn.preprocessing import StandardScaler
            self.sc_X = StandardScaler()
            self.sc_y = StandardScaler()
            X_train = self.sc_X.fit_transform(X_train)
            y_train = self.sc_y.fit_transform(y_train)
            # print(X)
            # print(y)
            # Training the SVR model on the whole dataset
            from sklearn.svm import SVR
            self.svr = SVR(kernel = 'rbf')
            y_train = np.ravel(y_train)
            self.svr.fit(X_train, y_train)
            filefmt = '.gz'
            dumpfile = self.dirDefault+os.sep+regressor+filefmt
            joblib.dump(self.svr,dumpfile)
            dumpfile = self.dirDefault+os.sep+regressor+'_X'+filefmt
            joblib.dump(self.sc_X,dumpfile)
            dumpfile = self.dirDefault+os.sep+regressor+'_y'+filefmt
            joblib.dump(self.sc_y,dumpfile)
            print('Support Vector Regression - Trained ~'+self.getfilesize(dumpfile))
        
        if regressor == 'DTR':
            ## Decision Tree
            from sklearn.tree import DecisionTreeRegressor
            self.dtr = DecisionTreeRegressor(random_state = 42)
            self.dtr.fit(X_train, y_train)
            filefmt = '.gz'
            dumpfile = self.dirDefault+os.sep+regressor+filefmt
            joblib.dump(self.dtr,dumpfile)
            print('Decision Tree Regression - Trained ~'+self.getfilesize(dumpfile))
        
        if regressor == 'RFR':
            ## Random Forest
            from sklearn.ensemble import RandomForestRegressor
            self.rfr = RandomForestRegressor(n_estimators = 10, random_state = 42)
            y_train = np.ravel(y_train)
            self.rfr.fit(X_train, y_train)
            filefmt = '.gz'
            dumpfile = self.dirDefault+os.sep+regressor+filefmt
            joblib.dump(self.rfr,dumpfile)
            print('Random Forest Regression - Trained ~'+self.getfilesize(dumpfile))

    # test machine learning model with data for accuracy
    def test(self, X_test, y_test, regressor, index_result):
        if regressor == 'MLR':
            print('Multiple Linear Regression')
            y_pred = self.mlr.predict(X_test)
        
        if regressor == 'PLR':
            print('Polynomial Linear Regression')
            # from sklearn.preprocessing import PolynomialFeatures
            # poly_reg = PolynomialFeatures(degree = 2)
            X_test = self.poly_reg.transform(X_test)
            y_pred = self.plr.predict(X_test)
            # y_pred = self.plr.predict(self.poly_reg.transform(X_test))
        
        if regressor == 'SVR':
            print('Support Vector Regression')
            # from sklearn.preprocessing import StandardScaler
            # sc_X = StandardScaler()
            # sc_y = StandardScaler()
            X_test = self.sc_X.transform(X_test)
            y_test = self.sc_y.transform(y_test)
            # Predicting a new result
            y_pred = self.svr.predict(X_test)
            y_pred = y_pred.reshape((len(y_pred),1))
            y_pred = self.sc_y.inverse_transform(y_pred)
            # print(y_pred)
            y_test = y_test.reshape((len(y_test),1))
            y_test = self.sc_y.inverse_transform(y_test)
            # print(y_test)
        
        if regressor == 'DTR':
            print('Decision Tree Regression')
            y_pred = self.dtr.predict(X_test)
            y_pred = y_pred.reshape(len(y_pred),1)
        
        if regressor == 'RFR':
            print('Random Forest Regression')
            y_pred = self.rfr.predict(X_test)
            y_pred = y_pred.reshape(len(y_pred),1)
            
        if len(index_result)>0:
            sigma_y = np.mean(np.abs(y_pred-y_test))
            sigma_perc_y = np.mean(np.abs(y_pred-y_test)/y_test*100)
            y_matrix = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
            np.set_printoptions(precision=2)
            print(y_matrix[index_result])
            print(sigma_y)
            print(sigma_perc_y)
            dumpfile = self.dirDefault+os.sep+regressor+'.gz'
            regsize = self.getfilesize(dumpfile)
        
        return sigma_y, sigma_perc_y, y_matrix[index_result], regsize

    # train multiple lm models 
    def train(self):
        # Read the data and randomly select samples
        # dataset = pd.read_csv(self.dirFile,header = 0,usecols=[0,1,2,3,4,5,6,7])[::int(np.floor(self.numdata/self.numsample))]
        # print(dataset.shape[0])
        # X = dataset.loc[:self.numdata, self.inputs].values
        # y = dataset.loc[:self.numdata, self.output].values
        dataset = pd.read_csv(self.dirFile,header = 0)
        rowsample = random.sample(list(dataset.index),self.numsample)
        X = dataset.loc[rowsample, self.inputs].values
        y = dataset.loc[rowsample, self.output].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.split, random_state = 42)
        # print(dataset.columns.values)
        # print(len(X))
        print(self.inputs)
        # print(X)
        for reg in self.reglst:
            self.regression(self.X_train, self.y_train, regressor=reg)
    
    # predict results with multiple ml models
    def predict(self):
        result_vec = []
        index_result = []
        val_result = []
        sigma_y = []
        sigma_perc_y = []
        y_vector = []
        regsize = []
        result_vec = random.sample(list(enumerate(self.y_test)),self.numresult)
        for indx, val in result_vec:
            index_result.append(indx)
            val_result.append(val)
        result_dic = {}
        for reg in self.reglst:
            sigma_y, sigma_perc_y, y_vector, regsize = self.test(self.X_test, self.y_test, regressor=reg, index_result = index_result)
            result_dic.update({reg:[sigma_y,sigma_perc_y,y_vector,regsize]})
        # print(result_dic)
        return result_dic
    
# function to get absolute path to data file
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# creating tkinter window
window = tk.Tk()
window.title('ML Regressor')
window.geometry('640x720')

# design parameters
w_cb = 8
w_cl = 25
w_lb = 8
w_bt = 10
w_xb = 2
n_padx = 10
n_pady_cl = 5
n_pady = 2
lb_just = tk.W

# label groups
lf1 = ttk.Labelframe(window,text='Data Configuration')
lf1.grid(column = 0, row = 1, padx = n_padx, pady=n_pady_cl)
lf2 = ttk.LabelFrame(window,text='Model Configuration')
lf2.grid(column = 2, row = 1, rowspan = 2, padx = n_padx, pady=n_pady_cl)
lf3 = ttk.LabelFrame(window,text='Results Configuration')
lf3.grid(column = 0, row = 2, padx = n_padx, pady=n_pady_cl)

# labels
lb1 = ('Data Configuration')
lb2 = ('Points:')
lb3 = ('Split:')
lb4 = ('Results Display')
lb5 = ('Model Configuration')
lb6 = ('Inputs:')
lb7 = ('Output:')
lb8 = ('Models:')
ttk.Label(lf1, text = lb2, width = w_lb).grid(column = 0,row = 3, padx = n_padx, pady = n_pady, sticky = lb_just)
ttk.Label(lf1, text = lb3, width = w_lb).grid(column = 0,row = 4, padx = n_padx, pady = n_pady, sticky = lb_just)
ttk.Label(lf3, text = lb2, width = w_lb).grid(column = 0,row = 7, padx = n_padx, pady = n_pady, sticky = lb_just)
ttk.Label(lf2, text = lb6, width = w_lb).grid(column = 2,row = 3, padx = n_padx, pady = n_pady, sticky = lb_just)
ttk.Label(lf2, text = lb7, width = w_lb).grid(column = 2,row = 8, padx = n_padx, pady = n_pady, sticky = lb_just)
ttk.Label(lf2, text = lb8, width = w_lb).grid(column = 4,row = 3, padx = n_padx, pady = n_pady, sticky = lb_just)
lb14 = StringVar()
lb14.set('Status: Not Loaded...')
ttk.Label(window, textvariable = lb14).grid(column = 0,row = 15, columnspan=2, padx = n_padx, pady = n_pady, sticky = lb_just)
lb15 = StringVar()
lb15.set('Data points: Not ready...')
ttk.Label(window, textvariable = lb15).grid(column = 2,row = 15, padx = n_padx, pady = n_pady, sticky = tk.E)

# comboboxes
list1 = [100,400,1000]
list2 = [0.1,0.2,0.3,0.4,0.5]
list3 = [5,10,25]
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []
cb1 = ttk.Combobox(lf1, width = w_cb, textvariable = tk.StringVar())
cb1['values'] = (list1)
cb1.grid(column = 1, row = 3, padx = n_padx, pady = n_pady, sticky = lb_just)
cb2 = ttk.Combobox(lf1, width = w_cb, textvariable = tk.StringVar())
cb2['values'] = (list2)
cb2.grid(column = 1, row = 4, padx = n_padx, pady = n_pady, sticky = lb_just)
cb3 = ttk.Combobox(lf3, width = w_cb, textvariable = tk.StringVar())
cb3['values'] = (list3)
cb3.grid(column = 1, row = 7, padx = n_padx, pady = n_pady, sticky = lb_just)
cb4 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb4['values'] = (list4)
cb4.grid(column = 3, row = 3, padx = n_padx, pady = n_pady, sticky = lb_just)
cb5 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb5['values'] = (list5)
cb5.grid(column = 3, row = 4, padx = n_padx, pady = n_pady, sticky = lb_just)
cb6 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb6['values'] = (list6)
cb6.grid(column = 3, row = 5, padx = n_padx, pady = n_pady, sticky = lb_just)
cb7 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb7['values'] = (list7)
cb7.grid(column = 3, row = 6, padx = n_padx, pady = n_pady, sticky = lb_just)
cb8 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb8['values'] = (list8)
cb8.grid(column = 3, row = 7, padx = n_padx, pady = n_pady, sticky = lb_just)
cb9 = ttk.Combobox(lf2, width = w_cb, textvariable = tk.StringVar())
cb9['values'] = (list9)
cb9.grid(column = 3, row = 8, padx = n_padx, pady = n_pady, sticky = lb_just)
# cb1['state']='readonly'
# cb2['state']='readonly'
# cb3['state']='readonly'
cb4['state']='readonly'
cb5['state']='readonly'
cb6['state']='readonly'
cb7['state']='readonly'
cb8['state']='readonly'
cb9['state']='readonly'

# check boxes
var1 = IntVar()
xb1 = Checkbutton(lf2, text="MLR", variable=var1).grid(column=5, row=3, padx = n_padx, pady = n_pady, sticky = lb_just)
var2 = IntVar()
xb2 = Checkbutton(lf2, text="PLR", variable=var2).grid(column=5, row=4, padx = n_padx, pady = n_pady, sticky = lb_just)
var3 = IntVar()
xb3 = Checkbutton(lf2, text="SVR", variable=var3).grid(column=5, row=5, padx = n_padx, pady = n_pady, sticky = lb_just)
var4 = IntVar()
xb4 = Checkbutton(lf2, text="DTR", variable=var4).grid(column=5, row=6, padx = n_padx, pady = n_pady, sticky = lb_just)
var5 = IntVar()
xb5 = Checkbutton(lf2, text="RFR", variable=var5).grid(column=5, row=7, padx = n_padx, pady = n_pady, sticky = lb_just)

# canvas for data plots
matplotlib.use('TkAgg')
fig, ax = plt.subplots(1,1) 
canvas = FigureCanvasTkAgg(fig, master=window)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=9, column=0, rowspan = 6,columnspan = 5)
toolbar_frame = Frame(window)
toolbar_frame.grid(row=15, column=0, columnspan = 5)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)

ml = ML()

# function to terminate app
def app_quit():
    response=messagebox.askyesno('Exit','Do you want to close app?')
    if response:
        # is_stop.set()
        # sleep(2)
        window.quit()
        window.destroy()

# load data to app
def load():
    filetypes = (('csv files','*.csv'),('txt files','*.txt'),('all files','*.*'))
    f = fd.askopenfile(title="Open Data File",filetypes = filetypes,initialdir = ml.dirDefault)
    if not f is None:
        try: 
            datafilepath = f.name
            print('Data File Loaded: '+datafilepath)
            ml.dirDefault = os.path.dirname(datafilepath)
            ml.dirFile = datafilepath
            dataset = pd.read_csv(ml.dirFile,header = 0)
            print('Number of data points: '+str(dataset.shape[0]))
            # print(dataset.columns.values)
            lb15.set('Data points: '+str(dataset.shape[0])+' loaded')
            ml.numdata = dataset.shape[0]
            ml.numsample = ml.numdata
            header = list(dataset.columns.values)
            cb4['values'] = ['']+header
            cb5['values'] = ['']+header
            cb6['values'] = ['']+header
            cb7['values'] = ['']+header
            cb8['values'] = ['']+header
            cb9['values'] = ['']+header
            ml.loaded = True
            lb14.set('Status: Data Loaded!')
        except Exception:
            messagebox.showerror('Error','Check Data File!')
            return   

# train models based on app inputs
def train():
    if not ml.loaded:
        messagebox.showerror('Error','Data Not Loaded!')
        return       
    if cb1.get() != '':
        try: 
            if int(cb1.get()) > ml.numdata:
                messagebox.showerror('Error','Points must be not exceed '+str(ml.numdata)+'!')
                return
            else:
                ml.numsample = int(cb1.get())
        except Exception:
            messagebox.showerror('Error','Input Error!')
            return
    else:
        messagebox.showerror('Error','Enter Points!')
        return
    if cb2.get() != '':
        try: 
            if not 0.0 < float(cb2.get()) < 1.0:
                messagebox.showerror('Error','Split must be between 0 and 1!')
                return
            else:
                ml.split = float(cb2.get())
        except Exception:
            messagebox.showerror('Error','Input Error!')
            return            
    else:
        messagebox.showerror('Error','Enter Split!')
        return
    cblst = [cb4.get(),cb5.get(),cb6.get(),cb7.get(),cb8.get()]
    inlst = []
    for cb in cblst:
        if cb:
            inlst.append(cb)
    outlst = []
    if cb9.get():
        outlst.append(cb9.get())
    ml.inputs = inlst
    ml.output = outlst
    if ml.inputs == []:
        messagebox.showerror('Error','Choose Inputs!')
        return
    if ml.output == []:
        messagebox.showerror('Error','Choose Output!')
        return
    xblst = [(var1.get(),'MLR'),(var2.get(),'PLR'),(var3.get(),'SVR'),(var4.get(),'DTR'),(var5.get(),'RFR')]
    reglst = []
    for val, txt in xblst:
        if val == 1:
            reglst.append(txt)
    # print(reglst)
    ml.reglst = reglst
    if ml.reglst == []:
        messagebox.showerror('Error','Choose Model!')
        return
    try: 
        t1 = time.time()
        ml.train()
        t2 = time.time()
        lb14.set('Status: Trained ~'+str(round((t2-t1)*1e3,1))+'ms')
    except Exception:
        messagebox.showerror('Error','Training Error!')
        return   

# predict results based on app inputs and plot data
def predict():
    xblst = [(var1.get(),'MLR'),(var2.get(),'PLR'),(var3.get(),'SVR'),(var4.get(),'DTR'),(var5.get(),'RFR')]
    reglst = []
    for val, txt in xblst:
        if val == 1:
            reglst.append(txt)
    # print(reglst)
    ml.reglst = reglst
    if cb3.get() != '':
        try: 
            if int(cb3.get()) > len(ml.y_test):
                messagebox.showerror('Error','Points must not exceed '+str(len(ml.y_test))+'!')
                return
            else:
                ml.numresult = int(cb3.get())
        except Exception:
            messagebox.showerror('Error','Input Error!')
            return                
    else: 
        messagebox.showerror('Error','Enter Points!')
        return
    models = []
    legendlst = []
    result = {}
    t1 = time.time()
    result = ml.predict()
    t2 = time.time()
    lb14.set('Status: Predicted ~'+str(round((t2-t1)*1e3,1))+'ms')
    models = [*result]
    colorlst = {'MLR':'b','PLR':'g','SVR':'r','DTR':'m','RFR':'y'}
    markerlst = {'MLR':'P','PLR':'o','SVR':'s','DTR':'^','RFR':'*'}
    ax.cla()
    units = {'Dens':'g/L','Visco':'cSt','Temp':"\N{DEGREE SIGN}C"}
    limits = {'Dens':[750,950],'Visco':[0,1000],'Temp':[20,150]}
    output = cb9.get()
    linedata = np.arange(limits[output][0],limits[output][1])
    ax.plot(linedata, linedata, 'k--', linewidth=1, label='_nolegend_')
    ax.set_ylim(limits[output])
    ax.set_xlim(limits[output])
    ax.set_xlabel("Real,"+units[output])
    ax.set_ylabel("Predicted,"+units[output])
    for model in models:
        ax.scatter(result[model][2][:,1],result[model][2][:,0],alpha=0.8,c = colorlst[model],marker=markerlst[model])
        legendlst.append(model+' \u03C3='+str(round(result[model][0],2))+'('+str(round(result[model][1],2))+'%) ~'+result[model][3])
    ax.legend(legendlst)
    toolbar.update()
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

# buttons and main window loop
bt1 = ttk.Button(lf1,text="Load",command = load, width = w_bt).grid(column = 1,row = 5, padx = n_padx, pady = n_pady, sticky = lb_just)
bt2 = ttk.Button(lf2,text="Train",command = train, width = w_bt).grid(column = 5,row = 8, padx = n_padx, pady = n_pady, sticky = lb_just)
bt3 = ttk.Button(lf3,text="Predict",command = predict, width = w_bt).grid(column = 1,row = 8, padx = n_padx, pady = n_pady, sticky = lb_just)
window.protocol('WM_DELETE_WINDOW', app_quit)
window.mainloop()