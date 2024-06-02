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
import ctypes


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


class GUI:
    def __init__(self):
        # creating tkinter window
        self.window = tk.Tk()
        self.window.title('ML Regressor')
        self.window.geometry('640x720')

        # design parameters
        self.w_bt = 10
        self.w_xb = 2
        self.w_cb = 10
        self.n_padx = 5
        self.n_pady = 2
        self.sticky_just = tk.W

        # initialise lists and objects
        self.lb7to8 = []
        self.cb1to9 = []
        self.var1to5 = []
        self.ax = None
        self.fig = None
        self.toolbar = None
        self.ml = ML()

        # reset app scaling due to dpi change on windows
        if sys.platform == 'win32':
            ctypes.windll.shcore.SetProcessDpiAwareness(0)

    # function to get absolute path to data file
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def create_group(self, txt, column, row, rowspan):
        lf = ttk.Labelframe(self.window, text=txt)
        lf.grid(column=column, row=row, rowspan=rowspan, padx=self.n_padx, pady=self.n_pady)
        return lf

    def create_label(self, parent, txt, column, row, columnspan):
        lb = tk.StringVar()
        lb.set(txt)
        ttk.Label(parent, textvariable=lb).grid(column=column, row=row, columnspan=columnspan, padx=self.n_padx,
                                           pady=self.n_pady, sticky=self.sticky_just)
        return lb

    def create_combobox(self, parent, lst, column, row, columnspan):
        cb = ttk.Combobox(parent, width=self.w_cb, textvariable=tk.StringVar())
        cb['values'] = lst
        cb.grid(column=column, row=row, columnspan=columnspan)
        cb['state'] = 'readonly'
        return cb

    def create_checkbox(self, parent, txt, column, row):
        var = IntVar()
        Checkbutton(parent, text=txt, variable=var).grid(column=column, row=row, padx=self.n_padx, pady=self.n_pady,
                                                         sticky=self.sticky_just)
        return var

    def load_gui(self):
        # Groups
        txt1to3 = ('Data Configuration','Model Configuration','Results Configuration')
        val1to3 = ((0,1,1),(2,1,2),(0,2,1))
        lf1to3 = []
        for text, (column, row, rowspan) in zip(txt1to3, val1to3):
            lf1to3.append(self.create_group(text, column, row, rowspan))

        # Labels
        txt1to6 = ('Points:','Split:','Points:','Inputs:','Output:','Models:')
        val1to6 = ((1,0,3,1),(1,0,4,1),(3,0,7,1),(2,2,3,1),(2,2,8,1),(2,4,3,1))
        lb1to6 = []
        for text, (n_lf, column, row, columnspan) in zip(txt1to6, val1to6):
            lb1to6.append(self.create_label(lf1to3[n_lf-1], text, column, row, columnspan))
        txt7to8 = ('Status: Not Loaded...','Data: Not Ready...')
        val7to8 = ((0,15,2),(2,15,2))
        self.lb7to8 = []
        for text, (column, row, columnspan) in zip(txt7to8, val7to8):
            self.lb7to8.append(self.create_label(self.window, text, column, row, columnspan))

        # ComboBoxes
        list1 = [100, 400, 1000]
        list2 = [0.1, 0.2, 0.3, 0.4, 0.5]
        list3 = [5, 10, 25]
        list4 = []
        list5 = []
        list6 = []
        list7 = []
        list8 = []
        list9 = []
        lst1to9 = (list1, list2, list3, list4, list5, list6, list7, list8, list9)
        val1to9 = ((1,1,3),(1,1,4),(3,1,7),(2,3,3),(2,3,4),(2,3,5),(2,3,6),(2,3,7),(2,3,8))
        self.cb1to9 = []
        for lst, (n_lf, column, row) in zip(lst1to9, val1to9):
            self.cb1to9.append(self.create_combobox(lf1to3[n_lf-1], lst,  column, row, 1))
        self.cb1to9[0]['state'] = 'normal'
        self.cb1to9[1]['state'] = 'normal'
        self.cb1to9[2]['state'] = 'normal'

        # CheckBoxes
        txt1to5 = ("MLR", "PLR", "SVR", "DTR", "RFR")
        val1to5 = ((5,3),(5,4),(5,5),(5,6),(5,7))
        self.var1to5 = []
        for text, (column, row) in zip(txt1to5, val1to5):
            self.var1to5.append(self.create_checkbox(lf1to3[1], text, column, row))

        # Canvas for data plots
        matplotlib.use('TkAgg')
        self.fig, self.ax = plt.subplots(1, 1)
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        plot_widget = canvas.get_tk_widget()
        plot_widget.grid(row=9, column=0, rowspan=6, columnspan=6)
        toolbar_frame = Frame(self.window)
        toolbar_frame.grid(row=15, column=0, columnspan=6)
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)

        # Buttons
        bt1 = ttk.Button(lf1to3[0], text="Load", command=self.load_data, width=self.w_bt).grid(column=1, row=5,
                                                                                               padx=self.n_padx,
                                                                                               pady=self.n_pady,
                                                                                               sticky=self.sticky_just)
        bt2 = ttk.Button(lf1to3[1], text="Train", command=self.train, width=self.w_bt).grid(column=5, row=8,
                                                                                            padx=self.n_padx,
                                                                                            pady=self.n_pady,
                                                                                            sticky=self.sticky_just)
        bt3 = ttk.Button(lf1to3[2], text="Predict", command=self.predict, width=self.w_bt).grid(column=1, row=8,
                                                                                                padx=self.n_padx,
                                                                                                pady=self.n_pady,
                                                                                                sticky=self.sticky_just)

        # Loop window
        self.window.protocol('WM_DELETE_WINDOW', self.app_quit)
        self.window.mainloop()

    def load_data(self):
        filetypes = (('csv files', '*.csv'), ('txt files', '*.txt'), ('all files', '*.*'))
        f = fd.askopenfile(title="Open Data File", filetypes=filetypes, initialdir=self.ml.dirDefault)
        if not f is None:
            try:
                datafilepath = f.name
                print('Data File Loaded: ' + datafilepath)
                self.ml.dirDefault = os.path.dirname(datafilepath)
                self.ml.dirFile = datafilepath
                dataset = pd.read_csv(self.ml.dirFile, header=0)
                print('Number of data points: ' + str(dataset.shape[0]))
                # print(dataset.columns.values)
                self.lb7to8[1].set('Data points: ' + str(dataset.shape[0]) + ' loaded')
                self.ml.numdata = dataset.shape[0]
                self.ml.numsample = self.ml.numdata
                header = list(dataset.columns.values)
                for i in range(len(header)+1):
                    self.cb1to9[i+3]['values'] = [''] + header
                self.ml.loaded = True
                self.lb7to8[0].set('Status: Data Loaded!')
            except Exception:
                messagebox.showerror('Error', 'Check Data File!')
                return

    def get_ml(self):
        xblst = [(self.var1to5[0].get(), 'MLR'), (self.var1to5[1].get(), 'PLR'), (self.var1to5[2].get(), 'SVR'),
                 (self.var1to5[3].get(),
                                                                                                      'DTR'),
                 (self.var1to5[4].get(), 'RFR')]
        reglst = []
        for val, txt in xblst:
            if val == 1:
                reglst.append(txt)
        # print(reglst)
        return reglst

    # train models based on app inputs
    def train(self):
        if not self.ml.loaded:
            messagebox.showerror('Error', 'Data Not Loaded!')
            return
        if self.cb1to9[0].get() != '':
            try:
                if int(self.cb1to9[0].get()) > self.ml.numdata:
                    messagebox.showerror('Error', 'Points must be not exceed ' + str(self.ml.numdata) + '!')
                    return
                else:
                    self.ml.numsample = int(self.cb1to9[0].get())
            except Exception:
                messagebox.showerror('Error', 'Input Error!')
                return
        else:
            messagebox.showerror('Error', 'Enter Points!')
            return
        if self.cb1to9[1].get() != '':
            try:
                if not 0.0 < float(self.cb1to9[1].get()) < 1.0:
                    messagebox.showerror('Error', 'Split must be between 0 and 1!')
                    return
                else:
                    self.ml.split = float(self.cb1to9[1].get())
            except Exception:
                messagebox.showerror('Error', 'Input Error!')
                return
        else:
            messagebox.showerror('Error', 'Enter Split!')
            return
        cblst = [self.cb1to9[3].get(), self.cb1to9[4].get(), self.cb1to9[5].get(), self.cb1to9[6].get(),
                 self.cb1to9[7].get()]
        inputlst = []
        for cb in cblst:
            if cb:
                inputlst.append(cb)
        outputlst = []
        if self.cb1to9[8].get():
            outputlst.append(self.cb1to9[8].get())
        self.ml.inputs = inputlst
        self.ml.output = outputlst
        if self.ml.inputs == []:
            messagebox.showerror('Error', 'Choose Inputs!')
            return
        if self.ml.output == []:
            messagebox.showerror('Error', 'Choose Output!')
            return
        reglst = self.get_ml()
        self.ml.reglst = reglst
        if self.ml.reglst == []:
            messagebox.showerror('Error', 'Choose Model!')
            return
        try:
            t1 = time.time()
            self.ml.train()
            t2 = time.time()
            self.lb7to8[0].set('Status: Trained ~' + str(round((t2 - t1) * 1e3, 1)) + 'ms')
        except Exception:
            messagebox.showerror('Error', 'Training Error!')
            return

    # predict results based on app inputs and plot data
    def predict(self):
        reglst = self.get_ml()
        self.ml.reglst = reglst
        if self.cb1to9[2].get() != '':
            try:
                if int(self.cb1to9[2].get()) > len(self.ml.y_test):
                    messagebox.showerror('Error', 'Points must not exceed ' + str(len(self.ml.y_test)) + '!')
                    return
                else:
                    self.ml.numresult = int(self.cb1to9[2].get())
            except Exception:
                messagebox.showerror('Error', 'Input Error!')
                return
        else:
            messagebox.showerror('Error', 'Enter Points!')
            return
        models = []
        legendlst = []
        result = {}
        t1 = time.time()
        result = self.ml.predict()
        t2 = time.time()
        self.lb7to8[0].set('Status: Predicted ~' + str(round((t2 - t1) * 1e3, 1)) + 'ms')
        models = [*result]
        colorlst = {'MLR': 'b', 'PLR': 'g', 'SVR': 'r', 'DTR': 'm', 'RFR': 'y'}
        markerlst = {'MLR': 'P', 'PLR': 'o', 'SVR': 's', 'DTR': '^', 'RFR': '*'}
        self.ax.cla()
        units = {'Dens': 'g/L', 'Visco': 'cSt', 'Temp': "\N{DEGREE SIGN}C"}
        limits = {'Dens': [750, 950], 'Visco': [0, 1000], 'Temp': [20, 150]}
        output = self.cb1to9[8].get()
        linedata = np.arange(limits[output][0], limits[output][1])
        self.ax.plot(linedata, linedata, 'k--', linewidth=1, label='_nolegend_')
        self.ax.set_ylim(limits[output])
        self.ax.set_xlim(limits[output])
        self.ax.set_xlabel("Real," + units[output])
        self.ax.set_ylabel("Predicted," + units[output])
        for model in models:
            self.ax.scatter(result[model][2][:, 1], result[model][2][:, 0], alpha=0.8, c=colorlst[model],
                       marker=markerlst[model])
            legendlst.append(
                model + ' \u03C3=' + str(round(result[model][0], 2)) + '(' + str(round(result[model][1], 2)) + '%) ~' +
                result[model][3])
        self.ax.legend(legendlst)
        self.toolbar.update()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def app_quit(self):
        response = messagebox.askyesno('Exit', 'Do you want to close app?')
        if response:
            self.window.quit()
            self.window.destroy()


if __name__ == '__main__':
    gui = GUI()
    gui.load_gui()
