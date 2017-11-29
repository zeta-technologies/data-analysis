# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:10:52 2017

@author: Robin
"""

import random
import binascii
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import csv
import glob
import os
import numpy as np
import pandas
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#from cleaning_data import *
import math

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_median(x, N) :
    df = pd.DataFrame({'col':x})
    roll = pd.rolling_median(df, N)
    return roll

def filtering_and_showing(si, lowH, highH):
        fs_Hz = 200.0
        hp_cutoff_Hz = 2.0
        bp_stop_Hz = np.array([lowH, highH])
        b, at = signal.butter(1,bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
        channel = signal.lfilter(b, at, si, axis=0)
        b2, a2 = signal.butter(2, hp_cutoff_Hz/ (fs_Hz / 2.0), 'highpass')
        eeg_alt = signal.lfilter(b2, a2, channel, axis = 0)
        f, Pxx_den = signal.welch(eeg_alt[0:len(eeg_alt)-1], 200)
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        axes = plt.gca()
        axes.set_xlim([1,60])
        return eeg_alt

def filtering(si, lowH, highH):
        fs_Hz = 200.0
        hp_cutoff_Hz = 1.0
        bp_stop_Hz = np.array([lowH, highH])
        b, at = signal.butter(1,bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
        channel = signal.lfilter(b, at, si, axis=0)
        b2, a2 = signal.butter(1, hp_cutoff_Hz/ (fs_Hz / 2.0), 'highpass')
        eeg_alt = signal.lfilter(b2, a2, channel, axis = 0)
        return eeg_alt

def signal_to_noise_ratio1(liste_freq):
    maxi = max(liste_freq)
    mini1= min(liste_freq[0:len(liste_freq)/2])
    mini2= min(liste_freq[len(liste_freq)/2:len(liste_freq)])
    mini_vrai= (mini1+mini2)/2.0
    s_t_n_r= maxi/mini_vrai
    return s_t_n_r

def comparaison_niveau_alphas(ferme,ouvert):
    mean1=np.mean(ferme)
    mean2=np.mean(ouvert)
    return mean1/mean2


def compare_50Hz (a):
        f, Pxx_den = signal.welch(a[0:len(a)], 200)
        print "pompom"
        print extract_freqband(f, Pxx_den, 40,53)
        print signal_to_noise_ratio1(extract_freqband(f, Pxx_den, 40,53))
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        axes = plt.gca()
        axes.set_xlim([40,60])





def cut_compare(a):
        f, Pxx_den = signal.welch(a[0:len(a)/2], 200)
        f2,Pxx_den2 = signal.welch(a[len(a)/2:len(a)-1], 200)
        print extract_freqband(f, Pxx_den, 8,12)
        print extract_freqband(f2, Pxx_den2, 8,12)
        print signal_to_noise_ratio1(extract_freqband(f2, Pxx_den2, 6,14))
        print signal_to_noise_ratio1(extract_freqband(f, Pxx_den, 6,14))
        print "comparaison des niveaux moyens de alphas : " + str(comparaison_niveau_alphas(extract_freqband(f2, Pxx_den2, 8,12), extract_freqband(f, Pxx_den, 8,12)))
        print "hopla"
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        axes = plt.gca()
        axes.set_xlim([4,20])
        plt.semilogy(f2, Pxx_den2)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        axes = plt.gca()
        axes.set_xlim([4,20])

def get_clean_data_from_openBCI(t_csv):

    l_fichiers=[]
    for name in t_csv:

        #print name
        g= open(name,'rb')
        gcsv = csv.reader(g, delimiter=',')
        l = []

        for row in gcsv:
            for data in row:

                    l.append(float(data))
        a = np.asarray(l)
        #cut_compare(a)
        #compare_50Hz(a)
        print len(a)
#        f, Pxx_den = signal.welch(a[0:10000], 200)
#        plt.figure(figsize=(8, 4))
#        plt.semilogy(f, Pxx_den)
#        plt.xlabel('frequency [Hz]')
#        plt.ylabel('PSD [V**2/Hz]')
        l_fichiers.append([name, a])
        #print a.shape
    return l_fichiers


def ploter(a,stri):
    plt.figure(figsize=(8, 4))
    plt.plot(a)
    plt.ylabel(stri)
    plt.show()

def extract_freqband (f, Pxx_den, fmin, fmax):
    l_indices=[]
    for i in range(len(f)):
        if (f[i]<=fmax):
            if (f[i]>=fmin):
                l_indices.append(i)
    bandfreq=[]
    for ind in l_indices:
        bandfreq.append(Pxx_den[ind])
    #ploter(bandfreq, "voici les bandes de frequences")
    return bandfreq

def extract_freqband_centree (f, Pxx_den, fmin, fmax):
    l_indices=[]
    for i in range(len(f)):
        if (f[i]<=fmax):
            if (f[i]>=fmin):
                l_indices.append(i)
    bandfreq=[]
    for ind in l_indices:
        bandfreq.append(Pxx_den[ind])
    Pmax=max(bandfreq)
    li=[]
    for p in Pxx_den:
        li.append(p)
    ind_Pmax = li.index(Pmax)
    print ind_Pmax
    fenetre_centree=Pxx_den[ind_Pmax-4:ind_Pmax+5]
    print "hi " + str(fenetre_centree)
    ploter(fenetre_centree)
    ploter(bandfreq)
    return bandfreq




def ploterfourrier(signe):
    f, Pxx_den = signal.welch(signe, 200)
    test = extract_freqband(f,Pxx_den, 8, 12)
    print test
    #ploter(test)
    print "hello"
    print np.mean(test)

def suppressoutlayers(sign,scale):
    moy=np.mean(sign)
    for i in range(len(sign)):
        if (sign[i]>scale*moy):
            sign[i]=sign[i-1]
    return sign

def creerlistealphas(sign, windowlen, overlap):
    l = []
    nb_pas= int (len(sign)/overlap)
    for i in range(nb_pas):
      if ((i*overlap+windowlen) < len(sign)):
        f, Pxx_den = signal.welch(sign[i*overlap:(i*overlap+windowlen)], 200)
        t =extract_freqband(f, Pxx_den, 8,12)
        l.append(np.mean(t))
    return l

def creerlistedeltas(sign, windowlen, overlap):
    l = []
    nb_pas= int (len(sign)/overlap)
    for i in range(nb_pas):
      if ((i*overlap+windowlen) < len(sign)):
        f, Pxx_den = signal.welch(sign[i*overlap:(i*overlap+windowlen)], 200)
        t =extract_freqband(f, Pxx_den, 2.5,4.5)
        l.append(np.mean(t))
    return l

def printer(outputalphas):
#    print (outputalphas[80:120])
#    ploter(outputalphas[120:170],"voici les alphas")
    #ploter(outputalphas,"voici les alphas")
    val = moving_average(outputalphas,40)
    print "Voici la valeur moyenne des alphas : " + str(np.mean(val[50:len(val)/2]))
    print "Voici la valeur moyenne des alphas : " + str(np.mean(val[len(val)/2:len(val)]))
    ploter(moving_average(outputalphas,40),"voici les alphas moyennes")
    #ploter(moving_median(outputalphas,300),"voici les alphas medians")

def printer_d(outputalphas):
#    print (outputalphas[80:120])
#    ploter(outputalphas[120:170],"voici les alphas")
    #ploter(outputalphas,"voici les alphas")
    val = moving_average(outputalphas,40)
    print "Voici la valeur moyenne des deltas : " + str(np.mean(val[50:len(val)/2]))
    print "Voici la valeur moyenne des deltas : " + str(np.mean(val[len(val)/2:len(val)]))
    ploter(moving_average(outputalphas,40),"voici les deltas moyennes")
    #ploter(moving_median(outputalphas,300),"voici les alphas medians")


def regre(alphas):
    li=[]
    al=[]
    si = len(alphas)
    for i in range(len(alphas)):
        li.append([i])
        al.append([alphas[i]])
    li = np.asarray(li)
    li = li.reshape((si,1))
    alphas=np.asarray(al)
    alphas = alphas.reshape((si,1))
    #print alphas
    si = int(si*0.75)

    # Create linear regression object
    X_train = li[:si]
    X_test = li[si:]

    # Split the targets into training/testing sets
    alphas_train = alphas[:si]
    alphas_test = alphas[si:]

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, alphas_train)

    # Make predictions using the testing set
    alphas_pred = regr.predict(X_test)

    regr_coef= regr.coef_
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print "Mean squared error: " + str(mean_squared_error(alphas_test, alphas_pred))
    # Explained variance score: 1 is perfect prediction
    print 'Variance score: ' + str(r2_score(alphas_test, alphas_pred))

    # Plot outputs
    ploter(alphas_pred, "alpha predit")
    ploter(alphas_test, "alpha vrai")
    #print alphas_pred

    return regr_coef[0][0]

def suppressbourrin(sign):
    seuil = 10
    lentot = len(sign)
    compt = 0
    moy=np.mean(sign)
    signint=[sign[0]]
    for i in range(len(sign)):
     if(i!=0):
        if (sign[i]>0.00005):


            compt +=1

        else:
            if (sign[i]<(-1)*0.00005):

                compt +=1

            else:
                signint.append(sign[i])
    print len(signint)
    print len(sign)
    stdi = np.std(signint)
    l = []
    for i in range(len(sign)):
      if(i!=0):
        if (sign[i]>2*stdi+moy):

            compt +=1
            l.append (i)
        if (sign[i]<(-1)*2*stdi+moy):

            compt +=1
            l.append(i)
    ratio_suppression = 1.0*compt/lentot
    p , len_p =decoupe_seq_suivie(l)
    new_eeg=[]
    fin=[]

    for elm in p:

        if (len(elm) <= seuil):
            for ind in elm:
               sign[ind]= sign[elm[0]-1]  #on met tout le segment à la même valeur de la valeur avant dépassement du seuil
        else:
            if (len(fin)>0):
              if ((elm[0]-fin[len(fin)-1])>200):
                #print "voici elm[0] " + str(elm[0])
                #print "voici fin " + str(fin[len(fin)-1])
                new_eeg.append(sign[fin[len(fin)-1]:elm[0]])
                #on découpe l'EEG en deux bout pour sauter le bout bruité
                fin.append(elm[len(elm)-1])
              else:
                  fin.append(elm[len(elm)-1])
            else:
                 new_eeg.append(sign[0:elm[0]])
                 fin.append(elm[len(elm)-1])
    new_eeg.append(sign[fin[len(fin)-1]:len(sign)-1])
    print "voici le ratio de la suppression des donnees: " + str(ratio_suppression)
    if (len(new_eeg[0])<200):
        new_eeg=new_eeg[1:]
    #print new_eeg
    if (len(new_eeg[len(new_eeg)-1])<200):
        new_eeg=new_eeg[:len(new_eeg)-1]
    #print new_eeg[:2]
    verif_len_eeg(new_eeg)
    print len(sign)
    #print new_eeg
#    for li in new_eeg:
#        ploter(li, "fragment EEG")
#    ploter(sign, "avant suppression raffinee")
    return new_eeg


def verif_len_eeg(new_eeg):
    compt = 0
    for li in new_eeg:
        for val in li:
            compt+=1
    print compt



def decoupe_seq_suivie(l):
    p=[]
    for i in range(len(l)):

        if (i==0):
          p.append([l[i]])
        else:
            if (l[i]==l[i-1]+1):
                p[len(p)-1].append(l[i])
            else:
                p.append([l[i]])
    #print p
    len_l=[]
    for li in p:
        len_l.append(len(li))
#    print len_l
#    ploter(len_l,"distribution des longueur de listes supprimees")
    return p, len_l



def processing(a):
#Routine d'affichage

    sign = filtering(a, 48,52)
    plt.figure(figsize=(8, 4))
    #Le spectrogramme, le 50 Hz doit être plutôt "vide" (filtrage) le reste doit être jaune rouge tacheté
    _ = plt.specgram(sign, NFFT=128, Fs=200, noverlap=64)
    sign = suppressbourrin(sign)
    serie_alpha=[]
    serie_delta=[]
    for fragmt in sign:
        outputalphas = creerlistealphas(fragmt,200,25)
        outputdeltas = creerlistedeltas(fragmt,200,25)
        serie_alpha.extend(outputalphas)
        serie_delta.extend(outputdeltas)
#    regr_coef = regre(serie_alpha)
#    print "voici le coefficient de regression des alphas " + str(regr_coef)
#    regr_coef = regre(serie_delta)
#    print "voici le coefficient de regression des deltas " + str(regr_coef)
    #print output
    printer(serie_alpha)
    printer_d(serie_delta)
#    print "et maintenant avec suppressoutlayers"
#    regr_coef = regre(suppressoutlayers(serie_alpha,2))
#    print "voici le coefficient de regression des alphas " + str(regr_coef)
#    regr_coef = regre(suppressoutlayers(serie_delta,2))
#    print "voici le coefficient de regression des deltas " + str(regr_coef)
    #print output
    printer(suppressoutlayers(serie_alpha, 2))
    printer_d(suppressoutlayers(serie_delta,2))
    return [[regre(serie_alpha),regre(serie_delta)],[np.mean(serie_alpha),np.mean(serie_delta)]]

def processing_RS(a):


    sign = filtering(a, 48,52)
    plt.figure(figsize=(8, 4))
    #Le spectrogramme, le 50 Hz doit être plutôt "vide" (filtrage) le reste doit être jaune rouge tacheté
    _ = plt.specgram(sign, NFFT=128, Fs=200, noverlap=64)
    sign = suppressbourrin(sign)
    serie_alpha=[]
    serie_delta=[]
    for fragmt in sign:
        outputalphas = creerlistealphas(fragmt,200,25)
        outputdeltas = creerlistedeltas(fragmt,200,25)
        serie_alpha.extend(outputalphas)
        serie_delta.extend(outputdeltas)

    printer(serie_alpha)
    printer_d(serie_delta)
    return [np.mean(serie_alpha),np.mean(serie_delta)]
#    printer(suppressoutlayers(serie_alpha, 2))
#    printer_d(suppressoutlayers(serie_delta,2))

def merging(li): #fait la moyenne des quatre signaux
    len_liste=[]
    for i in range(len(li)):
        len_liste.append(len(li[i]))
    minilen=min(len_liste)
    eeg_moyen=[]
    for i in range(minilen):
        point_moyen=0
        for j in range(len(li)):
            point_moyen+=li[j][i]
        eeg_moyen.append(point_moyen/(len(li)))
    return eeg_moyen

print merging([[0.0],[1.5],[2.5],[3.5]])




def parcoursdossier(name, numb):
    os.chdir("C:\Users\Robin\Donnees_EEG_tests_Zeta")
    print os.getcwd()
    os.chdir(name)
    print os.getcwd()
    os.chdir(str(numb))
    print os.getcwd()

def read_RS1():
    os.chdir("RS1-data")
    print os.getcwd()
    topen=[["RS_ch1_session0_.txt"], ["RS_ch2_session0_.txt"], ["RS_ch3_session0_.txt"], ["RS_ch4_session0_.txt"]]
    li=[]
    for nami in topen:
        li.append(get_clean_data_from_openBCI(nami)[0][1])
    signale=merging(li)
    return processing_RS(signale)



def read_RS2():
    os.chdir("RS2-data")
    print os.getcwd()
    topen=[["RS2_ch1_session1_.txt"], ["RS2_ch2_session1_.txt"], ["RS2_ch3_session1_.txt"], ["RS2_ch4_session1_.txt"]]
    li=[]
    for nami in topen:
        li.append(get_clean_data_from_openBCI(nami)[0][1])
    signale=merging(li)
    processing_RS(signale)


def read_T():
    os.chdir("training-data")
    print os.getcwd()
    topen=[["T_ch1_session0_.txt"], ["T_ch2_session0_.txt"], ["T_ch3_session0_.txt"], ["T_ch4_session0_.txt"]]
    li=[]
    for nami in topen:
        li.append(get_clean_data_from_openBCI(nami)[0][1])
    signale=merging(li)
    return processing(signale)



#Ici mettre l'emplacement du fichier txt à analyser
#t2=["rpi_1_data/data/Casparian/1-TB/saving-data/saving-ch1-session1-.txt"]
t2=["rpi_1_data/data/Casparian/6-TB/RS-dataRS_ch2_session1_.txt"]
t2=["CASPARIAN2/casparianfinal2/1/RS1-data/RS_ch2_session0_.txt", "CASPARIAN2/casparianfinal2/1/RS1-data/RS_ch2_session0_.txt", "CASPARIAN2/casparianfinal2/1/RS1-data/RS_ch2_session0_.txt", "CASPARIAN2/casparianfinal2/1/RS1-data/RS_ch2_session0_.txt"]
t2=["proto6/11_1_18_b2fea39f7c90093449482f218d3777/saving-data/Saving-ch1-session1-.txt"]
num_seance=9
seq_moyenne_RS=[]
seq_reg=[]
seq_moyenne_T=[]
for num_seance in range(9):

    parcoursdossier("CASPARIAN2/casparianfinal2",num_seance+1)
    print "etat moyen RS"
    seq_moyenne_RS.append(read_RS1())
    parcoursdossier("CASPARIAN2/casparianfinal2",num_seance+1)
    print "now reading the training phases"
    reg, m_T = read_T()
    seq_reg.append(reg)
    seq_moyenne_T.append(m_T)
#    parcoursdossier("CASPARIAN2/casparianfinal2",num_seance+1)
#    print "etat moyen RS final"
#    read_RS2()

def extract_arg(seq, ind):
    li = []
    for elm in seq:
        if (not(math.isnan(elm[ind]))):
            li.append(elm[ind])
    return li


print seq_moyenne_RS
print "les alphas et les deltas moyens isolés RS"
ploter(extract_arg(seq_moyenne_RS,0), "et voici les alphas moyens RS")
ploter(extract_arg(seq_moyenne_RS,1), "et voici les deltas moyens RS")
print seq_moyenne_T
print "les alphas et les deltas moyens isolés T"
ploter(extract_arg(seq_moyenne_T,0), "et voici les alphas moyens T")
ploter(extract_arg(seq_moyenne_T,1), "et voici les deltas moyens T")
print seq_reg
ploter(extract_arg(seq_reg,0), "et voici les alphas moyens T")
ploter(extract_arg(seq_reg,1), "et voici les deltas moyens T")
