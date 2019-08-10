from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import tensorflow as tf
#from keras.layers import Conv2D,AveragePooling2D,MaxPooling2D,BatchNormalization,Activation,Flatten,Dense
#from keras.activations import relu
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
from subprocess import Popen 
from optparse   import OptionParser
from time       import gmtime, strftime
from array import array 
from root_numpy import root2array, array2tree, rec2array, fill_hist
import ROOT
import math
import random
import uproot
import os
from ROOT import gROOT, TPaveLabel, TPie, gStyle, gSystem, TGaxis, TStyle, TLatex, TString, TF1,TFile,TLine, TLegend, TH1D,TH2D,THStack, TGraph, TGraphErrors,TChain, TCanvas, TMatrixDSym, TMath, TText, TPad, TVectorD, RooFit, RooArgSet, RooArgList, RooArgSet, RooAbsData, RooAbsPdf, RooAddPdf, RooWorkspace, RooExtendPdf,RooCBShape, RooLandau, RooFFTConvPdf, RooGaussian, RooBifurGauss, RooArgusBG,RooDataSet, RooExponential,RooBreitWigner, RooVoigtian, RooNovosibirsk, RooRealVar,RooFormulaVar, RooDataHist, RooHist,RooCategory, RooChebychev, RooSimultaneous, RooGenericPdf,RooConstVar, RooKeysPdf, RooHistPdf, RooEffProd, RooProdPdf, TIter, kTRUE, kFALSE, kGray, kRed, kDashed, kGreen,kAzure, kOrange, kBlack,kBlue,kYellow,kCyan, kMagenta, kWhite

parser = OptionParser()
parser.add_option('--maxevents',action="store",type="int",dest="maxevents",default=1000000000)
parser.add_option('--epochs',action="store",type="int",dest="epochs",default=5)
parser.add_option('--branch',action="store",type="int",dest="branch",default=1)
#branch:1-23,the branch to be drawn
(options, args) = parser.parse_args()

def UnderOverFlow1D(h):
    Bins=h.GetNbinsX()
    h.SetBinContent( 1,  h.GetBinContent(1)+h.GetBinContent(0) )
    h.SetBinError(   1,  math.sqrt( h.GetBinError(1)*h.GetBinError(1) + h.GetBinError(0)*h.GetBinError(0)) )
    h.SetBinContent( Bins,  h.GetBinContent(Bins)+h.GetBinContent(Bins+1) )
    h.SetBinError(   Bins,  math.sqrt( h.GetBinError(Bins)*h.GetBinError(Bins) + h.GetBinError(Bins+1)*h.GetBinError(Bins+1)) )
    return h

def array2D_float(array):
    for i in range(0,array.shape[0]):
           for j in range(0,array.shape[1]):
                    array[i][j]=array[i][j][0]
    return array

def train_and_apply():

    np.random.seed(1)
    ROOT.gROOT.SetBatch()

    #Extract data from root file
    tree = uproot.open("out_all.root")["outA/Tevts"]
    branch_mc=["MC_B_P","MC_B_eta","MC_B_phi","MC_B_pt","MC_D0_P","MC_D0_eta","MC_D0_phi","MC_D0_pt","MC_Dst_P","MC_Dst_eta","MC_Dst_phi","MC_Dst_pt","MC_Est_mu","MC_M2_miss","MC_mu_P","MC_mu_eta","MC_mu_phi","MC_mu_pt","MC_pis_P","MC_pis_eta","MC_pis_phi","MC_pis_pt","MC_q2"]
    branch_rec=["B_P","B_eta","B_phi","B_pt","D0_P","D0_eta","D0_phi","D0_pt","Dst_P","Dst_eta","Dst_phi","Dst_pt","Est_mu","M2_miss","mu_P","mu_eta","mu_phi","mu_pt","pis_P","pis_eta","pis_phi","pis_pt","q2"]
    nvariable=len(branch_mc)
    x_train = tree.array(branch_mc[0], entrystop=options.maxevents)
    for i in range(1,nvariable):
          x_train = np.vstack((x_train,tree.array(branch_mc[i], entrystop=options.maxevents)))
    x_test = tree.array(branch_rec[0], entrystop=options.maxevents)
    for i in range(1,nvariable):
          x_test = np.vstack((x_test,tree.array(branch_rec[i], entrystop=options.maxevents)))
    x_train=x_train.T
    x_test=x_test.T
    x_test=array2D_float(x_test)
    #Different type of reconstruction variables

    #BN normalization   
    gamma=0
    beta=0.2
  
    ar=np.array(x_train)
    a = K.constant(ar[:,0])
    mean = K.mean(a)
    var = K.var(a)
    x_train = K.eval(K.batch_normalization(a, mean, var, gamma, beta))
    for i in range(1,nvariable):
        a = K.constant(ar[:,i])
        mean = K.mean(a)
        var = K.var(a)
        a = K.eval(K.batch_normalization(a, mean, var, gamma, beta))
        x_train = np.vstack((x_train, a))
    x_train=x_train.T

    ar=np.array(x_test)
    a = K.constant(ar[:,0])
    mean = K.mean(a)
    var = K.var(a)
    x_test = K.eval(K.batch_normalization(a, mean, var, gamma, beta))
    for i in range(1,nvariable):
        a = K.constant(ar[:,i])
        mean = K.mean(a)
        var = K.var(a)
        a = K.eval(K.batch_normalization(a, mean, var, gamma, beta))
        x_test = np.vstack((x_test, a))
    x_test=x_test.T

    #Add noise, remain to be improved
    noise = np.random.normal(loc=0.0, scale=0.01, size=x_train.shape)
    x_train_noisy = x_train + noise
    noise = np.random.normal(loc=0.0, scale=0.01, size=x_test.shape)
    x_test_noisy = x_test + noise
    x_train = np.clip(x_train, -1., 1.)
    x_test = np.clip(x_test, -1., 1.)
    x_train_noisy = np.clip(x_train_noisy, -1., 1.)
    x_test_noisy = np.clip(x_test_noisy, -1., 1.)

    # Network parameters
    input_shape = (x_train.shape[1],)
    batch_size = 128
    latent_dim = 2

    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)
    
    # Generate the latent vector
    latent = Dense(latent_dim, name='latent_vector')(x)
    
    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()
     
    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense( shape[1] )(latent_inputs)
    x = Reshape((shape[1],))(x)     
    outputs = Activation('tanh', name='decoder_output')(x)
    
    # Instantiate Decoder Model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    
    autoencoder.compile(loss='mse', optimizer='adam')
    
    # Train the autoencoder
    autoencoder.fit(x_train_noisy,
                    x_train,
                    validation_data=(x_test_noisy,x_test),
                    epochs=options.epochs,
                    batch_size=batch_size)
    
    # Predict the Autoencoder output from corrupted test imformation
    x_decoded = autoencoder.predict(x_test_noisy)

    # Draw Comparision Plots 
    c = TCanvas("c","c", 700,700)
    fPads1 = TPad("pad1", "Run2", 0.0, 0.29, 1.00, 1.00)
    fPads2 = TPad("pad2", "", 0.00, 0.00, 1.00, 0.29)
    fPads1.SetBottomMargin( 0.007)
    fPads1.SetLeftMargin(   0.10)
    fPads1.SetRightMargin(  0.03)
    fPads2.SetLeftMargin(   0.10 )
    fPads2.SetRightMargin(  0.03) 
    fPads2.SetBottomMargin( 0.25)
    fPads1.Draw()
    fPads2.Draw()
    fPads1.cd()
    nbin=50
    min=-1.
    max=1.
    variable="P^{B}"
    lbin=(max-min)/nbin
    lbin=str(float((max-min)/nbin))
    xtitle=branch_rec[options.branch-1]
    ytitle="Events/"+lbin+"GeV"
    h_rec=TH1D("h_rec",""+";%s;%s"%(xtitle,ytitle),nbin,min,max)
    h_rec.Sumw2()
    h_pre=TH1D("h_pre",""+";%s;%s"%(xtitle,ytitle),nbin,min,max)  
    h_pre.Sumw2()
    for i in range(x_test_noisy.shape[0]):
              h_rec.Fill(x_test_noisy[i][options.branch-1])
              h_pre.Fill(x_decoded[i][options.branch-1])
    h_rec=UnderOverFlow1D(h_rec)
    h_pre=UnderOverFlow1D(h_pre)
    maxY = TMath.Max( h_rec.GetMaximum(), h_pre.GetMaximum() ) 
    h_rec.SetLineColor(2)
    h_rec.SetFillStyle(0)
    h_rec.SetLineWidth(2)
    h_rec.SetLineStyle(1)
    h_pre.SetLineColor(3) 
    h_pre.SetFillStyle(0)
    h_pre.SetLineWidth(2)
    h_pre.SetLineStyle(1)   
    h_rec.SetStats(0)
    h_pre.SetStats(0)
    h_rec.GetYaxis().SetRangeUser( 0 , maxY*1.1 )
    h_rec.Draw("HIST")
    h_pre.Draw("same HIST")
    h_rec.GetYaxis().SetTitleSize(0.06)
    h_rec.GetYaxis().SetTitleOffset(0.78)
    theLeg = TLegend(0.5, 0.45, 0.95, 0.82, "", "NDC")
    theLeg.SetName("theLegend")
    theLeg.SetBorderSize(0)
    theLeg.SetLineColor(0)
    theLeg.SetFillColor(0)
    theLeg.SetFillStyle(0)
    theLeg.SetLineWidth(0)
    theLeg.SetLineStyle(0)
    theLeg.SetTextFont(42)
    theLeg.SetTextSize(.05)
    theLeg.AddEntry(h_rec,"Reconstruction","L");
    theLeg.AddEntry(h_pre,"Prediction","L");
    theLeg.SetY1NDC(0.9-0.05*6-0.005)
    theLeg.SetY1(theLeg.GetY1NDC())
    fPads1.cd()
    theLeg.Draw()
    title = TLatex(0.91,0.93,"AE prediction compare with reconstruction, epochs="+str(options.epochs))
    title.SetNDC()
    title.SetTextSize(0.05)  
    title.SetTextFont(42)
    title.SetTextAlign(31)  
    title.SetLineWidth(2)
    title.Draw()
    fPads2.cd()
    h_Ratio = h_pre.Clone("h_Ratio")
    h_Ratio.Divide( h_rec )    
    h_Ratio.SetLineColor(1)
    h_Ratio.SetLineWidth(2)
    h_Ratio.SetMarkerStyle(8)
    h_Ratio.SetMarkerSize(0.7)
    h_Ratio.GetYaxis().SetRangeUser( 0 , 2 )
    h_Ratio.GetYaxis().SetNdivisions(504,0);
    h_Ratio.GetYaxis().SetTitle("Pre/Rec")
    h_Ratio.GetYaxis().SetTitleOffset(0.35)
    h_Ratio.GetYaxis().SetTitleSize(0.13)
    h_Ratio.GetYaxis().SetTitleSize(0.13)
    h_Ratio.GetYaxis().SetLabelSize(0.11)
    h_Ratio.GetXaxis().SetLabelSize(0.1)
    h_Ratio.GetXaxis().SetTitleOffset(0.8)
    h_Ratio.GetXaxis().SetTitleSize(0.14);
    h_Ratio.SetStats(0)
    axis1=TGaxis( min,1,max,1, 0,0,0, "L")
    axis1.SetLineColor(1)
    axis1.SetLineWidth(1)
    for i in range(1,h_Ratio.GetNbinsX()+1,1):
                    D  = h_rec.GetBinContent(i)
                    eD = h_rec.GetBinError(i)
                    if D==0: eD=0.92
                    B  = h_pre.GetBinContent(i)
                    eB = h_pre.GetBinError(i)
                    if B<0.1 and eB>=B :
                                   eB=0.92
                                   Err= 0.
                    if B!=0.: 
                                   Err=TMath.Sqrt((eD*eD)/(B*B)+(D*D*eB*eB)/(B*B*B*B))
                                   h_Ratio.SetBinContent(i, D/B)
                                   h_Ratio.SetBinError(i, Err)
                    if B==0.:
                                   Err=TMath.Sqrt( (eD*eD)/(eB*eB)+(D*D*eB*eB)/(eB*eB*eB*eB) )
                                   h_Ratio.SetBinContent(i, D/0.92)
                                   h_Ratio.SetBinError(i, Err)
                    if D==0 and B==0:  
                                   h_Ratio.SetBinContent(i, -1)
                                   h_Ratio.SetBinError(i, 0)
                    h_Ratio.Draw("e0"); axis1.Draw();

    c.SaveAs(branch_rec[options.branch-1]+"_comparision.png")

if __name__ == '__main__':
    train_and_apply()
