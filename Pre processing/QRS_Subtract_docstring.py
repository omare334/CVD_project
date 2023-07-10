import numpy as np
from scipy import signal


def Adaptive_singular_qrs(peak_index,ECG,Fs,Gaussian = True):
    """
    QRS Substraction Function using singular value decompostion
    
        Input:
            peak_index: N by 1 Numpy array where N is the number of QRS complex. 
                Elements specify the indexes of the R peaks.
            ECG: F by 1 Numpy array containing the ECG trace.
            Fs: Sampling rate in float.
            Gaussian: True to use the Gaussian window method for smooth joining of templates.
        Output:
            AAFG/AAF: Numpy array of the same size as ECG, containing the ECG after substraction.
    
    1. The alogorithm segments each QRS complex defined by a window start at 0.3* minimum RR interval
    before the Q waves and 0.7* minimum RR interval.
    2. The sign of the QRST(i.e. upstroke or downstroke from the baseline) is determined by comparing the
    maxium and minimum value of the segment this is checked for consistency later against the sign of the PC
    3. SVD is applied to the stacked rectangular matrix containing all the segments and the first temporal 
    eigenvector is identified as the PC
    4. PC(Tt) times the ratio of span of the segment(T[:,t]) vs span of the PC(Tt) is subtracted
    5. The means at start and end of each segment are calculated (ks and ke)
    6. A gaussian window of length m is used to joining subtracted segment and the original signal at start
    and end of each segment
    
    Xili Shi xs1219@ic.ac.uk 2021 April
    """


    peak_index0 = np.concatenate((np.array([0]),peak_index[:-1]),axis = 0)

    RRint = peak_index - peak_index0

    RRintmin = np.min(RRint)
    RRintminExT0 = np.min(RRint[1::])

    if peak_index[0]-np.floor(0.3*RRintminExT0)>0 and peak_index[-1]+np.floor(0.7*RRintminExT0) <=ECG.size:
        RRintmin = RRintminExT0
    cc = np.floor(0.3*RRintmin)+np.floor(0.7*RRintmin)
    rr = peak_index.size
    T = np.zeros((int(cc),rr))
    rcounter = int(0)
    QRS_Sign_Counter = 0
    for ri in peak_index:
        T[:,rcounter] = ECG[ri-int(np.floor(0.3*RRintmin)):ri+int(np.floor(0.7*RRintmin)),0]
        if np.abs(np.max(T[:,rcounter]))>np.abs(np.min(T[:,rcounter])):
            QRS_Sign_Counter = QRS_Sign_Counter + 1
        rcounter = rcounter + 1

    if QRS_Sign_Counter > np.floor(peak_index.size/2):
        QRS_Sign_Counter = 1
    else:
        QRS_Sign_Counter = 0
    


    u, s, vh = np.linalg.svd(T, full_matrices=False)

    PCs = u@np.diag(s)

    PC = PCs[:,0]
    if np.abs(np.max(PC))>np.abs(np.min(PC)):
        PC_Sign = 1
    else:
        PC_Sign = 0
    
    if PC_Sign != QRS_Sign_Counter:
        PC = PC*-1
    M=20
    M=np.floor(M*Fs/1000);
    stdev = (2*M-1)/(2*2.5)
    window = signal.windows.gaussian(int(2*M), stdev)

    Tt=PC
    M = int(M)
    w1=window[0:M]
    w2=window[M-1:-1]
    AAF=ECG.copy()
    AAFG=ECG.copy()

    for i in range(len(peak_index)):
        TAAFG=T[:,i]-Tt*((max(T[:,i])-min(T[:,i]))/(max(Tt)-min(Tt)))
        AAFG[int(peak_index[i]-np.floor(0.3*RRintmin)):int(peak_index[i]+np.floor(0.7*RRintmin)),0]=TAAFG.copy()
        AAF[int(peak_index[i]-np.floor(0.3*RRintmin)):int(peak_index[i]+np.floor(0.7*RRintmin)),0]=TAAFG.copy()    
        ks=(AAFG[int(peak_index[i]-np.floor(0.3*RRintmin))-1]-AAFG[int(peak_index[i]-np.floor(0.3*RRintmin))])/2
        ke=(AAFG[int(peak_index[i]+np.floor(0.7*RRintmin))-1]-AAFG[int(peak_index[i]+np.floor(0.7*RRintmin))])/2

        AAFG[int(peak_index[i]-np.floor(0.3*RRintmin))-int(M):int(peak_index[i]-np.floor(0.3*RRintmin)),0]=(
            AAFG[int(peak_index[i]-np.floor(0.3*RRintmin))-int(M):int(peak_index[i]-np.floor(0.3*RRintmin)),0]-ks*w1)
    
        AAFG[int(peak_index[i]-np.floor(0.3*RRintmin)):int(peak_index[i]-np.floor(0.3*RRintmin))+int(M),0]=(
            AAFG[int(peak_index[i]-np.floor(0.3*RRintmin)):int(peak_index[i]-np.floor(0.3*RRintmin))+int(M),0]+ks*w2)
    
        AAFG[int(peak_index[i]+np.floor(0.7*RRintmin))-int(M):int(peak_index[i]+np.floor(0.7*RRintmin)),0]=(
            AAFG[int(peak_index[i]+np.floor(0.7*RRintmin))-int(M):int(peak_index[i]+np.floor(0.7*RRintmin)),0]-ke*w1)
    
        AAFG[int(peak_index[i]+np.floor(0.7*RRintmin)):int(peak_index[i]+np.floor(0.7*RRintmin))+int(M),0]=(
            AAFG[int(peak_index[i]+np.floor(0.7*RRintmin)):int(peak_index[i]+np.floor(0.7*RRintmin))+int(M),0]-ke*w2)


    if Gaussian == True:
        return AAFG
    else:
        return AAF