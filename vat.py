
import librosa
import numpy as np
import scipy
import sounddevice as sd
import torch 

from gfm_iaif import gfm_iaif
from glottis import Glottis

import tqdm 

# input, fs = librosa.load("C#2.wav")
input, fs = librosa.load("0_47.wav")

framelength = 1024
hoplength = 128
fmin, fmax = 70, 500
ncilinders = 44

frames = librosa.util.frame(input, frame_length=framelength, hop_length=hoplength)
nframes = frames.shape[1]

X = librosa.amplitude_to_db(np.abs(librosa.stft(input, n_fft=framelength))).squeeze()

f0 = np.concatenate([librosa.yin(frames[:,i] / np.max(np.abs(frames[:,i])), fmin=fmin, fmax=fmax, frame_length=framelength, hop_length=hoplength, sr=fs, center=False, trough_threshold=0.1) for i in range(nframes)])

glottis = np.zeros_like(input)
vtcoeffs = np.empty((ncilinders+1,nframes))
glcoeffs = np.empty((4, nframes))
lipcoeffs = np.empty((2, nframes))

for i in range(nframes):
    frame = frames[:, i]
    vtcoeffs[:,i], glcoeffs[:,i], lipcoeffs[:,i] = gfm_iaif(frame, n_vt=ncilinders)
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    glottis[idx] += scipy.signal.lfilter(vtcoeffs[:,i], [1], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

#freqresp = np.empty((framelength//2, nframes), dtype=np.complex64)
#for i in range(nframes):
#    w, freqresp[:,i] = scipy.signal.freqz([1], vtcoeffs[:,i], plot=lambda w, h: plot.line(w, 20. * np.log10(np.abs(h)), alpha=0.05), fs=fs)
#    w, freqresp[:,i] = scipy.signal.freqz(vtcoeffs[:,i], [1], plot=lambda w, h: plot.line(w, 20. * np.log10(np.abs(h)), alpha=0.05, color="red"), fs=fs)

X = librosa.amplitude_to_db(np.abs(librosa.stft(glottis, n_fft=framelength))).squeeze()

gframes = librosa.util.frame(glottis, frame_length=framelength, hop_length=hoplength)

Rd = np.empty(nframes)

for i in range(nframes):
    X = librosa.amplitude_to_db(np.abs(librosa.stft(gframes[:,i], n_fft=framelength, hop_length=framelength)))
    h1bin = int(np.round(f0[i] / fs * framelength))
    h2bin = int(np.round(2 * f0[i] / fs * framelength))
    Rd[i] = (X[h1bin,1] - X[h2bin,1] + 7.6) / 11.
    #gframes[:,i]

tenseness = np.clip(1 - Rd / 3, 0, 1)
loudness = librosa.feature.rms(y=input, frame_length=framelength, hop_length=hoplength)

glottis = Glottis(ncilinders, fs)
glottis_signal = glottis.get_waveform(tenseness=torch.Tensor(tenseness), freq=torch.Tensor(f0.reshape(-1, 1)), frame_len=hoplength).detach().numpy()
gframes = librosa.util.frame(glottis_signal, frame_length=framelength, hop_length=hoplength).copy()

for i in range(gframes.shape[1]):
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    glottis_signal[idx] *= (loudness[0,i] + 10**(5/20))

devices = sd.query_hostapis()

print("Glottis...")
sd.play(glottis_signal)
sd.wait()

gframes = librosa.util.frame(glottis_signal, frame_length=framelength, hop_length=hoplength)

out = np.zeros_like(input)

for i in tqdm(range(min(nframes, gframes.shape[1]))):
    frame = gframes[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    # out[idx] += np.fft.irfft(np.fft.rfft(frame * scipy.signal.get_window("hamming", framelength))[0:-1] * Hkl[:,i], n=framelength)
    out[idx] += scipy.signal.lfilter([1], vtcoeffs[:,i], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

out = np.nan_to_num(out, nan=0.0)

print("Resynth...")
sd.play(out)
sd.wait()




