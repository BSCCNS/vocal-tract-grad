import librosa
import numpy as np
import scipy
import threading

from gfm_iaif import gfm_iaif

import sounddevice as sd

sd.default.latency = 'low'

unused = """
import soundfile as sf
from functools import partial
from utils import h1h2, weighted_log_mag_mse_loss
from tract_proxy import VocalTractProxy
from optimize import TractControlsOptimizer
"""

current_frame = 0
class Resynth:
    def __init__(self, framelength=1024, hoplength=256, fmin=70, fmax = 500, ncilinders = 44, fs=44100):
        self.framelength = framelength
        self.hoplength = hoplength
        self.fmin = fmin
        self.fmax = fmax
        self.ncilinders = ncilinders
        self.fs = fs
        self.process_blocks = 1
        self.stream = None
        self.prev_frames = 2
        self.prev_audio_orig = None
        self.params = {
            'vt_shifts': [0,0,0],
            'glottis_shifts': None,
            'tenseness_factor': None
        }
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []   
        self.update_devices()     

    def __del__(self) -> None:
        self.stop_stream()

    def start_stream(self):
        self.prev_audio_orig = np.zeros( self.prev_frames * self.framelength, dtype="float32")        

        self.stream = sd.Stream(
            channels=1,
            callback=self.audio_callback,
            blocksize=self.process_blocks*self.framelength, # TODO HARDCODED Buffer length !!!!
            samplerate=self.fs,
            dtype="float32")
        print("Starting stream")
        self.stream.start()

    def stop_stream(self):
        if self.stream is not None:
            print("Stopping stream")
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_latency(self):       
        if self.stream is not None:
            L = self.stream.latency
            return L[0]+L[1]
        else:
            return -1
    
    def update_parameter(self,param,value):
        self.params.update({param: value})

    def play_audio(self,audio_file):  
        global current_frame      
        # input_wav = librosa.to_mono(audio_file.T)
        # # read parameters
        # if "vt_shifts" in self.params:
        #     vt_shifts = self.params['vt_shifts']
        # else:
        #     vt_shifts = []

        # if "glottis_shifts" in self.params:
        #     glottis_shifts = self.params['glottis_shifts']  
        # else:
        #     glottis_shifts = None
            
        # if "tenseness_factor" in self.params:
        #     tenseness_factor = self.params['tenseness_factor']  
        # else:
        #     tenseness_factor = None        

        # output_wav = self.process(input_wav, vt_shifts=vt_shifts, 
        #                    glottis_shift=glottis_shifts, 
        #                    tenseness_factor=tenseness_factor)        
        # sd.play(output_wav, samplerate=self.fs)
        # self.fs = 44100
        # return
        # TODO Play the file using real time engine and not full file conversion

        current_frame = 0

        self.prev_audio_orig = np.zeros( self.prev_frames * self.framelength, dtype="float32")        

        def callback(outdata, frames, time, status):
            global current_frame
            chunksize = min(audio_file.shape[0] - current_frame, frames)
            
            indata = audio_file[current_frame:current_frame + chunksize]
            if chunksize < frames: 
                outdata[chunksize:] = 0
                raise sd.CallbackStop()

            _outdata = np.empty_like(outdata[:chunksize])
            self.audio_callback(indata, _outdata, frames, time, status)
            outdata[:chunksize] = _outdata #audio_file[current_frame:current_frame + chunksize]

            current_frame += chunksize

        event = threading.Event()
        stream = sd.OutputStream(channels=1,
                                callback=callback,
                                blocksize=self.process_blocks*self.framelength, 
                                samplerate=self.fs,
                                dtype="float32",
                        finished_callback=event.set)
        with stream:
            event.wait()
        self.fs = 44100
        return

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
    
        if indata.shape[0]==0:
            outdata[:]=0
            return
        
        input_wav = np.concatenate( (self.prev_audio_orig, librosa.to_mono(indata.T)))

        # read parameters
        if "vt_shifts" in self.params:
            vt_shifts = self.params['vt_shifts']
        else:
            vt_shifts = []

        if "glottis_shifts" in self.params:
            glottis_shifts = self.params['glottis_shifts']  
        else:
            glottis_shifts = None
            
        if "tenseness_factor" in self.params:
            tenseness_factor = self.params['tenseness_factor']  
        else:
            tenseness_factor = None
        # 
        # call audio processing!!!
        #
        output_wav = self.process(input_wav, vt_shifts=vt_shifts, 
                                  glottis_shift=glottis_shifts, 
                                  tenseness_factor=tenseness_factor)

        #  keep previous 2 frames
        self.prev_audio_orig = input_wav[-2*self.framelength:] 
        # remove previous ending from signal and extra frame at beginning
        output_wav = output_wav[self.framelength:-self.framelength]
        
        # TODO find a  way to check channels!!!!
        output_wav = output_wav.reshape(outdata.shape)

        #outdata[:] = output_wav
        outdata[:] = output_wav


    def update_devices(self):
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        self.output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        self.input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        self.output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]

    def get_devices(self):
        return self.input_devices, self.output_devices
    
    def process(self, audio_input, vt_shifts=None, glottis_shift=None, tenseness_factor=None):
        
        # esta pirula de aqui es porque estoy probando ajustarme a un framelength variable
        # para dejar que sounddevice controle la latencia. No funciona todavia
        inner_framelength = self.framelength #min(self.framelength,audio_input.shape[0])
        inner_hoplength = self.hoplength #max(8,inner_framelength/default_hopratio)
        #
        #
        # first decompose in frames
        input_frames = librosa.util.frame(audio_input, 
                                          frame_length=inner_framelength, 
                                          hop_length=inner_hoplength)
        nframes = input_frames.shape[1]

        # FIRST STAGE 
        # Get the LPC coefficients
        #
        vtcoeffs = np.empty((nframes, self.ncilinders+1))
        glcoeffs = np.empty((nframes,4))
        lipcoeffs = np.empty((nframes, 2))
        for i in range(nframes):
            frame = input_frames[:, i]
            vtcoeffs[i,:], glcoeffs[i,:], lipcoeffs[i,:] = gfm_iaif(frame, n_vt=self.ncilinders)
        #
        # let's get the glottis source isolated
        glottis_iaif = np.zeros_like(audio_input)
        #vocalt_iaif = np.zeros_like(audio_input)
        for i in range(nframes):  
            frame = input_frames[:, i]
            framepad = np.pad(frame, ((0,self.ncilinders+1)), mode='edge')
            idx = np.arange(librosa.frames_to_samples(i, hop_length=inner_hoplength), librosa.frames_to_samples(i, hop_length=inner_hoplength)+inner_framelength)
            glottis_iaif[idx] += scipy.signal.lfilter(vtcoeffs[i,:], [1], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)
            #vocalt_iaif[idx]  += scipy.signal.lfilter(glcoeffs[i,:], [1], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)
        glottis_frames = librosa.util.frame(glottis_iaif, frame_length=inner_framelength, hop_length=inner_hoplength)
        #
        # and we can now obtain the excitation, removing ALSO the glottis filter
        # no_glottis = np.zeros_like(audio_input)
        # for i in range(nframes):  
        #     frame = glottis_frames[:, i]
        #     framepad = np.pad(frame, ((0,self.ncilinders+1)), mode='edge')
        #     idx = np.arange(librosa.frames_to_samples(i, hop_length=inner_hoplength), librosa.frames_to_samples(i, hop_length=inner_hoplength)+inner_framelength)
        #     no_glottis[idx] += scipy.signal.lfilter(glcoeffs[i,:], [1], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)
        # no_glottis_frames = librosa.util.frame(no_glottis, frame_length=inner_framelength, hop_length=inner_hoplength)

        # # SECOND STAGE 
        # # Identify the current filter pàrameters (frequencies, etc)
        # #
        # # some frames have sound issues and the filters are not physical, we must skip them
        valid_frame_mask = np.empty(nframes)

        # # Glottis roots
        # glottis_poles = np.empty((nframes,3),dtype=np.complex128)
        # glottis_phase_poles = np.empty((nframes,1),dtype=np.complex128)
        # glottis_real_poles = np.empty((nframes,1),dtype=np.complex128)
        # glottis_frequencies = np.empty((nframes,1))
        # for n in range(nframes):
        #     poles = np.roots(glcoeffs[n,:])
        #     phase_poles = np.array([r for r in poles if np.imag(r) > 0])
        #     if phase_poles.shape[0]==1:
        #         glottis_poles[n,:] = poles.copy()        
        #         glottis_phase_poles[n,:] = phase_poles.copy()
        #         glottis_real_poles[n,:] = np.array([r for r in poles if np.imag(r) == 0])
        #         glottis_frequencies[n,:] = np.arctan2(phase_poles.imag, phase_poles.real) * (self.fs / (2 * np.pi))
        #         valid_frame_mask[n] = True
        #     else:
        #         glottis_poles[n,:] = 0  
        #         glottis_phase_poles[n,:] = 0
        #         glottis_real_poles[n,:] = 0
        #         glottis_frequencies[n,:] = 0
        #         valid_frame_mask[n] = False

        lpc_glottis = np.zeros_like(glcoeffs)
        for i in range(nframes):
            frame = glottis_frames[:, i]
            lpc_glottis[i,:] = librosa.lpc(frame, order=3)
        # LPC Glottis roots
        glottis_poles = np.empty((nframes,3),dtype=np.complex128)
        glottis_phase_poles = np.empty((nframes,1),dtype=np.complex128)
        glottis_real_poles = np.empty((nframes,1),dtype=np.complex128)
        glottis_frequencies = np.empty((nframes,1))
        glottis_qualityfactor = np.empty((nframes,1))
        for n in range(nframes):
            poles = np.roots(lpc_glottis[n,:])
            phase_poles = np.array([r for r in poles if np.imag(r) > 0])
            if phase_poles.shape[0]==1:
                glottis_poles[n,:] = poles       
                glottis_phase_poles[n,:] = phase_poles
                glottis_real_poles[n,:] = np.array([r for r in poles if np.imag(r) == 0])
                glottis_frequencies[n,:] = np.arctan2(phase_poles.imag, phase_poles.real) * (self.fs / (2 * np.pi))
                glottis_qualityfactor[n,:] = np.angle(np.log(phase_poles))
                valid_frame_mask[n] = True
            else:
                valid_frame_mask[n] = False

        glottis_formant = glottis_frequencies.mean()
        #
        # Now get the roots of the vocal tract
        # Vocal tract roots
        vt_poles = np.zeros((nframes,self.ncilinders),dtype=np.complex128)
        vt_phase_poles = np.empty((nframes,int(self.ncilinders/2)),dtype=np.complex128)
        vt_frequencies = np.empty((nframes,int(self.ncilinders/2)))

        for n in range(nframes):    
            poles = np.roots(vtcoeffs[n,:])
            phase_poles = np.array([r for r in poles if np.imag(r) > 0])
            if phase_poles.shape[0] == self.ncilinders/2:
                vt_poles[n,:] = poles.copy()
                vt_phase_poles[n,:] = phase_poles.copy()    
                freqs = np.arctan2(phase_poles.imag, phase_poles.real) * (self.fs / (2 * np.pi)) 
                idx_sort = freqs.argsort() 
                vt_frequencies[n,:] = freqs[idx_sort].real
                vt_phase_poles[n,:] = vt_phase_poles[n,:][idx_sort]
                valid_frame_mask[n] = True
            else:
                vt_poles[n,:] = 0
                vt_phase_poles[n,:] = 0
                vt_frequencies[n,:] = 0
                valid_frame_mask[n] = False

        # THIRD STAGE: vocal conversion from the input parameters
        # 
        # First we convert vocal tract % parameters to frequencies
        # 
        vt_shift_hz = self.shifts_to_freqs(vt_shifts, vt_frequencies, glottis_formant)
        #vt_shift_hz = np.zeros_like(vt_shift_hz)
        # NOW we compute the new vocal tract model
        # by shifting the formants
        # 
        new_vt_phase_poles =vt_phase_poles.copy()
        for n in range(nframes):
            for s,shift in enumerate(vt_shift_hz[n,:]):
                new_vt_phase_poles[n,s] = new_vt_phase_poles[n,s]*np.exp(shift*1j)  # TODO aqui va con menos o con mas?

        new_vt_poles = np.concatenate( (new_vt_phase_poles,
                                            new_vt_phase_poles.conjugate()) , axis=1)

        new_vtcoeffs = np.empty((nframes,self.ncilinders+1))

        for n in range(nframes):
            if valid_frame_mask[n]:
                new_vtcoeffs[n,:] = np.poly(new_vt_poles[n,:])
            else:
                new_vtcoeffs[n,:] = vtcoeffs[n,:] # for failed frames we do nothing, might create noise

        if tenseness_factor is not None:
            #
            # Change tenseness and vocal force
            #        
            f0 = np.concatenate([librosa.yin(input_frames[:,i] / np.max(np.abs(input_frames[:,i])), 
                                             fmin=self.fmin, fmax=self.fmax, frame_length=inner_framelength, hop_length=inner_hoplength, 
                                             sr=self.fs, center=False, trough_threshold=0.1) for i in range(nframes)])
            
            Rd = np.empty(nframes)
            for i in range(nframes):
                X = librosa.amplitude_to_db(np.abs(librosa.stft(glottis_frames[:,i], n_fft=inner_framelength, hop_length=inner_framelength)))
                h1bin = int(np.round(f0[i] / self.fs * inner_framelength))
                h2bin = int(np.round(2 * f0[i] / self.fs * inner_framelength))
                Rd[i] = (X[h1bin,1] - X[h2bin,1] + 7.6) / 11.
            tenseness = np.clip(1 - Rd / 3, 0, 1)
            #loudness = librosa.feature.rms(y=input, frame_length=inner_framelength, hop_length=inner_hoplength)

            if tenseness_factor>0:
                tenseness = tenseness + (1-tenseness)*(tenseness_factor)
            else:
                tenseness = tenseness + (tenseness)*(tenseness_factor)

            # make a synthethic glottis from this tenseness
            # synthetic_glottis = Glottis(self.ncilinders, self.fs)
            # glottis_signal = synthetic_glottis.get_waveform(tenseness=torch.Tensor(tenseness), freq=torch.Tensor(f0.reshape(-1, 1)), frame_len=inner_hoplength).detach().numpy()
            
            


            # and here we change the glottis to the synthetic one with different tenseness
            glottis_frames = librosa.util.frame(glottis_signal, frame_length=inner_framelength, hop_length=inner_hoplength)
            
        # Create the audio with the new vocaltract model
        audio_output = np.zeros_like(audio_input)
        for i in range(nframes):  
            frame = glottis_frames[:, i]
            framepad = np.pad(frame, ((0,self.ncilinders+1)), mode='edge')
            idx = np.arange(librosa.frames_to_samples(i, hop_length=inner_hoplength), librosa.frames_to_samples(i, hop_length=inner_hoplength)+inner_framelength)
            audio_output[idx] += scipy.signal.lfilter([1], new_vtcoeffs[i,:], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)

        print(f"deviation: {np.sum(np.abs(new_vtcoeffs-vtcoeffs))}, masked {sum(valid_frame_mask)/nframes:.2} ")
        #if glottis_shift is None:
        return audio_output
        

        ## else, we continue changing the glottis
        # We remove the glottis from the current audio
        glottis_frames = librosa.util.frame(audio_output, frame_length=inner_framelength, hop_length=inner_hoplength)
        no_glottis = np.zeros_like(audio_input)
        for i in range(nframes):  
            frame = glottis_frames[:, i]
            framepad = np.pad(frame, ((0,self.ncilinders+1)), mode='edge')
            idx = np.arange(librosa.frames_to_samples(i, hop_length=inner_hoplength), librosa.frames_to_samples(i, hop_length=inner_hoplength)+inner_framelength)
            no_glottis[idx] += scipy.signal.lfilter(glcoeffs[i,:], [1], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)

        # Glottis roots
        glottis_poles = np.empty((nframes,3),dtype=np.complex128)
        glottis_phase_poles = np.empty((nframes,1),dtype=np.complex128)
        glottis_real_poles = np.empty((nframes,1),dtype=np.complex128)
        glottis_frequencies = np.empty((nframes,1))
        for n in range(nframes):
            poles = np.roots(glcoeffs[n,:])
            phase_poles = np.array([r for r in poles if np.imag(r) > 0])
            if phase_poles.shape[0]==1:
                glottis_poles[n,:] = poles.copy()        
                glottis_phase_poles[n,:] = phase_poles.copy()
                glottis_real_poles[n,:] = np.array([r for r in poles if np.imag(r) == 0])
                glottis_frequencies[n,:] = np.arctan2(phase_poles.imag, phase_poles.real) * (self.fs / (2 * np.pi))
                valid_frame_mask[n] = True
            else:
                glottis_poles[n,:] = 0  
                glottis_phase_poles[n,:] = 0
                glottis_real_poles[n,:] = 0
                glottis_frequencies[n,:] = 0
                valid_frame_mask[n] = False

            # shifting the glottis
            glottal_shift = glottis_shift *(2*np.pi)/self.fs
            new_glottis_phase_poles = np.where(valid_frame_mask,
                                            (glottis_phase_poles[:,0])*np.exp(glottal_shift*1j),0)


            new_glottis_poles = np.empty((nframes,3))
            new_glcoeffs = np.empty((nframes,4))

            new_glottis_poles = np.stack( (new_glottis_phase_poles,
                                                np.conjugate(new_glottis_phase_poles),
                                                glottis_real_poles[:,0]) ).T
            #esto no me funciona, tengo que hacer el loop como un pobre campesino
            #np.where(valid_frame_mask,np.poly(new_glottis_poles),0)
            for n in range(nframes):
                if valid_frame_mask[n]:
                    new_glcoeffs[n,:] = np.poly(new_glottis_poles[n])
                else:
                    new_glcoeffs[n,:] = glcoeffs[n,:]

            # new glottis
            audio_output = np.zeros_like(audio_input)
            for i in range(nframes):  
                frame = glottis_frames[:, i]
                framepad = np.pad(frame, ((0,self.ncilinders+1)), mode='edge')
                idx = np.arange(librosa.frames_to_samples(i, hop_length=inner_hoplength), librosa.frames_to_samples(i, hop_length=inner_hoplength)+inner_framelength)
                audio_output[idx] += scipy.signal.lfilter([1], new_glcoeffs[i,:], framepad)[self.ncilinders+1:] * scipy.signal.get_window("hamming", inner_framelength)

        return audio_output

    def shifts_to_freqs(self, percent_shifts, frequencies_orig, F0): # convert 3 freqs slider of percentage to frequencies
        percent_shifts = np.array(percent_shifts)*0.99/100 # conversion to -1,1 but not getting quite there so freqz don't overlap
        frames = frequencies_orig.shape[0]
        nfreqs = percent_shifts.shape[0]
        shifts = np.zeros((frames,nfreqs))
        if percent_shifts[0]<0 and percent_shifts[2]>=0:
            for n in range(frames): # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n,0:4]
                F1 = np.interp(percent_shifts[0], [-1,0], [ F0, F1o]   )
                F3 = np.interp(percent_shifts[1], [0, 1], [ F3o,F4 ] )
                F2 = np.interp(percent_shifts[2], [-1, 0, 1], [ F1, F2o, F3 ] )
                shifts[n,:] = [F1-F1o,F2-F2o,F3-F3o]
        if percent_shifts[0]>=0 and percent_shifts[2]<0:
            for n in range(frames): # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n,0:4]
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [ F1o, F2o, F3o ] )
                F1 = np.interp(percent_shifts[0], [ 0, 1], [ F1o, F2 ]   )
                F3 = np.interp(percent_shifts[2], [-1, 0], [ F2, F3o ] )
                shifts[n,:] = [F1-F1o,F2-F2o,F3-F3o]
        if percent_shifts[0]<0 and percent_shifts[2]<0:
            for n in range(frames): # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n,0:4]
                F1 = np.interp(percent_shifts[0], [-1,0], [ F0, F1o]   )
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [ F1, F2o, F3o ] )
                F3 = np.interp(percent_shifts[2], [-1, 0], [ F2, F3o ] )
                shifts[n,:] = [F1-F1o,F2-F2o,F3-F3o]
        if percent_shifts[0]>=0 and percent_shifts[2]>=0:
            for n in range(frames): # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n,0:4]
                F3 = np.interp(percent_shifts[2], [0,1], [ F3o, F4 ] )
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [ F1o, F2o, F3 ] )
                F1 = np.interp(percent_shifts[0], [0,1], [ F1o, F2]   )
                shifts[n,:] = [F1-F1o,F2-F2o,F3-F3o]
        
        return shifts*(2*np.pi)/self.fs # convertimos Hz a radianes




"""
MORE UNUSED CODE
# A la excitacion le hemos sacado primero el vocal tract, y ahora le sacamos la glottis
excitation_frames = librosa.util.frame(excitation_iaif, frame_length=framelength, hop_length=hoplength)
# new glottis
exc_plus_glottis = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = excitation_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    exc_plus_glottis[idx] += scipy.signal.lfilter([1], new_glcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)
# new vocal tract
EG_frames = librosa.util.frame(exc_plus_glottis, frame_length=framelength, hop_length=hoplength)
audio_output = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = EG_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    audio_output[idx] += scipy.signal.lfilter([1], new_vtcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

# O recomponemos a partir del new vocal tract nada mas...
audio_output = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = glottis_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    audio_output[idx] += scipy.signal.lfilter([1], new_vtcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

















self.update_devices()
self.default_input_device = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
self.default_output_device = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]

self.stream = sd.Stream(
    channels=2,
    callback=self.audio_callback,
    blocksize=self.block_frame,
    samplerate=self.config.samplerate,
    dtype="float32")
self.stream.start()


def update_devices(self):
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    self.input_devices = [
        f"{d['name']} ({d['hostapi_name']})"
        for d in devices
        if d["max_input_channels"] > 0
    ]
    self.output_devices = [
        f"{d['name']} ({d['hostapi_name']})"
        for d in devices
        if d["max_output_channels"] > 0
    ]
    self.input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
    self.output_devices_indices = [
        d["index"] for d in devices if d["max_output_channels"] > 0
    ]

def set_devices(self, input_device, output_device):
    sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
    sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
    print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
    print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))

   def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        start_time = time.perf_counter()
        print("\nStarting callback")
        self.input_wav[:] = np.roll(self.input_wav, -self.block_frame)
        self.input_wav[-self.block_frame:] = librosa.to_mono(indata.T)

        # infer
        _audio, _model_sr = self.svc_model.infer(
            self.input_wav,
            self.config.samplerate,
            spk_id=self.config.spk_id,
            threhold=self.config.threhold,
            pitch_adjust=self.config.f_pitch_change,
            use_spk_mix=self.config.use_spk_mix,
            spk_mix_dict=self.config.spk_mix_dict,
            use_enhancer=self.config.use_vocoder_based_enhancer,
            pitch_extractor_type=self.config.select_pitch_extractor,
            safe_prefix_pad_length=self.f_safe_prefix_pad_length,
        )

        # debug sola
        '''
        _audio, _model_sr = self.input_wav, self.config.samplerate
        rs = int(np.random.uniform(-200,200))
        print('debug_random_shift: ' + str(rs))
        _audio = np.roll(_audio, rs)
        _audio = torch.from_numpy(_audio).to(self.device)
        '''

        if _model_sr != self.config.samplerate:
            key_str = str(_model_sr) + '_' + str(self.config.samplerate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(_model_sr, self.config.samplerate,
                                                         lowpass_filter_width=128).to(self.device)
            _audio = self.resample_kernel[key_str](_audio)
        temp_wav = _audio[
                   - self.block_frame - self.crossfade_frame - self.sola_search_frame - self.last_delay_frame: - self.last_delay_frame]

        # sola shift
        conv_input = temp_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_frame, device=self.device)) + 1e-8)
        sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        temp_wav = temp_wav[sola_shift: sola_shift + self.block_frame + self.crossfade_frame]
        print('sola_shift: ' + str(int(sola_shift)))

        # phase vocoder
        if self.config.use_phase_vocoder:
            temp_wav[: self.crossfade_frame] = phase_vocoder(
                self.sola_buffer,
                temp_wav[: self.crossfade_frame],
                self.fade_out_window,
                self.fade_in_window)
        else:
            temp_wav[: self.crossfade_frame] *= self.fade_in_window
            temp_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer = temp_wav[- self.crossfade_frame:]

        outdata[:] = temp_wav[: - self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        end_time = time.perf_counter()
        print('infer_time: ' + str(end_time - start_time))
        if flag_vc:
            self.window['infer_time'].update(int((end_time - start_time) * 1000))

            

#
# mover F1 entre F0 y F2 (al bajar, jaw closing)
# mover F2 entre F1 y F3 (al subir lengua adelante)
# mover F3 entre F2 y F4 (al bajar cierra labios)
#
# Subir vocal force (higher glottal formant, higher tilt)
# Tenseness wider/higher glotal formant"""