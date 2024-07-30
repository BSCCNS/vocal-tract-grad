import librosa
import numpy as np
import scipy
import threading

from gfm_iaif import gfm_iaif

import sounddevice as sd

sd.default.latency = "low"

unused = """
import soundfile as sf
from functools import partial
from utils import h1h2, weighted_log_mag_mse_loss
from tract_proxy import VocalTractProxy
from optimize import TractControlsOptimizer
"""

current_frame = 0


class Resynth:
    def __init__(
        self,
        framelength=1024,
        hoplength=256,
        fmin=70,
        fmax=500,
        ncilinders=44,
        fs=44100,
    ):
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
            "vt_shifts": [0, 0, 0],
            "glottis_shifts": None,
            "tenseness_factor": None,
        }
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []
        self.update_devices()

    def __del__(self) -> None:
        self.stop_stream()

    def start_stream(self):
        self.prev_audio_orig = np.zeros(
            self.prev_frames * self.framelength, dtype="float32"
        )

        self.stream = sd.Stream(
            channels=1,
            callback=self.audio_callback,
            blocksize=self.process_blocks
            * self.framelength,  # TODO HARDCODED Buffer length !!!!
            samplerate=self.fs,
            dtype="float32",
        )
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
            return L[0] + L[1]
        else:
            return -1

    def update_parameter(self, param, value):
        self.params.update({param: value})

    def play_audio(self, audio_file):
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

        self.prev_audio_orig = np.zeros(
            self.prev_frames * self.framelength, dtype="float32"
        )

        def callback(outdata, frames, time, status):
            global current_frame
            chunksize = min(audio_file.shape[0] - current_frame, frames)

            indata = audio_file[current_frame : current_frame + chunksize]
            if chunksize < frames:
                outdata[chunksize:] = 0
                raise sd.CallbackStop()

            _outdata = np.empty_like(outdata[:chunksize])
            self.audio_callback(indata, _outdata, frames, time, status)
            outdata[:chunksize] = (
                _outdata  # audio_file[current_frame:current_frame + chunksize]
            )

            current_frame += chunksize

        event = threading.Event()
        stream = sd.OutputStream(
            channels=1,
            callback=callback,
            blocksize=self.process_blocks * self.framelength,
            samplerate=self.fs,
            dtype="float32",
            finished_callback=event.set,
        )
        with stream:
            event.wait()
        self.fs = 44100
        return

    def audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):

        if indata.shape[0] == 0:
            outdata[:] = 0
            return

        input_wav = np.concatenate((self.prev_audio_orig, librosa.to_mono(indata.T)))

        # read parameters
        if "vt_shifts" in self.params:
            vt_shifts = self.params["vt_shifts"]
        else:
            vt_shifts = []

        if "glottis_shifts" in self.params:
            glottis_shifts = self.params["glottis_shifts"]
        else:
            glottis_shifts = None

        if "tenseness_factor" in self.params:
            tenseness_factor = self.params["tenseness_factor"]
        else:
            tenseness_factor = None
        #
        # call audio processing!!!
        #
        output_wav = self.process(
            input_wav,
            vt_shifts=vt_shifts,
            glottis_shift=glottis_shifts,
            tenseness_factor=tenseness_factor,
        )

        #  keep previous 2 frames
        self.prev_audio_orig = input_wav[-2 * self.framelength :]
        # remove previous ending from signal and extra frame at beginning
        output_wav = output_wav[self.framelength : -self.framelength]

        # TODO find a  way to check channels!!!!
        output_wav = output_wav.reshape(outdata.shape)

        # outdata[:] = output_wav
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
        self.input_devices_indices = [
            d["index"] for d in devices if d["max_input_channels"] > 0
        ]
        self.output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]

    def get_devices(self):
        return self.input_devices, self.output_devices

    def process(
        self, audio_input, tract_shifts_per=None, glottis_shift=None, tenseness_mult=None
    ):
        # partition input in overlapping frames
        input_frames = librosa.util.frame(
            audio_input, frame_length=self.framelength, hop_length=self.hoplength
        )
        nframes = input_frames.shape[1]

        # get the LPC coefficients using the GFM-IAIB method
        tract_coeffs, glottis_coeffs, lip_coeffs = self.estimate_coeffs(input_frames)

        # isolate glottis signal
        glottis_signal = self.filter_frames(input_frames, tract_coeffs, np.ones([1]))
        glottis_frames = librosa.util.frame(glottis_signal, frame_length=self.framelength, hop_length=self.hoplength)

        # isolate excitation signal
        excitation_signal = self.filter_frames(glottis_frames, glottis_coeffs, np.ones([1]))
        excitation_frames = librosa.util.frame(excitation_signal, frame_length=self.framelength, hop_length=self.hoplength)

        # NEW METHOD TO
        # calculate resonant frequency and quality factor from glottis poles
        glottis_poles = np.apply_along_axis(np.roots, 1, glottis_coeffs.astype(np.complex128))
        glottis_poles = np.apply_along_axis(lambda x: x[x.imag.argsort()], 1, glottis_poles)
        glottis_poles_real = glottis_poles[1,:]
        glottis_poles_pos = glottis_poles[2,:]
        glottis_freqs = np.max(np.angle(glottis_poles_pos), axis=1)
        # glottis_qs = - 1 / np.tan(np.angle(glottis_poles) / 2)

        # calculate resonant frequencies of vocal tract
        tract_poles = np.apply_along_axis(np.roots, 1, tract_coeffs.astype(np.complex128)) 
        tract_poles_pos = tract_poles[tract_poles.imag > 0]
        tract_freqs = np.max(np.angle(tract_poles_pos), axis=1)
        # tract_qs = - 1 / np.tan(np.angle(tract_poles) / 2)

        # TODO apply tenseness and vocal effort multipliers
        glottal_shift = glottis_shift
        glottis_poles_pos *= np.exp(1j * glottal_shift) # TODO calculate glottal shift from tenseness and vocal effort/force
        # glottis_poles_real = ... # TODO change tilt calculated from vocal effort/force
        glottis_poles = np.stack((glottis_poles_real, glottis_poles_pos, glottis_poles_pos.conj()), axis=0).T
        glottis_coeffs = np.apply_along_axis(np.poly, 1, glottis_poles)

        # TODO apply F1, F2, F3 shifts
        f0 = glottis_freqs.mean() # TODO fix f0 estimation
        vt_shift_hz = self.shifts_to_freqs(tract_shifts_per, tract_freqs, f0) # TODO [para Fernando]
        tract_poles_pos = np.apply_along_axis(lambda poles: poles * np.exp(1j * shift) , 1, tract_poles_pos) # TODO [para Fernando]
        tract_poles = np.concatenate((tract_poles_pos, tract_poles_pos.conj()), axis=1)
        tract_coeffs = np.apply_along_axis(np.poly, 1, tract_poles)

        # regenerate signal
        audio_output = np.zeros_like(audio_input)
        for i in range(nframes):
            frame = excitation_frames[:, i]
            framepad = np.pad(frame, ((0, self.ncilinders + 1)), mode="edge") # TODO remove padding

            coeffs = np.polymul(glottis_coeffs, tract_coeffs)

            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=self.hoplength),
                librosa.frames_to_samples(i, hop_length=self.hoplength)
                + self.framelength,
            )
            audio_output[idx] += scipy.signal.lfilter(
                [1], coeffs[i, :], framepad
            )[self.ncilinders + 1 :] * scipy.signal.get_window(
                "hamming", self.framelength
            )

        return audio_output

    def shifts_to_freqs(
        self, percent_shifts, frequencies_orig, F0
    ):  # convert 3 freqs slider of percentage to frequencies
        percent_shifts = (
            np.array(percent_shifts) * 0.99 / 100
        )  # conversion to -1,1 but not getting quite there so freqz don't overlap
        frames = frequencies_orig.shape[0]
        nfreqs = percent_shifts.shape[0]
        shifts = np.zeros((frames, nfreqs))
        if percent_shifts[0] < 0 and percent_shifts[2] >= 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n, 0:4]
                F1 = np.interp(percent_shifts[0], [-1, 0], [F0, F1o])
                F3 = np.interp(percent_shifts[1], [0, 1], [F3o, F4])
                F2 = np.interp(percent_shifts[2], [-1, 0, 1], [F1, F2o, F3])
                shifts[n, :] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] >= 0 and percent_shifts[2] < 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n, 0:4]
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1o, F2o, F3o])
                F1 = np.interp(percent_shifts[0], [0, 1], [F1o, F2])
                F3 = np.interp(percent_shifts[2], [-1, 0], [F2, F3o])
                shifts[n, :] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] < 0 and percent_shifts[2] < 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n, 0:4]
                F1 = np.interp(percent_shifts[0], [-1, 0], [F0, F1o])
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1, F2o, F3o])
                F3 = np.interp(percent_shifts[2], [-1, 0], [F2, F3o])
                shifts[n, :] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] >= 0 and percent_shifts[2] >= 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[n, 0:4]
                F3 = np.interp(percent_shifts[2], [0, 1], [F3o, F4])
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1o, F2o, F3])
                F1 = np.interp(percent_shifts[0], [0, 1], [F1o, F2])
                shifts[n, :] = [F1 - F1o, F2 - F2o, F3 - F3o]

        return shifts * (2 * np.pi) / self.fs  # convertimos Hz a radianes

    def estimate_coeffs(self, data):
        nframes = data.shape[1]

        vtcoeffs = np.empty((nframes, self.ncilinders + 1))
        glcoeffs = np.empty((nframes, 4))
        lipcoeffs = np.empty((nframes, 2))

        for i in range(nframes):
            frame = data[:, i]
            vtcoeffs[i, :], glcoeffs[i, :], lipcoeffs[i, :] = gfm_iaif(frame, n_vt=self.ncilinders)

        return vtcoeffs, glcoeffs, lipcoeffs
    
    def filter_frames(self, data, b, a, framelength=None, hoplength=None, out=None):
        if framelength == None:
            framelength = self.framelength
        if hoplength == None:
            hoplength = self.hoplength

        nframes = data.shape[1]

        if a.ndim == 1:
            b = np.repeat(np.reshape(b, [1, -1]), nframes, axis=0)
        if a.ndim == 1:
            a = np.repeat(np.reshape(a, [1, -1]), nframes, axis=0)

        if out == None:
            if data.ndim == 1:
                out = np.zeros_like(data)
            else:
                out = np.zeros((nframes-1) * hoplength + framelength)
        
        if data.ndim == 1:
            data = librosa.util.frame(data, frame_length=framelength, hop_length=hoplength)

        for i in range(nframes):
            frame = data[:, i]
            framepad = np.pad(frame, ((0, self.ncilinders + 1)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=hoplength),
                librosa.frames_to_samples(i, hop_length=hoplength) + framelength,
            )

            out[idx] += scipy.signal.lfilter(b[i, :], a[i, :], framepad)[self.ncilinders + 1 :] * scipy.signal.get_window("hamming", framelength)

        return out


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
