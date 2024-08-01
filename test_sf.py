import soundfile as sf
import sounddevice as sd
import numpy as np
import threading
audio_file,fs = sf.read('C2.wav')
print (audio_file.shape, )
# def buffer(nframes):
#     print("INIT BUFFER")
#     end_idx=0
#     while end_idx<=len(audio_file):
#         buffer_idx = end_idx
#         end_idx = buffer_idx + nframes
#         print("BUFFERING",buffer_idx,end_idx)
#         yield audio_file[buffer_idx:end_idx]
        
current_frame = 0

def callback(outdata, frames, time, status):
    global current_frame
    if status:
        print(status)
    chunksize = min(audio_file.shape[0] - current_frame, frames)
    outdata[:chunksize] = audio_file[current_frame:current_frame + chunksize]
    if chunksize < frames:
        outdata[chunksize:] = 0
        raise sd.CallbackStop()
    current_frame += chunksize
    # try:
    #     print("Getting callback:",frames)
    #     indata = buffer(frames)
    #     indata = np.array(indata)
    #     print("Got:",indata.size)
    #     if indata.size <= 0:
    #         raise sd.CallbackAbort
    #     #_outdata = np.empty_like(outdata)
    #     #self.audio_callback(indata, _outdata, frames, time, status)
    #     outdata[:] = indata
    # except:
    #     raise sd.CallbackAbort
print(sd.query_devices())
event = threading.Event()

stream = sd.OutputStream(channels=audio_file.shape[1],
                        callback=callback,
                        blocksize=fs, 
                        samplerate=fs,
                        finished_callback=event.set)
with stream:
    event.wait()