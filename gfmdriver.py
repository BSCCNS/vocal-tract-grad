from gfm import Resynth

class GFMDriver:
    model = None

    def __init__(self):
        self.model = Resynth()
        self.nknobs = 22
        self.knobs = [[]]*self.nknobs

        self.inputs, self.outputs = self.model.get_devices()
        self.cur_input = self.inputs[1]
        self.cur_output = self.outputs[0]
        self.model.set_devices(self.cur_input,self.cur_output)
        self.audio = None
        self.samplerate = None

    def refresh_devices(self):
        self.inputs, self.outputs = self.model.get_devices()
        self.cur_input = self.inputs[1]
        self.cur_output = self.outputs[0]

    def update_selected_input(self, sel_input):
        self.cur_input = sel_input.value
        self.model.set_devices(self.cur_input,self.cur_output)

    def update_selected_output(self, sel_output):
        self.cur_output = sel_output.value
        self.model.set_devices(self.cur_input, self.cur_output)  

    def update_F1(self,value):
        self.model.params['vt_shifts'][0]=value

    def update_F2(self,value):
        self.model.params['vt_shifts'][1]=value

    def update_F3(self,value):
        self.model.params['vt_shifts'][2]=value

    def update_F0(self,value):
        self.model.params['glottis_shifts']=[value]

    def update_tenseness(self,value):
        self.model.params['tenseness_factor']=value

    def update_knob(self, n, value):
        pass

    def save_params(self):
        pass

    def load_params(self,filename):
        pass

    def get_latency(self):        
        return self.model.get_latency()
    
    def start_process(self):
        self.model.start_stream()

    def stop_process(self):
        self.model.stop_stream()
    
    def store_audio(self, data, samplerate):
        self.audio = data
        self.samplerate = samplerate

    def play_audio(self):        
        self.model.fs = self.samplerate
        self.model.play_audio(self.audio)

### 