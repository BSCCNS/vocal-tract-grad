from nicegui import ui, events
from gfmdriver import GFMDriver
import soundfile as sf
import io

driver = GFMDriver()

def handle_upload(e):
    data, samplerate = sf.read(io.BytesIO(e.content.read()))
    driver.store_audio(data, samplerate)
    #dialog.open()

ui.label('Voice resynthesis with GFM')
ui.separator()
with ui.row().classes('w-full no-wrap'):    
    # left panel
    with ui.column().classes('w-2/6'): # buttons
        with ui.row():
            ui.button('START', on_click=driver.start_process)
            ui.button('STOP', on_click=driver.stop_process, color='red')
        ui.separator()
        with ui.row():
            ui.button('Save conf', on_click=driver.save_params, color='grey')
            ui.button('Load conf', on_click=driver.load_params, color='grey')
        ui.separator()
        with ui.row():
            ui.label("Load audio:")
            ui.upload(on_upload=handle_upload).props('accept=.wav').classes('max-w-full')            
            ui.button('Play audio', on_click=driver.play_audio, color='grey')

        ui.separator() 
        ui.label("Select input")
        select_input = ui.select(driver.inputs, value=driver.cur_input, on_change=driver.update_selected_input)
        ui.label("Select output")
        select_output = ui.select(driver.outputs, value=driver.cur_output, on_change=driver.update_selected_output)
        ui.button('Refresh devices', on_click=driver.refresh_devices, color='grey')

    with ui.column().classes('w-2/6'): # sliders 
# mover F2 entre F1 y F3 (al subir lengua adelante)
# mover F3 entre F2 y F4 (al bajar cierra labios)
        F1_slider = ui.slider(min=-100, max=100, 
                              step=1, value=0).on('update:model-value', 
                                lambda e: driver.update_F1(e.args),throttle=1.0)
        ui.label().bind_text_from(F1_slider, 'value', 
                                  backward=lambda n: f'<-- close jaw        F1 %:: {n}          open jaw -->')
        # ui.label().bind_text_from(F1_slider, 'value')
        F2_slider = ui.slider(min=-100, max=100, step=1, value=0) \
            .on('update:model-value', lambda e: driver.update_F2(e.args),
            throttle=1.0)
        ui.label().bind_text_from(F2_slider, 'value', 
                                  backward=lambda n: f'<-- tongue forward      F2%::  {n}      tongue backwards')
        # ui.label().bind_text_from(F1_slider, 'value')
        F3_slider = ui.slider(min=-100, max=100, step=1, value=0) \
            .on('update:model-value', lambda e: driver.update_F3(e.args),
            throttle=1.0)
        ui.label().bind_text_from(F3_slider, 'value', 
                                  backward=lambda n: f'<-- close lips         F3%:: {n}       open lips -->')
        
        # ui.label().bind_text_from(F1_slider, 'value')        
        t_switch = ui.switch('Activate tenseness')
        with ui.column().bind_visibility_from(t_switch, 'value'):
            tenseness_slider = ui.slider(min=-100, max=100, step=1, value=0) \
                .on('update:model-value', lambda e: driver.update_tenseness(e.args),
                throttle=1.0)
            ui.label().bind_text_from(tenseness_slider, 'value', 
                                  backward=lambda n: f'Tenseness stretch (-100 to 100):: {n}')
            vocalness_slider = ui.slider(min=-100, max=100, step=1, value=0) \
                .on('update:model-value', lambda e: driver.update_tenseness(e.args),
                throttle=1.0)
            ui.label().bind_text_from(vocalness_slider, 'value', 
                                  backward=lambda n: f'Vocalness stretch (-100 to 100):: {n}')
    with ui.column().classes('bg-blue-100 w-2/6'): # knobs
        measure = ui.button('Latency', on_click=lambda: ui.notify(f'{int(driver.get_latency()*100)} ms'), color='grey')

        with ui.grid(columns=4):
            for n in range(driver.nknobs):
                driver.knobs[n] = ui.knob(0.5, show_value=True).classes('col-span-1').on(
                    'update:model-value', lambda e: driver.update_knob(n, e.args))


ui.run() #(native=True)


"""
    # right panel
    with ui.grid(columns=2):
        ui.label('label 1')

with ui.splitter() as splitter:
    with splitter.before:
        ui.label('This is some content on the left hand side.').classes('mr-2')
    with splitter.after:
        ui.label('This is some content on the right hand side.').classes('ml-2')

with ui.grid(columns=2):

class App:
    def __init__(self):
        self.number = 1

app = App()
v = ui.checkbox('tenseness', value=True)
with ui.column().bind_visibility_from(v, 'value'):
    ui.slider(min=1, max=3).bind_value(app, 'number')
    ui.toggle({1: 'A', 2: 'B', 3: 'C'}).bind_value(app, 'number')
    ui.number().bind_value(app, 'number')

ui.button('Click me!', on_click=lambda: ui.notify('You clicked me!'))

checkbox = ui.checkbox('check me')
ui.label('Check!').bind_visibility_from(checkbox, 'value')


slider = ui.slider(min=0, max=100, value=50)
ui.label().bind_text_from(slider, 'value')

knob = ui.knob(0.3, show_value=True)

with ui.knob(color='orange', track_color='grey-2').bind_value(knob, 'value'):
    ui.icon('volume_up'
            """
