# required for packaging nicegui
import multiprocessing
multiprocessing.freeze_support()

from nicegui import ui, events, native
from gfmdriver import GFMDriver
import soundfile as sf
import io

driver = GFMDriver()

ui.page_title("Impossible Larynx LPC")

def handle_upload(e):
    data, samplerate = sf.read(io.BytesIO(e.content.read()))
    driver.store_audio(data, samplerate)
    #dialog.open()

def refresh_devices():
    driver.refresh_devices()
    select_input.update()
    select_output.update()

with ui.dialog() as preferences, ui.card():
    # cosas a ajustar: framelength, hoplength, volume_reduction, blocks en RT, input_devices???    
    with ui.column():
        select_input = ui.select(driver.inputs, label="Input", 
                                 value=driver.cur_input, on_change=driver.update_selected_input)
        select_output = ui.select(driver.outputs, label="Output", 
                                  value=driver.cur_output , on_change=driver.update_selected_output)
        ui.button(on_click=refresh_devices)
        ui.separator() 
        hoplength = ui.select([16, 32, 64, 128, 256, 512], label= "Hop Length", 
                              value=128, on_change=driver.set_hoplen)
        framelength = ui.select([64, 128, 256, 512, 1024, 2048], label= "Frame Length", value=512,
                              on_change=driver.set_framelen)
        blocks = ui.select([1,2,3,4,5,6,7,8], label= "Blocks", value=1,
                              on_change=driver.set_blocks)
 
    with ui.row():
        ui.button('Close', on_click=lambda: preferences.submit(None))

async def show_preferences():
    result = await preferences
    if result is not None:
        ui.notify(f'Funny click?')

with ui.header().classes('justify-between'):
    ui.label('Impossible Larynx LPC').classes("text-2xl")
    ui.button(on_click=show_preferences).props('flat color=white icon=settings')

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

    with ui.column().classes('w-2/6'): 
        F1_slider = ui.slider(min=-100, max=100, 
                              step=1, value=0).on('update:model-value', 
                                lambda e: driver.update_F1(e.args),throttle=1.0)
        ui.label().bind_text_from(F1_slider, 'value', 
                                  backward=lambda n: f'<-- close jaw               F1 %:: {n}               open jaw -->')
        # ui.label().bind_text_from(F1_slider, 'value')
        F2_slider = ui.slider(min=-100, max=100, step=1, value=0) \
            .on('update:model-value', lambda e: driver.update_F2(e.args),
            throttle=1.0)
        ui.label().bind_text_from(F2_slider, 'value', 
                                  backward=lambda n: f'<-- tongue forward               F2%::  {n}               tongue backwards')
        # ui.label().bind_text_from(F1_slider, 'value')
        F3_slider = ui.slider(min=-100, max=100, step=1, value=0) \
            .on('update:model-value', lambda e: driver.update_F3(e.args),
            throttle=1.0)
        ui.label().bind_text_from(F3_slider, 'value', 
                                  backward=lambda n: f'<-- close lips               F3%:: {n}               open lips -->')
        
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


ui.run(native=True, reload=False, port=native.find_open_port())


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
