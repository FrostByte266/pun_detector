import queue
import threading
from time import sleep

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import PySimpleGUI as sg 

import networks

sg.ChangeLookAndFeel('Black')

def ask(question, yes_button='Yes', no_button='No', title='Prompt', close_behavior='exit'):
    layout = [
        [sg.Text(question)],
        [sg.Button(yes_button, key='yes'), sg.Button(no_button, key='no')]
    ]

    window = sg.Window('Prompt' ,layout=layout)

    while True:
        event, values = window.Read(timeout=100)
        if event is None:
            if close_behavior == 'exit':
                exit()
            else:
                window.Close()
                raise RuntimeError
        else:
            window.Close()
            return event == 'yes'

def save_model(model, tokenizer, default_model_path='/data/model.h5', default_tokenizer_path='/data/tokenizer.pickle'):
    layout = [
        [sg.InputText(default_model_path), sg.FileSaveAs(file_types=(('Hierarchical Data Format', '*.h5'),))],
        [sg.InputText(default_tokenizer_path), sg.FileSaveAs(file_types=(("Pickle", "*.pickle"),))],
        [sg.Save(bind_return_key=True), sg.Cancel()]
    ]

    window = sg.Window('Save model', layout=layout)

    while True:
        event, values = window.Read()
        if event in (None, 'Cancel'):
            break
        elif event == 'Save':
            networks.save_classifier_and_tokenizer(model, tokenizer, model_data=values[0], tokenizer_data=values[1])
            break




def load_model(default_model_path='/data/model.h5', default_tokenizer_path='/data/tokenizer.pickle'):
    layout = [
        [sg.InputText(default_model_path), sg.FileBrowse(file_types=(('Hierarchical Data Format', '*.h5'),))],
        [sg.InputText(default_tokenizer_path), sg.FileBrowse(file_types=(("Pickle", "*.pickle"),))],
        [sg.Open(bind_return_key=True), sg.Cancel()]
    ]

    window = sg.Window('Load model', layout=layout)

    while True:
        event, values = window.Read()
        if event in (None, 'Cancel'):
            rval = (None, None)
            break
        elif event == 'Open':
            rval = networks.load_classifier_and_tokenizer(model_path=values[0], tokenizer_path=values[1])
            break
    window.Close()
    return rval


def train_on_thread(queue):
    print('START')
    try:
        net, tokenizer = networks.train_classifier(maxlen=300, embedding_dim=300, glove_path='/data/glove.840B.300d.txt')
    except Exception:
        print('Encountered an exception')
    print('FINISH')
    queue.put((net, tokenizer))


class LoopVars:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self

def train_gui():

    gui_queue = queue.Queue()

    layout = [
        [sg.Text('Training in progress, this could take a while', key='txt')],
        [sg.T(' ' * 10, key='space'), sg.Image('/data/training.gif', key='img')]
    ]

    window = sg.Window('Training...', layout=layout)

    threading.Thread(target=train_on_thread, args=(gui_queue,), daemon=True).start()

    with LoopVars(c=1, updated=False, finished=False, timeout=100) as config:
        while True:
            event, values = window.Read(timeout=config.timeout)
            if config.finished:
                window['img'].UpdateAnimation('/data/check.gif', time_between_frames=config.timeout)
                config.c += 1
                if config.c > 91:
                    sleep(3)
                    window.Close()
                    return message
                else:
                    continue
            if event is None or event == 'Exit':
                break
            try:
                message = gui_queue.get_nowait()
            except queue.Empty:           
                message = None             

            window['img'].UpdateAnimation('/data/training.gif', time_between_frames=config.timeout)

            if message:
                window['txt'].Update('Training Complete!')
                window['space'].Update(' ' * 5)
                window['img'].Update('/data/check.gif')
                config.timeout = 20
                config.finished = True

if __name__ == '__main__':
    layout = [
        [sg.Text('Load a net?')],
        [sg.Button('Yes', key='yes'), sg.Button('No', key='no')]
    ]

    window = sg.Window('Prompt' ,layout=layout)

    while True:
        event, values = window.Read()
        window.Close()
        ask = event == 'yes'
        break
    if ask:
        net, tokenizer = load_model()
        assert net is not None or tokenizer is not None
        print(net)
    else:
        net, tokenizer = train_gui()
        print((net, tokenizer))