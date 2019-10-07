from base64 import b64encode
import queue
import threading
from time import sleep

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.backend import set_session
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


def train_on_thread(queue, graph, sess):
    with graph.as_default():
        set_session(sess)
        print('START')
        net, tokenizer = networks.train_classifier(maxlen=300, embedding_dim=300, glove_path='/data/glove.6B.300d.txt')
        print('FINISH')
        queue.put((net, tokenizer))


class config_vars:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self

def train_gui(graph, sess):

    training_queue = queue.Queue()

    with open('/data/training.gif', 'rb') as f:
        training_gif = b64encode(f.read())

    with open('/data/check.gif', 'rb') as f:
        training_done_gif = b64encode(f.read())

    training_layout = [
        [sg.Text('Training in progress, this could take a while')],
        [sg.T(' ' * 12), sg.Image(data=training_gif, key='animation')],
    ]

    finished_layout = [
        [sg.T(' ' * 15), sg.Text('Training complete!', font=('Ubuntu', 24))],
        [sg.T(' ' * 10), sg.Image(data=training_done_gif, key='animation')],
    ]

    training_window = sg.Window('Training...', layout=training_layout, finalize=True)
    finished_window = sg.Window('Complete!', layout=finished_layout, finalize=True)

    # Hide the windows while we load the GIFs into the cache
    training_window.Hide()
    finished_window.Hide()

    training_window['animation'].UpdateAnimation(training_gif)
    finished_window['animation'].UpdateAnimation(training_done_gif)

    training_window.UnHide()
    

    threading.Thread(target=train_on_thread, args=(training_queue, graph, sess, ), daemon=True).start()

    with config_vars(timeout=100) as config:
        # Event loop for training window
        while True:
            event, values = training_window.Read(timeout=config.timeout)
            if event in (None, 'Exit'):
                exit(1)
            try:
                result = training_queue.get_nowait()
            except queue.Empty:
                result = None
            
            training_window['animation'].UpdateAnimation(training_gif)

            if result:
                config.timeout = 20
                training_window.Close()
                finished_window.UnHide()
                break
        
        # Event loop for the training finished window
        for _ in range(104):
            event, values = finished_window.Read(timeout=config.timeout)
            if event in (None, 'Exit'):
                exit(1)
            finished_window['animation'].UpdateAnimation(training_done_gif)
        
        sleep(2)
        finished_window.Close()
        return result
            

if __name__ == '__main__':

    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    layout = [
        [sg.Text('Load a net?')],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    window = sg.Window('Prompt', layout=layout)

    while True:
        event, values = window.Read()
        window.Close()
        ask = event == 'Yes'
        break
    if ask:
        with graph.as_default():
            set_session(sess)
            net, tokenizer = load_model()
            assert net is not None or tokenizer is not None
    else:
        net, tokenizer = train_gui(graph, sess)

    layout = [
        [sg.Input(key='in')],
        [sg.Text('Prediction:'), sg.Text('', size=(30, 1), key='out')],
        [sg.Text('Confidence:'), sg.Text('', size=(30, 1), key='confidence')],
        [sg.Text('Prediction (Raw output):'), sg.Text('', size=(30, 1), key='raw_out')],
        [sg.Button('Predict!', bind_return_key=True), sg.Exit()]
    ]

    window = sg.Window('Prediction interface', layout=layout, finalize=True)

    while True:
        event, values = window.Read()
        if event in (None, 'Exit'):
            break
        elif event == 'Predict!':
            with graph.as_default():
                set_session(sess)
                vectorized = tokenizer.texts_to_sequences([values['in']])
                padded = pad_sequences(vectorized, padding='post', maxlen=300)
                prediction = net.predict(padded)
                pred = prediction[0][0]
                window['confidence'].Update(f'{(pred*100 if pred > 0.5 else 100-pred*100):.2f}%')
                window['out'].Update('Pun' if pred > 0.5 else 'Not a pun')
                window['raw_out'].Update(pred)
    