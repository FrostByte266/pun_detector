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

    window = sg.Window('Prompt' ,layout=layout, finalize=True)

    while True:
        event, values = window.Read()
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
    

if __name__ == '__main__':
    net, tokenizer = networks.load_classifier_and_tokenizer()
    if ask('Would you like to load a saved network?'):
        net, tokenizer = load_model()
        assert net is not None or tokenizer is not None
        print(net)
    else:
        # layout = [
        #     [sg.Text('Training in progress, this could take a while')],
        #     [sg.UpdateAnimation('/data/nn.gif')]
        # ]
        # window = sg.Window('Training...', layout=layout)
        net, tokenizer = networks.train_classifier(maxlen=300, embedding_dim=300, glove_path='/data/glove.840B.300d.txt')