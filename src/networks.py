from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, callbacks

import data_preprocessing
import visuals

def train_classifier(dataset='/data/training.csv', ratio=0.4, maxlen=256, embedding_dim=200, glove_path='/data/glove.twitter.27B.200d.txt'):

    log_dir="/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)

    train_examples, test_examples, train_labels, test_labels, vocab_size, embedding_matrix, tokenizer = data_preprocessing.make_train_test(maxlen=maxlen, embedding_dim=embedding_dim, glove_path=glove_path)

    net = Sequential()
    net.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True))
    net.add(layers.Conv1D(256, 2, activation='relu'))
    net.add(layers.GlobalMaxPooling1D())
    net.add(layers.Dense(128, activation='relu'))
    net.add(layers.Dense(1, activation='sigmoid'))
    net.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = net.fit(train_examples, train_labels,
        epochs=20,
        verbose=True,
        validation_data=(test_examples, test_labels),
        batch_size=16,
        callbacks=[tensorboard_callback, early_stop]
    )

    return net, tokenizer


if __name__ == '__main__':
    net, tokenizer = train_classifier(glove_path='/data/glove.840B.300d.txt', embedding_dim=300)

    try:
        while True:
            inp = input('Enter sentence: ')
            if inp == 'exit':
                raise KeyboardInterrupt
            example = tokenizer.texts_to_sequences([inp])
            example = pad_sequences(example, padding='post', maxlen=256)
            pred = net.predict(example)
            print(pred)
    except KeyboardInterrupt:
        print('\nExiting')
