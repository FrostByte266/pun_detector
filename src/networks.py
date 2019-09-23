from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks

import data_preprocessing
import visuals

def train_classifier(dataset='/data/training.csv', ratio=0.4, maxlen=256, embedding_dim=200, glove_path='/data/glove.twitter.27B.200d.txt'):

    log_dir="/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    early_stop = callbacks.EarlyStopping(monitor='val_acc', patience=5)

    train_examples, test_examples, train_labels, test_labels, vocab_size, embedding_matrix = data_preprocessing.make_train_test(maxlen=maxlen, embedding_dim=embedding_dim, glove_path=glove_path)

    net = Sequential()
    net.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True))
    net.add(layers.Conv1D(128, 5, activation='relu'))
    net.add(layers.GlobalMaxPooling1D())
    net.add(layers.Dense(32, activation='relu'))
    net.add(layers.Dense(1, activation='sigmoid'))
    net.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = net.fit(train_examples, train_labels,
        epochs=20,
        verbose=True,
        validation_data=(test_examples, test_labels),
        batch_size=32,
        callbacks=[tensorboard_callback, early_stop]
    )

    visuals.plot_history(history)

if __name__ == '__main__':
    train_classifier()