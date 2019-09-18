from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks

import data_preprocessing
import visuals

def train_classifier(dataset='/data/training.csv', ratio=0.4, maxlen=256):

    log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    train_examples, test_examples, train_labels, test_labels, vocab_size = data_preprocessing.make_train_test(maxlen=maxlen)

    embedding_dim = 256

    net = Sequential()
    net.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    net.add(layers.Conv1D(128, 5, activation='relu'))
    net.add(layers.GlobalMaxPooling1D())
    net.add(layers.Dense(32, activation='relu'))
    net.add(layers.Dense(1, activation='sigmoid'))
    net.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = net.fit(train_examples, train_labels,
        epochs=10,
        verbose=True,
        validation_data=(test_examples, test_labels),
        batch_size=32,
        callbacks=[tensorboard_callback]
    )

    visuals.plot_history(history)

if __name__ == '__main__':
    train_classifier()