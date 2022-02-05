import numpy as np
import keras


class UniversalDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))  
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class MyDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, array_dataX, array_dataY, batch_size=32, dim=(2000), n_channels=1, is_to_categorical=False,
                 n_classes=10, shuffle=True, is_embedding=True):
        'Initialization'
        self.is_embedding = is_embedding  # embedding
        self.dim = dim
        self.batch_size = batch_size
        self.array_dataX = array_dataX
        self.array_dataY = array_dataY
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.is_to_categorical = is_to_categorical
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        steps_per_epoch = int(np.floor(len(self.array_dataY) / self.batch_size))
        # print(steps_per_epoch)
        return steps_per_epoch
        return int(np.floor(len(self.array_dataY) / self.batch_size))

    def __getitem__(self, index):
        # print('index: %d ------------------------------' % index)
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.array_dataX[indexes]
        y = self.array_dataY[indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        if self.is_embedding is False:
            X = X[..., np.newaxis]  
        # X = X[:,:,np.newaxis]
        # X = np.expand_dims(X, -1)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.array_dataY))
        # print('self.indexes: %s' % self.indexes)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) 
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        if self.is_to_categorical:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


if __name__ == '__main__':
    import numpy as np


    params = {'dim': (32, 32, 32),
              'batch_size': 64,
              'n_classes': 6,
              'n_channels': 1,
              'shuffle': True}


    partition = [1, 2, 3, 4]  # IDs  #
    labels = [0, 0, 1, 1]  # Labels  #

    # Generators
    training_generator = UniversalDataGenerator(partition, labels, **params)
    validation_generator = UniversalDataGenerator(partition, labels, **params)

    # Design model
    # Import `Sequential` from `keras.models`
    from keras.models import Sequential
    
    # Import `Dense` from `keras.layers`
    from keras.layers import Dense

    # Initialize the model
    model = Sequential()

    # Add input layer 
    model.add(Dense(64, input_dim=12, activation='relu'))

    # Add output layer 
    model.add(Dense(1))


    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
    # workers。
    '''
    model.fit_generator(object, generator, steps_per_epoch, epochs = 1,
                        verbose = getOption("keras.fit_verbose", default = 1),
                        callbacks = NULL, view_metrics = getOption("keras.view_metrics",
                        default = "auto"), validation_data = NULL, validation_steps = NULL,
                        class_weight = NULL, max_queue_size = 10, workers = 1,
                        initial_epoch = 0)
    '''

    
    
