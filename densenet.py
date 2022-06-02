def classification_layer(inputs):
    x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(inputs)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(512, activation='relu')(x)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(x)
    return output


def regression_layer(inputs):
    x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(inputs)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(512, activation='relu')(x)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output = tensorflow.keras.layers.Dense(2, activation='sigmoid', name='malignancy_regression')(x)
    return output


def add_dense_blocks(inputs, filter_size):
    for i in range(3):
        x = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same')(inputs)
        inputs = tensorflow.keras.layers.Concatenate()([x, inputs])
        x = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same')(inputs)
        inputs = tensorflow.keras.layers.Concatenate()([x, inputs])
        filter_size += 32
    return inputs


def add_transition_blocks(x, idx):
    if idx <= 1:
        x = tensorflow.keras.layers.Conv3D(filters=80, kernel_size=(1, 1, 1), padding='same')(x)
        x = tensorflow.keras.layers.MaxPooling3D()(x)
    elif idx == 2:
        x = tensorflow.keras.layers.Conv3D(filters=96, kernel_size=(1, 1, 1), padding='same')(x)
        x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    elif idx == 3:
        x = tensorflow.keras.layers.Conv3D(filters=94, kernel_size=(1, 1, 1), padding='same')(x)
        x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    return x


def add_network_blocks(x):
    filter_list = [160, 176, 184, 188, 190]
    # Create dense block
    for block, filter_size in enumerate(filter_list):
        # Add dense block
        x = add_dense_blocks(x, filter_size)
        # Create transition layer
        x = add_transition_blocks(x, block)
    return x


def dense_model(m_classes, t_classes):
    input_layer = tensorflow.keras.layers.Input(shape=(64, 64, 64, 1))
    x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(input_layer)
    x = add_network_blocks(x)
    x = tensorflow.keras.layers.MaxPooling3D()(x)

    output_malignancy = regression_layer(x)
    output_type = classification_layer(x)
    d_model = tensorflow.keras.Model(inputs=input_layer, outputs=[output_malignancy, output_type])
    return d_model