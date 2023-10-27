import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def create_and_train_model(train_dir, test_dir, valid_dir):
    # tamanho da imagem
    img_width, img_height = 190, 190

    # Definir tamanho do lote
    batch_size = 64

    # Gerar dados de treinamento e dividir por 255 fazer imagem no intervalo -1,1
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input, validation_split=0.2)

    #dados de treino
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    # Gerar dados de validação
    valid_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    # Gerar dados de teste
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Construir a rede neural conv2d e densa
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="same", input_shape=(img_width, img_height, 3)),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treino
    history = model.fit(train_generator, epochs=30, validation_data=valid_generator)

    # Acurácia
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    return history

def plot_accuracy_and_loss(history):
    # Plotar gráfico de acurácia
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['treinamento', 'validação'], loc='upper left')
    plt.show()

    # Plotar gráfico de perda
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['treinamento', 'validação'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    train_dir = 'caminho_para_a_pasta_de_treinamento'
    test_dir = 'caminho_para_a_pasta_de_teste'
    valid_dir = 'caminho_para_pasta_de_validação'
    
    history = create_and_train_model(train_dir, test_dir, valid_dir)
    plot_accuracy_and_loss(history)
