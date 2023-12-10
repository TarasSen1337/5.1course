import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def test1():

    # Define constants
    input_shape = (300, 300, 3)  # Adjust the image size as needed
    batch_size = 20
    epochs = 10

    # Define the list of artist names (folders containing their artwork)
    artist_names = ["picasso", "dali", "gogh"]  # Add more artist folders as needed

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load and prepare the training data
    train_generator = train_datagen.flow_from_directory(
        "dataset/train",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        classes=artist_names,
        subset='training'
    )

    # Load and prepare the test data
    validation_generator = test_datagen.flow_from_directory(
        "dataset/train",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        classes=artist_names,
        subset='validation'
    )

    # Evaluate the model on test data
    test_generator = test_datagen.flow_from_directory(
        "dataset/test",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        classes=artist_names
    )

    # Build the neural network model
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(artist_names), activation='softmax')  # Multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    # Save the model
    # model.save("artist_classification_model.h5")

    # Predict a sample image (replace with your test image)
    sample_image_path = "test_image.jpg"  # Replace with the path to your test image
    sample_image = keras.preprocessing.image.load_img(
        sample_image_path, target_size=input_shape[:2]
    )
    sample_image = keras.preprocessing.image.img_to_array(sample_image)
    sample_image = np.expand_dims(sample_image, axis=0)
    predictions = model.predict(sample_image)

    predicted_artist_index = np.argmax(predictions)
    predicted_artist = artist_names[predicted_artist_index]

    print(f'The image is likely from the artist: {predicted_artist}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test1()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
