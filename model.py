import tensorflow as tf

def create_and_save_model():
    """
    Creates a simple CNN model and saves it as 'cifar10_model.h5'.
    
    This function is for demonstration purposes only. In a real project, you
    would replace this with your actual, trained model.
    """
    print("Creating a mock model file...")
    
    # Define a simple CNN architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save('cifar10_model.h5')
    print("Model saved as 'cifar10_model.h5'.")

if __name__ == "__main__":
    create_and_save_model()