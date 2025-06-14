try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.datasets import mnist
    from utils.visualization import visualize_predictions
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please make sure all dependencies are installed using:")
    print("pip install -r requirements.txt")
    exit(1)

def evaluate_model(model_path):
    # Load the trained model
    model = load_model(model_path)

    # Load the MNIST test dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess the data
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_accuracy:.4f}')

    # Visualize predictions on 5 sample images
    sample_indices = np.random.choice(len(x_test), 5, replace=False)
    sample_images = x_test[sample_indices]
    sample_labels = y_test[sample_indices]
    predictions = model.predict(sample_images)

    # Visualize the results
    visualize_predictions(sample_images, sample_labels, predictions)

if __name__ == "__main__":
    model_path = 'models/mnist_cnn_model.h5'  # Update this path
    evaluate_model(model_path)