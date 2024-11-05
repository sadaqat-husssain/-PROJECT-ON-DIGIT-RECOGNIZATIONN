# -PROJECT-ON-DIGIT-REGOCNOZATOPN
Auther:sadaqat hussain
<br>
This is my project on digit recognization and i have made a model on which the the model will will predict the digit 
<br>
This code is run on google colab and the data set is SVHN and load directly from internet
<br>

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import requests
from io import BytesIO

# Function to load SVHN dataset directly from the internet
def load_svhn_data(url, num_samples):
    response = requests.get(url)
    data = loadmat(BytesIO(response.content))
    X = np.transpose(data['X'], (3, 0, 1, 2))  # Transpose to (num_samples, height, width, channels)
    y = data['y'].flatten()
    y[y == 10] = 0  # Convert label '10' to '0'
    
    # Limit to num_samples
    X = X[:num_samples]
    y = y[:num_samples]
    
    # Normalize pixel values
    X = X / 255.0
    
    return X, y




# Load 3000 training images and 1000 testing images
train_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
test_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
X_train, y_train = load_svhn_data(train_url, 3000)
X_test, y_test = load_svhn_data(test_url, 1000)
#now from here you can see the model
 Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=20,
    steps_per_epoch=len(X_train) // 64,
    verbose=1
)
#now this part will show you the output you can change according to your need like  i have predict the 3 digit .

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def generate_digit_image(digit, img_size=32, font_size=24):
    """
    Generates an image of a given digit.
    
    Parameters:
    digit (int): The digit to display in the image (0-9).
    img_size (int): The size of the image (default is 32x32).
    font_size (int): Font size for the digit.
    
    Returns:
    numpy array: Generated image as a numpy array.
    """
    # Create a blank image with white background
    image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        # Use a default font (or specify a path to a TTF font if available)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Use default font if arial.ttf isn't available
    
    # Center the digit on the image
    text = str(digit)
    # Use textbbox to get bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)  
    text_width = text_bbox[2] - text_bbox[0] # Calculate width
    text_height = text_bbox[3] - text_bbox[1] # Calculate height
    position = ((img_size - text_width) // 2, (img_size - text_height) // 2)
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Convert the image to a numpy array and normalize pixel values
    image_np = np.array(image) / 255.0
    
    return image_np

def predict_generated_digit(model, digit):
    """
    Generates an image of a digit, then predicts the digit using the model.
    
    Parameters:
    model: Trained CNN model
    digit (int): The digit to generate and predict
    
    Returns:
    int: The model's predicted digit
    """
    # Generate the digit image
    image_np = generate_digit_image(digit)
    image_np = np.expand_dims(image_np, axis=0)  # Reshape to (1, 32, 32, 3)
    
    # Make a prediction
    prediction = model.predict(image_np)
    predicted_digit = np.argmax(prediction)
    
    # Display the generated image and prediction
    plt.imshow(image_np[0])
    plt.title(f"Generated Digit: {digit} | Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()
    
    return predicted_digit

# Example usage
input_digit = 3 # This is the digit you'd like to test
predicted_digit = predict_generated_digit(model, input_digit)
print(f"Predicted Digit: {predicted_digit}")





