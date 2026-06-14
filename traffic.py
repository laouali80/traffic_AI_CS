import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])


    # print(type(labels))
    # print(len(labels))
    # print(labels[:10])

    # # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # print("labels shape................:", labels.shape)
    # # Get a compiled neural network
    model = get_model()

    # print("x_train:", x_train.shape)
    # print("y_train:", y_train.shape)

    # print("First label:")
    # print(y_train[0])

    # # # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # # # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # # # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # open data directory and read in subdirectories

    images = []
    labels = []

    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory '{data_dir}' does not exist or is not a directory.")
    else:
        # print('Files in data directory:', os.listdir(data_dir))
        # i=0

        for category in os.listdir(data_dir):
            categoty_path = os.path.join(data_dir, category)

            
            # print(f"Processing category '{category}' at path '{categoty_path}'")

            # allow to skip the first directory of os.listdir(data_dir) .DS_Store ['.DS_Store', '0', '1', '2']
            if not os.path.isdir(categoty_path):
                # print(f"Skipping '{categoty_path}' since it's not a directory.")
                continue

            # loop through the subdirectory and read in image files
            for filename in os.listdir(categoty_path):
                if not filename.endswith(".ppm"):
                    continue
                
                img_path = os.path.join(categoty_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read image '{img_path}'. Skipping.")
                    continue

                # cv2.resize() function is used to resize the image to the specified dimensions (IMG_WIDTH, IMG_HEIGHT).
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                images.append(img)
                labels.append(int(category))

                # print(f"Image numpy array of [R,G,B]: {img}")


                # Image shape (height, width, channels): (30, 30, 3)
                # with channels representing the color channels (red, green, blue) of the image
                # print(f"Image shape (height, width, channels): {img.shape}")
                
                # if i>2:
                #     print("tuples:", (images, labels))
                #     break
                # i += 1
    
    # [https://chatgpt.com/s/t_6a2ebc5bf5408191beef81a0586f7d63]
    # The key is to think of the image as a grid of pixels.
    
    # If an image has shape: (30, 30, 3)

    # it means: (height, width, channels)

    # or:

    # 30 rows of pixels
    # 30 columns of pixels
    # 3 color values (R, G, B) per pixel
    
    # Visualizing the structure

    # An image is stored like this:

    # image[
    #     row
    # ][
    #     column
    # ][
    #     channel
    # ]

    # image = [
    #     row0,
    #     row1,
    #     row2,
    #     ...
    #     row29
    # ]


    # For example: image[5][10] means the pixel at row 5 and column 10 of the image. This pixel will have three color values (R, G, B) that can be accessed as follows:

    # might return: [245, 250, 253] meaning that the pixel at row 5 and column 10 has a red value of 245, a green value of 250, and a blue value of 253.

    # print(f"Images shape: {len(images)} images, each of shape {images[0].shape} (height, width, channels)")
    # print(f"Labels shape: {len(labels)} labels")

    
    
    return (images, labels)





def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

         # =========================================================
        # 1. INPUT CONVOLUTIONAL BLOCK
        # =========================================================

        # Convolutional layer. Learning xx filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(3, 3), # The kernel size determines how much of the image the filter looks at at one time (its receptive field).
            activation="relu", 
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        
        # [https://chatgpt.com/s/t_6a2ebc1ccd908191a6f016334ec164bf]
        # # # Filter Explanation:
        # filters=32 means that the convolutional layer will learn 32 different filters (or kernels) during training. Each filter will be able to detect different features in the input images, such as edges, textures, or patterns. The more filters you have, the more complex features the model can learn to recognize.
        # For example:

        # Filter 1 → vertical edges
        # Filter 2 → horizontal edges
        # Filter 3 → diagonal edges
        # Filter 4 → color transitions
        # ...
        # Filter 32 → more complex patterns

        # # # Kernel Size Explanation:
        # kernel_size=(3, 3) means that the convolutional layer will use a 3x3 filter to scan over the input image. This filter will look at a 3x3 grid of pixels at a time as it moves across the image, allowing the model to learn local patterns and features in the image.
        
        # Intuition determining the kernel size:
        
        # 2×2 kernel
        # Very small view.
        # Good for detecting extremely local patterns, but may miss larger structures.

        # 3×3 kernel
        # The most common choice.
        # Large enough to detect edges, corners, and small shapes while keeping the number of parameters low.
        # This is why architectures like VGGNet relied heavily on 3×3 convolutions.

        # 5×5 kernel
        # Sees a larger area immediately.
        # Can capture larger patterns but uses more parameters and computation.
        
        # Input shape Explanation:
        # input_shape=(IMG_WIDTH, IMG_HEIGHT, 3) specifies the shape of the input images that the model will receive. In this case, the images are expected to have a width of IMG_WIDTH pixels, a height of IMG_HEIGHT pixels, and 3 color channels (for red, green, and blue). This information is crucial for the model to understand the structure of the input data.



        # Pooling Layer.
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        # [https://chatgpt.com/s/t_6a2ebbbb807c8191b3eb9dafe3032702]
        # # # Pooling Explanation:
        # Pooling is used to reduce the size of an input by sampling from regions in the input.

        # # # Max Pooling vs Average Pooling:
        
        # Max Pooling: Takes the maximum value from each region. 
        # Average Pooling: Takes the average value from each region.

        # # # Pool Size Explanation:
        # 2 x 2 pooling : most common choice.
        # 3 x 3 pooling : captures more context but reduces spatial dimensions more aggressively.
        # 4 x 4 pooling : very aggressive downsampling, may lose important details.

        
        # =========================================================
        # 2. SECOND CONVOLUTIONAL BLOCK (HIGHER-LEVEL FEATURES)
        # =========================================================


        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),

         # - More filters (64) → more complex feature detection
        # - Builds on earlier features (edges → shapes → parts)

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

         # =========================================================
        # 3. TRANSITION TO DENSE LAYERS
        # =========================================================

        # Flatten units.
        tf.keras.layers.Flatten(),
        
        # Flatten:
        # - Converts 3D feature maps → 1D vector
        # - Required before fully connected layers


        # =========================================================
        # 4. FULLY CONNECTED (HIDDEN) LAYER
        # =========================================================

        # Addition of hidden layer.
        tf.keras.layers.Dense(128, activation="relu"),
        # # # # Hidden Layer Explanation:
        # A hidden layer is a layer of neurons in a neural network that sits between the input layer and the output layer. It allows the network to learn complex representations of the input data. The number of neurons in the hidden layer (in this case, 128) determines the capacity of the model to learn patterns.

        # # # # Activation Function Explanation:
        # The activation function introduces non-linearity into the model, allowing it to learn complex patterns. The ReLU (Rectified Linear Unit) activation function is commonly used in hidden layers because it helps the model learn faster and reduces the likelihood of vanishing gradients.

        # Dropout configuration to prevent overfitting.
        tf.keras.layers.Dropout(0.5),

         # Dropout:
        # - Randomly disables 50% of neurons during training
        # - Prevents overfitting
        # - Forces network to generalize better


        # =========================================================
        # 5. OUTPUT LAYER
        # =========================================================

        # Output layer with output units for all categories.
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

        # # # # Output Layer Explanation:
        # The output layer has NUM_CATEGORIES units, one for each category of traffic sign.

        # # # # Softmax Activation Explanation:
        # The softmax activation function is used in the output layer to convert the raw output scores (logits) into probabilities that sum to 1. Each output unit will represent the probability of the input image belonging to a specific category. The category with the highest probability will be the model's prediction.
    ])

     # =========================================================
    # 6. COMPILATION (HOW MODEL LEARNS)
    # =========================================================

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # [https://chatgpt.com/s/t_6a2ebb6673108191841b81f5805a14e2]
    # # # 1. Optimizer

    # The optimizer determines how the weights are updated during training.

    # Remember, every filter weight and neuron weight starts as a random number.

    # Example:

    # Weight = 0.53

    # After seeing training data, the network calculates:

    # How wrong was I?

    # The optimizer decides:

    # How should I change the weights to reduce the error?
    # Adam (most common)
    # optimizer='adam'

    # Adam automatically adjusts learning rates and generally works well without much tuning.

    # For most image-classification projects, this is the default choice.

    # SGD
    # optimizer='sgd'

    # Stands for:

    # Stochastic Gradient Descent

    # Updates weights more directly.

    # Example:

    # optimizer=tf.keras.optimizers.SGD(
    #     learning_rate=0.01
    # )
    # RMSprop
    # optimizer='rmsprop'

    # Often used for sequential or recurrent models, but can work for CNNs too.

    # # # 2. Loss Function

    # The loss function measures:

    # How wrong is the model?

    # Training is essentially:

    # Predict
    # ↓
    # Calculate loss
    # ↓
    # Adjust weights
    # ↓
    # Repeat
    # Example: Digit classification

    # Suppose there are 10 classes:

    # 0 1 2 3 4 5 6 7 8 9

    # True label:

    # 3

    # Model prediction:

    # [0.01,0.02,0.05,0.60,0.10,0.05,0.03,0.04,0.05,0.05]

    # The model thinks:

    # Class 3 = 60%

    # Since class 3 is correct, loss is relatively low.

    # If it predicted class 8 with high confidence, loss would be much larger.

    # Common loss functions
    # sparse_categorical_crossentropy

    # Most common for multiclass classification.

    # Use when labels are integers:

    # 0
    # 1
    # 2
    # 3
    # ...

    # Example:

    # y_train = [0, 2, 5, 1]

    # Compile:

    # loss='sparse_categorical_crossentropy'
    # categorical_crossentropy

    # Use when labels are one-hot encoded.

    # Instead of:

    # 3

    # you have:

    # [0,0,0,1,0,0,0,0,0,0]

    # Compile:

    # loss='categorical_crossentropy'
    # binary_crossentropy

    # For two classes only.

    # Example:

    # Cat
    # Dog

    # or

    # 0
    # 1

    # Compile:

    # loss='binary_crossentropy'
    
    
    # # # 3. Metrics

    # Metrics are what you want TensorFlow to report while training.

    # They do not control learning.

    # They only help you monitor performance.

    # Accuracy

    # Most common.

    # metrics=['accuracy']

    # Example:

    # 100 images tested
    # 92 correct

    # Accuracy = 92%

    # TensorFlow might print:

    # loss: 0.21
    # accuracy: 0.92
    # Precision

    # Useful when false positives matter.

    # metrics=['Precision']
    # Recall

    # Useful when false negatives matter.

    # metrics=['Recall']

    return model


if __name__ == "__main__":
    main()
