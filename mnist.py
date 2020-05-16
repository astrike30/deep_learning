import numpy as np
from nn import NeuralNetModel
import matplotlib.pyplot
import pickle


f = open("MNIST/mnist_train.csv")
data_list = f.readlines()
f.close()

train_images = []
for image in data_list:
    train_images.append([int(pixel) for pixel in image.split(",")])

f = open("MNIST/mnist_test.csv")
data_list = f.readlines()
f.close()

test_images = []
for image in data_list:
    test_images.append([int(pixel) for pixel in image.split(",")])


def show_digit(pixels):
    image_arr = np.asfarray(pixels[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_arr, cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()


def prep_data(images):
    scaled_inputs = []
    targets = []
    for (i, image) in enumerate(images):
        scaled_inputs.append((np.asfarray(image[1:]) / 255 * 0.99) + 0.01)

        targets.append(np.zeros(10) + 0.01)
        digit = int(image[0])
        targets[i][digit] = 0.99
    return scaled_inputs, targets


def test_model(nn):
    tests, labels = prep_data(test_images)

    correct_count = 0.0
    for i, test in enumerate(tests):
        result = nn.query([test])
        index = np.argmax(result, axis=1)[0]
        correct = np.argmax(labels[i], axis=0)
        if correct == index:
            correct_count += 1
    return correct_count / i


if __name__ == "__main__":

    inputs, targets = prep_data(train_images)

    input_nodes = 784
    hidden_nodes = [5, 5, 5, 5, 5]
    outputs = 10
    nn = NeuralNetModel(input_nodes, hidden_nodes, outputs, 0.1)

    epochs = 20
    for i in range(epochs):
        print("Training Epoch: {}/{}".format((i + 1), epochs))
        for i, digit in enumerate(inputs):
            nn.train(digit, [targets[i]])

    print("Accuracy of Model: {}%".format(test_model(nn) * 100))
