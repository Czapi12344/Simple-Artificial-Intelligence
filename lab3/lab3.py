import numpy as np
import NeuralNetwork as NN

def read_labels(filename: str):

    labels = []

    with open(filename, "rb") as file:
        file.read(4)
        length = int.from_bytes(file.read(4), "big")

        for i in range(0, length):
            label = file.read(1)
            arr = np.zeros((10, 1))
            number = int.from_bytes(label, "little")
            arr[number] = 1
            labels.append(arr)

        return np.concatenate(labels, axis=1)


def read_images(filename: str):
    with open(filename, "rb") as file:
        xd = file.read(4)
        length = int.from_bytes(file.read(4), "big")
        rows = int.from_bytes(file.read(4), "big")
        cols = int.from_bytes(file.read(4), "big")

        image_data = []
        for i in range(0, length):
            img = np.frombuffer(file.read(rows*cols), dtype=np.uint8) / 255
            image_data.append(img.reshape(-1, 1))

        return np.concatenate(image_data, axis=1)

def read_file(filename):
    inputs = []
    goals = []

    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_split = line.split()
            inputs.append([float(line_split[0]), float(line_split[1]), float(line_split[2])])
            goal = np.zeros((4, 1))
            goal[int(line_split[3]) - 1] = 1
            goals.append(goal)

    return np.array(inputs).reshape(len(lines), 3).T, np.concatenate(goals, axis=1)


trainIn, trainGoal = read_file('training_colors.txt')
testIn, testGoal = read_file('test_colors.txt')
trainIn  = trainIn.T
trainGoal = trainGoal.T
testIn = testIn.T
testGoal =testGoal.T

images = read_images("train-images.idx3-ubyte").T
labels = read_labels("train-labels.idx1-ubyte").T
test_images = read_images("train-images.idx3-ubyte").T
test_labels = read_labels("train-labels.idx1-ubyte").T



w_h = np.asarray([0.1 , 0.1 , -0.3 , 0.1 , 0.2, 0.0 , 0.0 ,0.7 ,0.1, 0.2 , 0.4, 0.0 ,-0.3 , 0.5 , 0.1])
w_h = w_h.reshape(5,3)



w_y = np.asarray([0.7 , 0.9 , -0.4 , 0.8 , 0.1 , 0.8 , 0.5 , 0.3 , 0.1 ,0.0 , -0.3 , 0.9 , 0.3 , 0.1 ,-0.2])
w_y = w_y.reshape(3,5)

input_x = np.array([0.5, 0.1, 0.2, 0.8, 0.75, 0.3, 0.1, 0.9, 0.1, 0.7, 0.6, 0.2]).reshape(3,4).T

target_y =  np.array([0.1, 0.5, 0.1, 0.7, 1.0, 0.2, 0.3, 0.6, 0.1, -0.5, 0.2, 0.2]).reshape(3,4).T



nn = NN.NeuralNetwork(input_size=3)
nn.add_layer(n=5, activation_function=NN.relu ,weight= w_h )
nn.add_layer(n=3, activation_function=None , weight= w_y )


alpha = 0.01
epochs = 50

nn.train( input_x, target_y , learning_rate=alpha, epochs=epochs, mod=1)
nn.calculate_accuracy(input_x, target_y)



#
#
# print("Start")
#
# net = NN.NeuralNetwork(input_size= 784)
# net.add_layer(40, activation_function= NN.relu)
# net.add_layer(10)
# net.train(inputs=images, targets=labels, learning_rate=0.01, epochs=1000000, mod=1 ,filename= "zad3.3" , savemod= 100)

# #
# print("Start2")

# net2 = NN.NeuralNetwork(input_size= 3 )
# net2.add_layer(200 ,activation_function= NN.relu )
# net2.add_layer(4)
# net2.train(inputs=trainIn, targets=trainGoal, learning_rate=0.1, epochs=10, mod=1 ,filename="zad3.4")

