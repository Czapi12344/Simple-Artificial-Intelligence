import numpy as np
import NeuralNetwork as NN

# ZAD 2.1

print("\nZAD 1\n")

x = np.array([2])
x = np.reshape(x, (1, 1))
y = np.array([0.8])
y = np.reshape(y, (1, 1))

nn = NN.NeuralNetwork(input_size=1)

nn.add_layer(n=1, limits=(0.5, 0.5))
nn.train( x, y, learning_rate=0.1, epochs=20  , mod= 5 )



# ZAD 2.2
print("\nZAD 2\n ")
nn = NN.NeuralNetwork(input_size=3)
nn.add_layer(n=5, limits=(-0.5, 0.5))

##nn.load_pickle()

x = [0.5, 0.1, 0.2, 0.8, 0.75, 0.3, 0.1, 0.9, 0.1, 0.7, 0.6, 0.2]
x = np.reshape(x, (3, 4))

y = [0.1, 0.5, 0.1, 0.7, 1.0, 0.2, 0.3, 0.6, 0.1, -0.5, 0.2, 0.2, 0.0, 0.3, 0.9, -0.1, -0.1, 0.7, 0.1, 0.8]
y = np.reshape(y, (5, 4))

wh = [0.1, 0.1, -0.3, 0.1, 0.2, 0.0, 0.0, 0.7, 0.1, 0.2, 0.4, 0.0, -0.3, 0.5, 0.1]
wh = np.reshape(wh, (5, 3))
nn.layers[0] = wh
nn.train( x, y , learning_rate=0.01, epochs=999 ,mod= 100)
nn.totalError(x,y)


print("\nZAD 3\n ")

# zad 2.3

def read_values_from_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        length = len(lines)

        file_inputs = np.zeros((3, length))

        file_goal = []

        for k in range(0, length):
            matrix = np.genfromtxt(fname=filename, delimiter=' ', skip_header=k, max_rows=1)
            file_inputs[:, k] = matrix[0:-1]
            file_goal.append(int(matrix[-1]))

        return file_inputs, file_goal


inputs, labels = read_values_from_file("training_colors.txt")
test_inputs, test_labels = read_values_from_file("test_colors.txt")

goal = np.zeros((4, len(labels)))
for i in range(len(labels)):
    goal[labels[i] - 1, i] = 1

nn = NN.NeuralNetwork(3)
nn.add_layer(4)

output = nn.predictMultiple(test_inputs)
errors = 0

for i in range(0, len(test_labels)):
    index = np.argmax(output[:, i])
    if index + 1 != test_labels[i]:
        errors += 1

print(f"Accur={(len(test_labels) - errors) / len(test_labels) * 100:.2f}%")

nn.train(inputs, goal , 0.01, 100)

output = nn.predictMultiple(test_inputs)
errors = 0

for i in range(0, len(test_labels)):
    index = np.argmax(output[:, i])
    if index + 1 != test_labels[i]:
        errors += 1

print(f"Accur={(len(test_labels) - errors) / len(test_labels) * 100:.2f}%")