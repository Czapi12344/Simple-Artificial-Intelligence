import numpy as np
import pickle


def checkoutput(number, goal, output, accuracy=0):
    print("------------------")
    if (np.round(goal, accuracy) == np.round(output, accuracy)).all():
        print(f'{number} - OK')
    else:
        print(f'{number} - ERROR')

    if output.size == 1:
        print(f'  OUTPUT -> {output}  GOAL -> {goal}')
    else:
        print(f'  OUTPUT -> {output.T}\n  GOAL   -> {goal.T}')

class NeuralNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, n=0, limits=(-0.5, 0.5)):

        if not self.layers:
            shape = (n, self.input_size)
        else:
            shape = (n,self.layers[- 1].shape[0])
        self.layers.append(np.random.uniform(limits[0], limits[1], shape).round(3))

    def save_weights(self, filename='model_layers'):
        with open(f'{filename}.pkl', 'wb') as file:
                pickle.dump(self.layers, file)

    def load_pickle(self):
        with open(f'{filename}.pkl', 'rb') as file:
            self.layers = pickle.load(file)

    #dopytać
    def error(self, output, target):
        # return  np.sum( np.power( np.subtract(output , target) , 2))
        return np.average( np.power( np.subtract(output , target) , 2))

    def delta(self , output, target, input):
        return 2 / output.size * (np.outer( np.subtract(output , target) , input))

    def weight_update(self , weight, delta, alpha):
        return weight - alpha * delta

    def predict(self, inputs):
        output = inputs
        for layer in self.layers:
            output = np.matmul(layer, output)

        return np.reshape(output, (-1, 1))


    def predictMultiple(self, inputs):
        outputs = np.zeros((self.layers[0].shape[0], inputs.shape[1]))
        for i in range(inputs.shape[1]):
            input_data = inputs[:, i].reshape(-1, 1)
            output = self.predict(input_data)
            outputs[:,i] = output.reshape(-1)
        return outputs



    def train(self, inputs, targets, learning_rate=0.01, epochs=1000 , mod = 100):

        for epoch in range( epochs+1):
            total_loss = 0.0
            for i in range(inputs.shape[1]):
                input_data = inputs[:, i].reshape(-1, 1)
                target = targets[:, i].reshape(-1, 1)

                output = self.predict(input_data)

                loss = self.error(output, target)

                deltas = self.delta(output , target , input_data)

                self.layers[0] = self.weight_update( self.layers[0], deltas , learning_rate)

                total_loss += loss
            if (epoch+1 ) % mod == 0 :
               print(f"Epoch { epoch+1 }: Output = {output.reshape(-1)} Total Error = {total_loss:.10f}")

        print("Training completed")


    def totalError(self , inputs , targets):
        total = 0
        for i in range(inputs.shape[1]):
            input_data = inputs[:, i].reshape(-1, 1)
            target = targets[:, i].reshape(-1, 1)

            output = self.predict(input_data)

            loss = self.error(output, target)
            total += loss
            print(f"Error {i+1} : {loss}")

        print(f"Total Error : {total}")

    def relu(self , x ):
        return max(x,0)

