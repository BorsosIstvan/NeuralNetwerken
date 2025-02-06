import numpy as np
import json

# Functie voor de sigmoid activatiefunctie
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Afgeleide van de sigmoid functie
def sigmoid_derivative(x):
    return x * (1 - x)

# Trainingsdata voor de 3x3 patronen
X = np.array([
    [0, 1, 0, 1, 1, 1, 0, 1, 0],  # + patroon (plus teken)
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # - patroon (min teken)
    [1, 0, 0, 1, 0, 0, 1, 0, 0],  # | patroon (verticale lijn)
    [0, 0, 1, 0, 1, 0, 1, 0, 0],  # / patroon (schuin omhoog)
    [1, 0, 0, 0, 1, 0, 0, 0, 1]   # \ patroon (schuin omlaag)
])

# Gewenste uitvoer voor de patronen
y = np.array([
    [1, 0, 0, 0, 0],  # + patroon
    [0, 1, 0, 0, 0],  # - patroon
    [0, 0, 1, 0, 0],  # | patroon
    [0, 0, 0, 1, 0],  # / patroon
    [0, 0, 0, 0, 1]   # \ patroon
])

# Initialiseer gewichten en biases
np.random.seed(42)  # Voor reproduceerbaarheid

# Gewichten tussen invoer en verborgen laag (9x6)
w_input_hidden = np.random.rand(9, 6)

# Gewichten tussen verborgen laag en uitvoerlaag (6x5)
w_hidden_output = np.random.rand(6, 5)

# Biasen voor de verborgen laag (6 neuronen)
b_hidden = np.random.rand(6)

# Biasen voor de uitvoerlaag (5 neuronen)
b_output = np.random.rand(5)

# Leersnelheid
learning_rate = 0.1

# Aantal iteraties (epochs)
epochs = 10000

# Train het netwerk
for epoch in range(epochs):
    total_error = 0

    # Voor elke invoer in de trainingdata
    for i in range(len(X)):
        # Haal de invoer en het gewenste patroon op
        x = X[i]
        desired_output = y[i]

        # Voer de berekeningen uit voor de verborgen laag (sigmoid activatie)
        hidden_input = np.dot(x, w_input_hidden) + b_hidden
        hidden_output = sigmoid(hidden_input)

        # Bereken de uitvoer van het netwerk (sigmoid activatie)
        output_input = np.dot(hidden_output, w_hidden_output) + b_output
        output = sigmoid(output_input)

        # Foutberekening
        error = desired_output - output
        total_error += np.sum(error ** 2)  # Voeg de fout toe voor deze invoer

        # Backpropagation: bereken de gradiënten en pas de gewichten aan
        # Gradiënt van de uitvoerlaag
        d_output = error * sigmoid_derivative(output)
        d_w_hidden_output = np.outer(hidden_output, d_output)
        d_b_output = d_output

        # Gradiënt van de verborgen laag
        d_hidden_output = np.dot(d_output, w_hidden_output.T) * sigmoid_derivative(hidden_output)
        d_w_input_hidden = np.outer(x, d_hidden_output)
        d_b_hidden = d_hidden_output

        # Update de gewichten en biases
        w_input_hidden += learning_rate * d_w_input_hidden
        w_hidden_output += learning_rate * d_w_hidden_output
        b_hidden += learning_rate * d_b_hidden
        b_output += learning_rate * d_b_output

    # Na elke epoch (iteratie), print de gemiddelde fout
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Gemiddelde Fout: {total_error / len(X)}")

# Zet de gewichten en biases om naar een dictionary
model_data = {
    "w_input_hidden": w_input_hidden.tolist(),
    "w_hidden_output": w_hidden_output.tolist(),
    "b_hidden": b_hidden.tolist(),
    "b_output": b_output.tolist()
}

# Sla de modeldata op in een JSON-bestand
with open("trained_model.json", "w") as json_file:
    json.dump(model_data, json_file)

print("\nNetwerk is opgeslagen in JSON-formaat!")
