import numpy as np
import json

# Functie voor de sigmoid activatiefunctie
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Functie voor het testen van een handmatige invoer
def test_pattern(input_matrix, w_input_hidden, w_hidden_output, b_hidden, b_output):
    # Zet de matrix om naar een vector van 9 elementen
    x = np.array(input_matrix).flatten()

    # Bereken de uitvoer van het netwerk
    hidden_output = sigmoid(np.dot(x, w_input_hidden) + b_hidden)
    output = sigmoid(np.dot(hidden_output, w_hidden_output) + b_output)

    # Normaliseer de uitvoer tussen 0 en 1
    output = np.round(output, 2)  # Ronden voor beter zicht (optioneel)

    # Zoek de hoogste waarde in de uitvoer
    predicted_pattern = np.argmax(output)

    # Print de invoer en herkend patroon als het herkend is
    if np.max(output) >= 0.5:  # drempelwaarde, kan worden aangepast
        print(f"Invoer: \n{input_matrix}")
        print(f"Netwerkuitvoer (genormaliseerd): {output}")
        print(f"Herkenning: {['plus', 'min', 'lijn', 'per', 'slash'][predicted_pattern]}")
        print("-----------")
    else:
        print("Geen herkenning met hoge zekerheid.")
        print("-----------")

# Laad de opgeslagen modeldata van het JSON-bestand
with open("trained_model.json", "r") as json_file:
    model_data = json.load(json_file)

# Laad de gewichten en biases
w_input_hidden = np.array(model_data["w_input_hidden"])
w_hidden_output = np.array(model_data["w_hidden_output"])
b_hidden = np.array(model_data["b_hidden"])
b_output = np.array(model_data["b_output"])

# Test de getrainde netwerkinvoer met een handmatige invoer
print("Test handmatige invoer:\n")

# Vul hier je 3x3 matrix in (bijvoorbeeld een plus teken '+')
handmatige_invoer = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
]

test_pattern(handmatige_invoer, w_input_hidden, w_hidden_output, b_hidden, b_output)
