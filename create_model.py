import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import csv


def load_training_data(filename):
    game_states = []
    optimal_moves = []

    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            game_state = np.array(eval(row[0]))
            optimal_move = np.array(eval(row[1]))

            game_state_flat = game_state.flatten()
            optimal_move_flat = optimal_move.flatten()

            diff_indices = np.where(game_state_flat != optimal_move_flat)[0]

            move_index = diff_indices[0]
            game_states.append(game_state_flat)
            optimal_moves.append(move_index)

    return np.array(game_states), np.array(optimal_moves)


game_states, optimal_moves = load_training_data("training_data.csv")

game_states = game_states.reshape(game_states.shape[0], 9)

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(9,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

optimal_moves_categorical = to_categorical(optimal_moves, num_classes=9)

model.fit(game_states, optimal_moves_categorical, epochs=300, batch_size=32)

model.save("game_ai_model.keras")
