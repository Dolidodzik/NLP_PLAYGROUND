import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_dataset_filename = "TRAIN_REGRESSION_OF_FUNCTION_a=1_b=-2_c=-8_d=10.csv"
validation_dataset_filename = "VALIDATION_REGRESSION_OF_FUNCTION_a=1_b=-2_c=-8_d=10.csv"

# Loading train dataset
train_dataset_df = pd.read_csv(train_dataset_filename)
train_dataset_length = len(train_dataset_df)
train_x_values = np.array(train_dataset_df['x'].tolist())
train_y_values = np.array(train_dataset_df['y'].tolist())

print(f"Loaded training dataset from {train_dataset_filename}:")
print(f"Training dataset length: {train_dataset_length}")

# Loading validation dataset
validation_dataset_df = pd.read_csv(validation_dataset_filename)
validation_dataset_length = len(validation_dataset_df)
val_x_values = np.array(validation_dataset_df['x'].tolist())
val_y_values = np.array(validation_dataset_df['y'].tolist())

print(f"Loaded validation dataset from {validation_dataset_filename}:")
print(f"Validation dataset length: {validation_dataset_length}")

# Initializing parameters randomly
params = {
    'a': 11,
    'b': 11,
    'c': 11,
    'd': 11
}

# HYPER PARAMETERS
learning_rate = 0.001
epochs = 10

# Mean Squared Error (MSE) Loss function
def mse_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def train_epoch(params):
    total_loss = 0
    for i in range(train_dataset_length):
        x = train_x_values[i]
        y_true = train_y_values[i]
        
        # Predict the output based on the current parameters
        y_pred = params['a'] * x**3 + params['b'] * x**2 + params['c'] * x + params['d']
        
        # Calculate the loss
        loss = mse_loss(np.array([y_true]), np.array([y_pred]))
        total_loss += loss

        # adjust a,b,c,d using gradients
        common_factor = -2 * (y_true - y_pred)
        gradient_a = common_factor * x**3
        gradient_b = common_factor * x**2
        gradient_c = common_factor * x
        gradient_d = common_factor * 1

        new_params = {
            'a': params['a'] - learning_rate * gradient_a,
            'b': params['b'] - learning_rate * gradient_b,
            'c': params['c'] - learning_rate * gradient_c,
            'd': params['d'] - learning_rate * gradient_d
        }

        new_y_pred = new_params['a'] * x**3 + new_params['b'] * x**2 + new_params['c'] * x + new_params['d']
        new_loss = mse_loss(np.array([y_true]), np.array([new_y_pred]))
        
        if new_loss < loss:
            params = new_params
        else:
            pass
            #print("gradient adjustment resulted in higher loss! we might be in a valley, or the learning rate is too big!")

    # Return the average loss for the epoch
    avg_loss = total_loss / train_dataset_length
    return avg_loss, params

def validate(params):
    total_loss = 0
    for i in range(validation_dataset_length):
        x = val_x_values[i]
        y_true = val_y_values[i]

        y_pred = params['a'] * x**3 + params['b'] * x**2 + params['c'] * x + params['d']

        loss = mse_loss(np.array([y_true]), np.array([y_pred]))
        total_loss += loss
    avg_loss = total_loss / validation_dataset_length
    return avg_loss


def plot_graph(params):
    x_range = np.linspace(min(train_x_values.min(), val_x_values.min()),
                        max(train_x_values.max(), val_x_values.max()), 500)

    y_predicted = (params['a'] * x_range**3 +
                params['b'] * x_range**2 +
                params['c'] * x_range +
                params['d'])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(train_x_values, train_y_values, color='orange', label='Training Data')
    plt.scatter(val_x_values, val_y_values, color='red', label='Validation Data')
    plt.plot(x_range, y_predicted, color='green', linewidth=2, label='Learned Function')

    plt.title("Cubic Function Fit to Training and Validation Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    current_params_validation_loss = validate(params)
    for epoch in range(epochs):
        if epoch % 1000 == 0:
            print(f"epoch {epoch} out of {epochs}")
        #print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, potential_new_params = train_epoch(params)
        potential_new_params_validation_loss = validate(potential_new_params)

        print(f"Training Loss: {train_loss}, Parameters: {params}")
        print(f"New potential params validation Loss: {potential_new_params_validation_loss}, Parameters: {params}")
        print(f"Current params validation loss")

        if current_params_validation_loss > potential_new_params_validation_loss:
            print("found new better params!! ", potential_new_params)
            current_params_validation_loss = potential_new_params_validation_loss
            params = potential_new_params
        else:
            print("We haven't found better values over entire epoch! Something might be wrong!!!")

        print("====================================================")
        print("================ EPOCH ENDED =======================")
        print("====================================================")

    print("\n\n================================================================================")
    print("final params: ", params)
    print("final vaildation loss: ", current_params_validation_loss)
    print("\n\n")
    
    plot_graph(params)

        