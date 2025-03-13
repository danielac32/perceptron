#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Estructura para definir una muestra de distancia y acción
typedef struct {
    double distance;  // Distancia medida
    int action;       // Acción (0: avanzar, 1: retroceder, 2: girar izquierda, 3: girar derecha)
} DistanceSample;

// Lista de acciones predefinidas
const char *action_names[] = {"Avanzar", "Retroceder", "Girar a la izquierda", "Girar a la derecha"};

// Lista de datos de entrenamiento (global)
DistanceSample global_training_data[] = {
    {10.0, 1},  // Si la distancia es 10 cm, retroceder
    {15.0, 1},  // Si la distancia es 15 cm, retroceder
    {20.0, 2},  // Si la distancia es 20 cm, girar a la izquierda
    {25.0, 2},  // Si la distancia es 25 cm, girar a la izquierda
    {30.0, 3},  // Si la distancia es 30 cm, girar a la derecha
    {35.0, 3},  // Si la distancia es 35 cm, girar a la derecha
    {40.0, 0},  // Si la distancia es 40 cm, avanzar
    {45.0, 0},  // Si la distancia es 45 cm, avanzar
    {50.0, 0},  // Si la distancia es 50 cm, avanzar
    {55.0, 0},  // Si la distancia es 55 cm, avanzar
    {60.0, 0},  // Si la distancia es 60 cm, avanzar
    {70.0, 0}   // Si la distancia es 70 cm, avanzar
};

// Tamaño de la lista de datos de entrenamiento
const int global_training_data_size = sizeof(global_training_data) / sizeof(DistanceSample);

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (1 para distancia)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (4 para acciones)
} MLP;

// Función de activación (ReLU)
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivada de la función ReLU
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Función de activación (Softmax)
void softmax(double *x, int size) {
    double max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Inicializar la red neuronal
MLP* create_mlp(int input_size, int hidden_size, int output_size) {
    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    mlp->input_size = input_size;
    mlp->hidden_size = hidden_size;
    mlp->output_size = output_size;

    // Inicializar pesos y biases
    srand(time(NULL));

    mlp->weights_input_hidden = (double**)malloc(input_size * sizeof(double*));
    for (int i = 0; i < input_size; i++) {
        mlp->weights_input_hidden[i] = (double*)malloc(hidden_size * sizeof(double));
        for (int j = 0; j < hidden_size; j++) {
            mlp->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / input_size);  // Inicialización de He
        }
    }

    mlp->weights_hidden_output = (double**)malloc(hidden_size * sizeof(double*));
    for (int i = 0; i < hidden_size; i++) {
        mlp->weights_hidden_output[i] = (double*)malloc(output_size * sizeof(double));
        for (int j = 0; j < output_size; j++) {
            mlp->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / hidden_size);  // Inicialización de He
        }
    }

    mlp->bias_hidden = (double*)malloc(hidden_size * sizeof(double));
    for (int i = 0; i < hidden_size; i++) {
        mlp->bias_hidden[i] = 0.0;  // Inicializar biases a 0
    }

    mlp->bias_output = (double*)malloc(output_size * sizeof(double));
    for (int i = 0; i < output_size; i++) {
        mlp->bias_output[i] = 0.0;  // Inicializar biases a 0
    }

    return mlp;
}

// Liberar memoria de la red neuronal
void free_mlp(MLP *mlp) {
    for (int i = 0; i < mlp->input_size; i++) {
        free(mlp->weights_input_hidden[i]);
    }
    free(mlp->weights_input_hidden);

    for (int i = 0; i < mlp->hidden_size; i++) {
        free(mlp->weights_hidden_output[i]);
    }
    free(mlp->weights_hidden_output);

    free(mlp->bias_hidden);
    free(mlp->bias_output);
    free(mlp);
}

// Predecir la salida para una entrada dada
void predict_mlp(MLP *mlp, double distance, double *output) {
    double input[1] = {distance};
    double hidden[mlp->hidden_size];

    // Capa oculta
    for (int i = 0; i < mlp->hidden_size; i++) {
        hidden[i] = 0.0;
        for (int j = 0; j < mlp->input_size; j++) {
            hidden[i] += input[j] * mlp->weights_input_hidden[j][i];
        }
        hidden[i] += mlp->bias_hidden[i];
        hidden[i] = relu(hidden[i]);
    }

    // Capa de salida
    for (int i = 0; i < mlp->output_size; i++) {
        output[i] = 0.0;
        for (int j = 0; j < mlp->hidden_size; j++) {
            output[i] += hidden[j] * mlp->weights_hidden_output[j][i];
        }
        output[i] += mlp->bias_output[i];
    }

    // Aplicar softmax
    softmax(output, mlp->output_size);
}

// Entrenar la red neuronal
void train_mlp(MLP *mlp, DistanceSample training_data[], int data_size, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < data_size; i++) {
            double output[mlp->output_size];
            double hidden[mlp->hidden_size];

            // Forward pass
            for (int j = 0; j < mlp->hidden_size; j++) {
                hidden[j] = 0.0;
                for (int k = 0; k < mlp->input_size; k++) {
                    hidden[j] += training_data[i].distance * mlp->weights_input_hidden[k][j];
                }
                hidden[j] += mlp->bias_hidden[j];
                hidden[j] = relu(hidden[j]);
            }

            for (int j = 0; j < mlp->output_size; j++) {
                output[j] = 0.0;
                for (int k = 0; k < mlp->hidden_size; k++) {
                    output[j] += hidden[k] * mlp->weights_hidden_output[k][j];
                }
                output[j] += mlp->bias_output[j];
            }
            softmax(output, mlp->output_size);

            // Calcular la pérdida (entropía cruzada)
            double loss = 0.0;
            for (int j = 0; j < mlp->output_size; j++) {
                loss += (j == training_data[i].action ? 1 : 0) * log(output[j] + 1e-10);
            }
            total_loss -= loss;

            // Backpropagation
            double delta_output[mlp->output_size];
            for (int j = 0; j < mlp->output_size; j++) {
                delta_output[j] = (j == training_data[i].action ? 1 : 0) - output[j];
            }

            double delta_hidden[mlp->hidden_size];
            for (int j = 0; j < mlp->hidden_size; j++) {
                delta_hidden[j] = 0.0;
                for (int k = 0; k < mlp->output_size; k++) {
                    delta_hidden[j] += delta_output[k] * mlp->weights_hidden_output[j][k];
                }
                delta_hidden[j] *= relu_derivative(hidden[j]);
            }

            // Actualizar pesos y biases de la capa de salida
            for (int j = 0; j < mlp->hidden_size; j++) {
                for (int k = 0; k < mlp->output_size; k++) {
                    mlp->weights_hidden_output[j][k] += learning_rate * delta_output[k] * hidden[j];
                }
            }
            for (int j = 0; j < mlp->output_size; j++) {
                mlp->bias_output[j] += learning_rate * delta_output[j];
            }

            // Actualizar pesos y biases de la capa oculta
            for (int j = 0; j < mlp->input_size; j++) {
                for (int k = 0; k < mlp->hidden_size; k++) {
                    mlp->weights_input_hidden[j][k] += learning_rate * delta_hidden[k] * training_data[i].distance;
                }
            }
            for (int j = 0; j < mlp->hidden_size; j++) {
                mlp->bias_hidden[j] += learning_rate * delta_hidden[j];
            }
        }

        // Imprimir la pérdida cada 100 épocas
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / data_size);
        }
    }
}

// Función para obtener el nombre de la acción
const char* get_action_name(int action) {
    return action_names[action];
}

// Función para cargar los datos de entrenamiento
void load_training_data(DistanceSample training_data[], int *data_size) {
    *data_size = global_training_data_size;
    for (int i = 0; i < global_training_data_size; i++) {
        training_data[i] = global_training_data[i];
    }
}

// Función para predecir y mostrar el resultado
void predict_and_print(MLP *mlp, double distance) {
    double output[mlp->output_size];
    predict_mlp(mlp, distance, output);

    int predicted_action = 0;
    double max_prob = output[0];
    for (int j = 1; j < mlp->output_size; j++) {
        if (output[j] > max_prob) {
            max_prob = output[j];
            predicted_action = j;
        }
    }

    printf("Distancia: %.2f cm -> Acción: %s\n", distance, get_action_name(predicted_action));
}

int main() {
    // Cargar datos de entrenamiento
    DistanceSample training_data[global_training_data_size];
    int data_size;
    load_training_data(training_data, &data_size);

    // Crear y entrenar el modelo
    MLP *mlp = create_mlp(1, 5, 4);  // 1 entrada (distancia), 5 neuronas en la capa oculta, 4 salidas (acciones)
    train_mlp(mlp, training_data, data_size, 0.01, 1000);

    // Probar el modelo con una distancia
    double test_distance = 20.0;  // Distancia en cm
    predict_and_print(mlp, test_distance);

    // Liberar memoria
    free_mlp(mlp);

    return 0;
}