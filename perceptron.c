#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Estructura para definir un color
typedef struct {
    double rgb[3];  // Valores RGB normalizados
    int label;      // Etiqueta del color
    const char *name;  // Nombre del color
} ColorDefinition;

// Lista de colores predefinidos
ColorDefinition color_definitions[] = {
    {{127 / 255.0, 18 / 255.0, 18 / 255.0}, 0, "Rojo"},
    {{0 / 255.0, 255 / 255.0, 0 / 255.0}, 1, "Verde"},
    {{0 / 255.0, 0 / 255.0, 255 / 255.0}, 2, "Azul"},
    {{255 / 255.0, 255 / 255.0, 0 / 255.0}, 3, "Amarillo"},
    {{126 / 255.0, 18 / 255.0, 172 / 255.0}, 4, "Magenta"},
    {{0 / 255.0, 255 / 255.0, 255 / 255.0}, 5, "Cian"},
    {{0 / 255.0, 0 / 255.0, 0 / 255.0}, 6, "Negro"},
    {{242 / 255.0, 112 / 255.0, 2 / 255.0}, 7, "Naranja"},
    {{200 / 255.0, 200 / 255.0, 200 / 255.0}, 8, "Blanco"},
    {{128 / 255.0, 128 / 255.0, 128 / 255.0}, 9, "Gris"},  // Nuevo color
    // Agrega más colores aquí
    {{153 / 255.0, 255 / 255.0, 153 / 255.0}, 10, "Verde claro"},
    {{204 / 255.0, 255 / 255.0, 204 / 255.0}, 11, "Verde claro"},
    {{255 / 255.0, 51 / 255.0, 153 / 255.0}, 12, "Rosado"},
};

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (3 para RGB)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (número de colores)
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
void predict_mlp(MLP *mlp, double *inputs, double *output) {
    // Capa oculta
    double hidden[mlp->hidden_size];
    for (int i = 0; i < mlp->hidden_size; i++) {
        hidden[i] = 0.0;
        for (int j = 0; j < mlp->input_size; j++) {
            hidden[i] += inputs[j] * mlp->weights_input_hidden[j][i];
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
void train_mlp(MLP *mlp, double training_data[][3], int *labels, int data_size, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < data_size; i++) {
            double output[mlp->output_size];
            double hidden[mlp->hidden_size];

            // Forward pass
            for (int j = 0; j < mlp->hidden_size; j++) {
                hidden[j] = 0.0;
                for (int k = 0; k < mlp->input_size; k++) {
                    hidden[j] += training_data[i][k] * mlp->weights_input_hidden[k][j];
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
                loss += (j == labels[i] ? 1 : 0) * log(output[j] + 1e-10);
            }
            total_loss -= loss;

            // Backpropagation
            double delta_output[mlp->output_size];
            for (int j = 0; j < mlp->output_size; j++) {
                delta_output[j] = (j == labels[i] ? 1 : 0) - output[j];
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
                    mlp->weights_input_hidden[j][k] += learning_rate * delta_hidden[k] * training_data[i][j];
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

// Función para normalizar un color
double* normalize_color(int r, int g, int b) {
    double *color = (double*)malloc(3 * sizeof(double));
    color[0] = r / 255.0;
    color[1] = g / 255.0;
    color[2] = b / 255.0;
    return color;
}

// Función para obtener el nombre del color
const char* get_color_name(int class_id) {
    for (size_t i = 0; i < sizeof(color_definitions) / sizeof(ColorDefinition); i++) {
        if (color_definitions[i].label == class_id) {
            return color_definitions[i].name;
        }
    }
    return "Desconocido";
}

// Función para cargar los datos de entrenamiento
void load_training_data(double training_data[][3], int *labels, size_t *data_size) {
    *data_size = sizeof(color_definitions) / sizeof(ColorDefinition);
    for (size_t i = 0; i < *data_size; i++) {
        training_data[i][0] = color_definitions[i].rgb[0];
        training_data[i][1] = color_definitions[i].rgb[1];
        training_data[i][2] = color_definitions[i].rgb[2];
        labels[i] = color_definitions[i].label;
    }
}

// Función para predecir y mostrar el resultado
void predict_and_print(MLP *mlp, double *color) {
    double output[mlp->output_size];
    predict_mlp(mlp, color, output);

    int predicted_class = 0;
    double max_prob = output[0];
    for (int j = 1; j < mlp->output_size; j++) {
        if (output[j] > max_prob) {
            max_prob = output[j];
            predicted_class = j;
        }
    }

    printf("Color: (%.0f, %.0f, %.0f) -> Predicción: %s\n",
           color[0] * 255, color[1] * 255, color[2] * 255, get_color_name(predicted_class));
}

int main() {
    // Cargar datos de entrenamiento
    double training_data[15][3];
    int labels[15];
    size_t data_size;
    load_training_data(training_data, labels, &data_size);

    // Crear y entrenar el modelo
    MLP *mlp = create_mlp(3, 20, sizeof(color_definitions) / sizeof(ColorDefinition));  // 3 entradas (RGB), 5 neuronas en la capa oculta, 8 salidas (colores + negro + naranja)
    train_mlp(mlp, training_data, labels, data_size, 0.01, 10000);  // Tasa de aprendizaje más baja y más épocas

    // Probar el modelo con un nuevo color
    int r = 255, g = 255, b = 0;  // Naranja
    double *test_color = normalize_color(r, g, b);

    // Predecir y mostrar resultados
    printf("Probando la red neuronal con un solo color RGB:\n");
    predict_and_print(mlp, test_color);

    // Liberar memoria
    free(test_color);
    free_mlp(mlp);

    return 0;
}