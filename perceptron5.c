#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Tamaño de la imagen
#define IMAGE_SIZE 10
#define INPUT_SIZE (IMAGE_SIZE * IMAGE_SIZE)
#define OUTPUT_SIZE 10

// Estructura para definir una muestra de imagen y su etiqueta
typedef struct {
    double image[INPUT_SIZE];  // Imagen de 10x10 píxeles
    int label;                 // Etiqueta (0-9)
} ImageSample;

// Lista de etiquetas predefinidas
const char *label_names[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Generar imágenes sintéticas de números (0-9)
void generate_synthetic_images(ImageSample training_data[], int data_size) {
    // Número 0
    double zero[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[0].image[i] = zero[i];
    training_data[0].label = 0;

    // Número 1
    double one[INPUT_SIZE] = {
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[1].image[i] = one[i];
    training_data[1].label = 1;
    
    // Número 2
    double two[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[2].image[i] = two[i];
    training_data[2].label = 2;

    // Número 3
    double three[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[3].image[i] = three[i];
    training_data[3].label = 3;

    // Número 4
    double four[INPUT_SIZE] = {
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[4].image[i] = four[i];
    training_data[4].label = 4;

    // Número 5
    double five[INPUT_SIZE] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[5].image[i] = five[i];
    training_data[5].label = 5;

    // Número 6
    double six[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[6].image[i] = six[i];
    training_data[6].label = 6;

    // Número 7
    double seven[INPUT_SIZE] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[7].image[i] = seven[i];
    training_data[7].label = 7;

    // Número 8
    double eight[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[8].image[i] = eight[i];
    training_data[8].label = 8;

    // Número 9
    double nine[INPUT_SIZE] = {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[9].image[i] = nine[i];
    training_data[9].label = 9;

    double nine2[INPUT_SIZE] = {
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[10].image[i] = nine2[i];
    training_data[10].label = 9;
    
     double one2[INPUT_SIZE] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[11].image[i] = one2[i];
    training_data[11].label = 1;
    
    
     double one3[INPUT_SIZE] = {
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0
    };
    for (int i = 0; i < INPUT_SIZE; i++) training_data[12].image[i] = one3[i];
    training_data[12].label = 1;
    

}

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (100 para 10x10 píxeles)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (10 para números del 0 al 9)
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
void predict_mlp(MLP *mlp, double image[INPUT_SIZE], double *output) {
    double hidden[mlp->hidden_size];

    // Capa oculta
    for (int i = 0; i < mlp->hidden_size; i++) {
        hidden[i] = 0.0;
        for (int j = 0; j < mlp->input_size; j++) {
            hidden[i] += image[j] * mlp->weights_input_hidden[j][i];
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
void train_mlp(MLP *mlp, ImageSample training_data[], int data_size, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < data_size; i++) {
            double output[mlp->output_size];
            double hidden[mlp->hidden_size];

            // Forward pass
            for (int j = 0; j < mlp->hidden_size; j++) {
                hidden[j] = 0.0;
                for (int k = 0; k < mlp->input_size; k++) {
                    hidden[j] += training_data[i].image[k] * mlp->weights_input_hidden[k][j];
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
                loss += (j == training_data[i].label ? 1 : 0) * log(output[j] + 1e-10);
            }
            total_loss -= loss;

            // Backpropagation
            double delta_output[mlp->output_size];
            for (int j = 0; j < mlp->output_size; j++) {
                delta_output[j] = (j == training_data[i].label ? 1 : 0) - output[j];
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
                    mlp->weights_input_hidden[j][k] += learning_rate * delta_hidden[k] * training_data[i].image[j];
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

// Guardar los pesos y biases en un archivo
void save_mlp(MLP *mlp, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error al abrir el archivo para guardar");
        return;
    }

    // Guardar pesos de la capa de entrada a la oculta
    for (int i = 0; i < mlp->input_size; i++) {
        fwrite(mlp->weights_input_hidden[i], sizeof(double), mlp->hidden_size, file);
    }

    // Guardar pesos de la capa oculta a la de salida
    for (int i = 0; i < mlp->hidden_size; i++) {
        fwrite(mlp->weights_hidden_output[i], sizeof(double), mlp->output_size, file);
    }

    // Guardar biases de la capa oculta
    fwrite(mlp->bias_hidden, sizeof(double), mlp->hidden_size, file);

    // Guardar biases de la capa de salida
    fwrite(mlp->bias_output, sizeof(double), mlp->output_size, file);

    fclose(file);
    printf("Pesos y biases guardados en %s\n", filename);
}

// Cargar los pesos y biases desde un archivo
void load_mlp(MLP *mlp, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error al abrir el archivo para cargar");
        return;
    }

    // Cargar pesos de la capa de entrada a la oculta
    for (int i = 0; i < mlp->input_size; i++) {
        fread(mlp->weights_input_hidden[i], sizeof(double), mlp->hidden_size, file);
    }

    // Cargar pesos de la capa oculta a la de salida
    for (int i = 0; i < mlp->hidden_size; i++) {
        fread(mlp->weights_hidden_output[i], sizeof(double), mlp->output_size, file);
    }

    // Cargar biases de la capa oculta
    fread(mlp->bias_hidden, sizeof(double), mlp->hidden_size, file);

    // Cargar biases de la capa de salida
    fread(mlp->bias_output, sizeof(double), mlp->output_size, file);

    fclose(file);
    printf("Pesos y biases cargados desde %s\n", filename);
}

// Función para obtener el nombre de la etiqueta
const char* get_label_name(int label) {
    return label_names[label];
}

// Función para predecir y mostrar el resultado
void predict_and_print(MLP *mlp, double image[INPUT_SIZE]) {
    double output[mlp->output_size];
    predict_mlp(mlp, image, output);

    int predicted_label = 0;
    double max_prob = output[0];
    for (int j = 1; j < mlp->output_size; j++) {
        if (output[j] > max_prob) {
            max_prob = output[j];
            predicted_label = j;
        }
    }

    printf("Predicción: %s\n", get_label_name(predicted_label));
}

int main() {
    // Generar datos de entrenamiento
    ImageSample training_data[20];
    int data_size = 13;
    generate_synthetic_images(training_data, data_size);

    // Crear el modelo
    MLP *mlp = create_mlp(INPUT_SIZE, 128, OUTPUT_SIZE);  // 100 entradas (10x10 píxeles), 128 neuronas en la capa oculta, 10 salidas (números del 0 al 9)

    const char *filename = "mlp_weights.bin";

    // Menú de opciones
    printf("Seleccione una opción:\n");
    printf("1. Entrenar y guardar pesos\n");
    printf("2. Cargar pesos y predecir\n");
    int option;
    scanf("%d", &option);

    if (option == 1) {
        // Entrenar el modelo y guardar los pesos
        train_mlp(mlp, training_data, data_size, 0.005, 10000);
        save_mlp(mlp, filename);
    } else if (option == 2) {
        // Cargar los pesos y predecir
        load_mlp(mlp, filename);
    } else {
        printf("Opción no válida.\n");
        free_mlp(mlp);
        return 1;
    }

    // Probar el modelo con una imagen
   /* double test_image[INPUT_SIZE] = {
        0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    };*/

    double test_image[INPUT_SIZE] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0
    };


    predict_and_print(mlp, test_image);

    // Liberar memoria
    free_mlp(mlp);

    return 0;
}