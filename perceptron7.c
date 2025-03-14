// este modelo de perceptron no aprovecha las capacidades del algoritmo , ya que usa una funcion para encontrar las palabras clave
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Estructura para definir una muestra de texto y etiqueta
typedef struct {
    char text[100];  // Texto de entrada
    int label;       // Etiqueta (1: "pago especial", 0: otro pago)
} TextSample;

// Lista de datos de entrenamiento (global)
TextSample global_training_data[] = {
    {"pago especial para institucion", 1},
    {"se le dara un pago especial", 1},
    {"se realizo un pago especial", 1},
    {"se le hizo a daniel pago especial", 1},
    {"pago especial para el personal", 1},
    {"se otorgó un pago especial a los empleados", 1},
    {"pago especial por servicios adicionales", 1},
    {"se aprobó un pago especial para los docentes", 1},
    {"pago especial para el equipo de trabajo", 1},
    {"se realizó un pago especial a los voluntarios", 1},
    {"pago especial por horas extras", 1},
    {"se entregó un pago especial a los contratistas", 1},
    {"pago especial para los becarios", 1},
    {"se asignó un pago especial para los guardias", 1},
    {"pago especial por desempeño excepcional", 1},
    {"se otorgó un pago especial a los consultores", 1},
    {"pago especial para los conductores", 1},
    {"se realizó un pago especial a los supervisores", 1},
    {"pago especial para los técnicos", 1},
    {"se entregó un pago especial a los analistas", 1},

    // Ejemplos de otros tipos de gastos
    {"se le realizo pago a niño especial", 0},
    {"se compro un bombillo especial para los baños", 0},
    {"se adquirió material especial para la oficina", 0},
    {"se pagó la factura de luz", 0},
    {"se compraron suministros para el mantenimiento", 0},
    {"se realizó el pago del alquiler", 0},
    {"se adquirió equipo especial para el laboratorio", 0},
    {"se pagó la nómina del mes", 0},
    {"se compraron alimentos para la cafetería", 0},
    {"se realizó el pago de servicios públicos", 0},
    {"se adquirió mobiliario para la sala de reuniones", 0},
    {"se pagó la factura de internet", 0},
    {"se compraron herramientas para el taller", 0},
    {"se realizó el pago de la prima de seguros", 0},
    {"se adquirió material de limpieza", 0},
    {"se pagó la factura de teléfono", 0},
    {"se compraron uniformes para el personal", 0},
    {"se realizó el pago de la licencia de software", 0},
    {"se adquirió equipo de seguridad", 0},
    {"se pagó la factura de agua", 0},
    {"se adquirió equipo especial para el laboratorio", 0},
    {"se compró material especial para la construcción", 0},
    {"se instaló un sistema especial de seguridad", 0},
    {"se diseñó un programa especial para estudiantes", 0},
    {"se implementó un protocolo especial de emergencia", 0},
    {"especial atención al cliente", 0},
    {"un evento especial para la comunidad", 0},
};

// Tamaño de la lista de datos de entrenamiento
const int global_training_data_size = sizeof(global_training_data) / sizeof(TextSample);

// Vocabulario de palabras únicas
const char *vocabulary[] = {
    "bombillo", "pago", "especial", "para", "institucion", "se", "le", "dara", "realizo", "hizo", "daniel", "niño", "compro", "baños",
    "personal", "empleados", "servicios", "adicionales", "aprobó", "docentes", "equipo", "trabajo", "voluntarios", "horas", "extras",
    "entregó", "contratistas", "becarios", "asignó", "guardias", "desempeño", "excepcional", "consultores", "conductores", "supervisores",
    "técnicos", "analistas", "adquirió", "material", "oficina", "pagó", "factura", "luz", "compraron", "suministros", "mantenimiento",
    "alquiler", "equipo", "laboratorio", "nómina", "mes", "alimentos", "cafetería", "servicios", "públicos", "mobiliario", "sala",
    "reuniones", "internet", "herramientas", "taller", "prima", "seguros", "limpieza", "teléfono", "uniformes", "licencia", "software",
    "seguridad", "agua"
};
const int vocabulary_size = sizeof(vocabulary) / sizeof(vocabulary[0]);

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (vocabulario_size + 1)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (2 para clasificación binaria)
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

// Función para verificar si "pago" y "especial" aparecen juntos
int contiene_pago_especial(const char *texto) {
    char text_copy[100];
    strcpy(text_copy, texto);  // Copiar el texto para no modificar el original

    char *token = strtok(text_copy, " ");
    char prev_token[20] = "";

    while (token != NULL) {
        if (strcmp(token, "pago") == 0) {
            strcpy(prev_token, token);
        } else if (strcmp(token, "especial") == 0 && strcmp(prev_token, "pago") == 0) {
            return 1;  // "pago" y "especial" aparecen juntos
        }
        token = strtok(NULL, " ");
    }

    return 0;  // No aparecen juntos
}

// Función para convertir un texto en un vector one-hot con característica adicional
void text_to_one_hot(const char *text, double *vector) {
    for (int i = 0; i < vocabulary_size; i++) {
        vector[i] = 0.0;  // Inicializar el vector a 0
    }

    char text_copy[100];
    strcpy(text_copy, text);  // Copiar el texto para no modificar el original

    char *token = strtok(text_copy, " ");
    while (token != NULL) {
        for (int i = 0; i < vocabulary_size; i++) {
            if (strcmp(token, vocabulary[i]) == 0) {
                vector[i] = 1.0;  // Marcar la palabra en el vector one-hot
            }
        }
        token = strtok(NULL, " ");
    }

    // Agregar la característica adicional: si "pago" y "especial" aparecen juntos
    vector[vocabulary_size] = contiene_pago_especial(text) ? 1.0 : 0.0;
}

// Predecir la salida para un texto dado
void predict_mlp_text(MLP *mlp, const char *text, double *output) {
    double input[vocabulary_size + 1];  // +1 para la característica adicional
    text_to_one_hot(text, input);  // Convertir texto a one-hot

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

// Entrenar la red neuronal con datos de texto
void train_mlp_text(MLP *mlp, TextSample training_data[], int data_size, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < data_size; i++) {
            double input[vocabulary_size + 1];  // +1 para la característica adicional
            text_to_one_hot(training_data[i].text, input);  // Convertir texto a one-hot

            double output[mlp->output_size];
            double hidden[mlp->hidden_size];

            // Forward pass
            for (int j = 0; j < mlp->hidden_size; j++) {
                hidden[j] = 0.0;
                for (int k = 0; k < mlp->input_size; k++) {
                    hidden[j] += input[k] * mlp->weights_input_hidden[k][j];
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
                    mlp->weights_input_hidden[j][k] += learning_rate * delta_hidden[k] * input[j];
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

// Función para predecir y mostrar el resultado
void predict_and_print_text(MLP *mlp, const char *text) {
    double output[mlp->output_size];
    predict_mlp_text(mlp, text, output);

    int predicted_label = (output[1] > output[0]) ? 1 : 0;  // Clasificación binaria
    printf("Texto: %s -> Predicción: %s\n", text, predicted_label ? "Pago especial" : "Otro pago");
}

int main() {
    // Crear el modelo
    MLP *mlp = create_mlp(vocabulary_size + 1, 10, 2);  // +1 para la característica adicional

    // Entrenar el modelo
    train_mlp_text(mlp, global_training_data, global_training_data_size, 0.01, 1000);

    // Probar el modelo con un texto
    const char *test_text = "se realizó un pago  a los voluntarios";
    predict_and_print_text(mlp, test_text);

    // Liberar memoria
    free_mlp(mlp);

    return 0;
}