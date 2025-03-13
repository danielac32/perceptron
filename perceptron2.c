#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Estructura para definir un texto
typedef struct {
    const char *text;  // Texto
    int label;         // Etiqueta (1 para "pago especial", 0 para otro tipo)
} TextDefinition;

// Lista de textos predefinidos
TextDefinition text_definitions[] = {
    // Ejemplos de "pago especial"
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
    {"se adquirió equipo especial para el laboratorio", 0},  // Nuevo ejemplo
    {"se compró material especial para la construcción", 0}, // Nuevo ejemplo
    {"se instaló un sistema especial de seguridad", 0},      // Nuevo ejemplo
    {"se diseñó un programa especial para estudiantes", 0},  // Nuevo ejemplo
    {"se implementó un protocolo especial de emergencia", 0},// Nuevo ejemplo
    {"especial atención al cliente", 0},                    // Nuevo ejemplo
    {"un evento especial para la comunidad", 0},            // Nuevo ejemplo
};

// Vocabulario (palabras únicas en los textos)
const char *vocab[] = {
    "bombillo", "pago", "especial", "para", "institucion", "se", "le", "dara", "realizo", "hizo", "daniel", "niño", "compro", "baños",
    "personal", "empleados", "servicios", "adicionales", "aprobó", "docentes", "equipo", "trabajo", "voluntarios", "horas", "extras",
    "entregó", "contratistas", "becarios", "asignó", "guardias", "desempeño", "excepcional", "consultores", "conductores", "supervisores",
    "técnicos", "analistas", "adquirió", "material", "oficina", "pagó", "factura", "luz", "compraron", "suministros", "mantenimiento",
    "alquiler", "equipo", "laboratorio", "nómina", "mes", "alimentos", "cafetería", "servicios", "públicos", "mobiliario", "sala",
    "reuniones", "internet", "herramientas", "taller", "prima", "seguros", "limpieza", "teléfono", "uniformes", "licencia", "software",
    "seguridad", "agua"
};
int vocab_size = sizeof(vocab) / sizeof(vocab[0]);  // +1 para la nueva característica

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (número de palabras únicas)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (1 para clasificación binaria)
} MLP;

// Función de activación (ReLU)
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivada de la función ReLU
double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

// Función de activación (Sigmoid)
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivada de la función Sigmoid
double sigmoid_derivative(double x) {
    return x * (1 - x);
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
double predict_mlp(MLP *mlp, double *inputs) {
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
    double output = 0.0;
    for (int i = 0; i < mlp->hidden_size; i++) {
        output += hidden[i] * mlp->weights_hidden_output[i][0];
    }
    output += mlp->bias_output[0];
    output = sigmoid(output);

    return output;
}

// Entrenar la red neuronal
void train_mlp(MLP *mlp, double training_data[][vocab_size], int *labels, int data_size, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < data_size; i++) {
            double output;
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

            output = 0.0;
            for (int j = 0; j < mlp->hidden_size; j++) {
                output += hidden[j] * mlp->weights_hidden_output[j][0];
            }
            output += mlp->bias_output[0];
            output = sigmoid(output);

            // Calcular la pérdida (entropía cruzada)
            double loss = -labels[i] * log(output + 1e-10) - (1 - labels[i]) * log(1 - output + 1e-10);
            total_loss += loss;

            // Backpropagation
            double delta_output = output - labels[i];
            double delta_hidden[mlp->hidden_size];
            for (int j = 0; j < mlp->hidden_size; j++) {
                delta_hidden[j] = delta_output * mlp->weights_hidden_output[j][0] * relu_derivative(hidden[j]);
            }

            // Actualizar pesos y biases de la capa de salida
            for (int j = 0; j < mlp->hidden_size; j++) {
                mlp->weights_hidden_output[j][0] -= learning_rate * delta_output * hidden[j];
            }
            mlp->bias_output[0] -= learning_rate * delta_output;

            // Actualizar pesos y biases de la capa oculta
            for (int j = 0; j < mlp->input_size; j++) {
                for (int k = 0; k < mlp->hidden_size; k++) {
                    mlp->weights_input_hidden[j][k] -= learning_rate * delta_hidden[k] * training_data[i][j];
                }
            }
            for (int j = 0; j < mlp->hidden_size; j++) {
                mlp->bias_hidden[j] -= learning_rate * delta_hidden[j];
            }
        }

        // Imprimir la pérdida cada 100 épocas
        if (epoch % 100 == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / data_size);
        }
    }
}

// Función para convertir texto en un vector de características (bolsa de palabras)
void text_to_vector(const char *text, double *vector, const char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size - 1; i++) {  // Usar vocab_size - 1
        vector[i] = 0.0; // Inicializar a 0
        if (strstr(text, vocab[i]) != NULL) { // Verificar si la palabra está en el texto
            vector[i] = 1.0; // Marcar como presente
        }
    }
    // Característica adicional: ¿Contiene la frase "pago especial"?
   vector[vocab_size - 1] = (strstr(text, "pago especial") != NULL) ? 1.0 : 0.0;
}

// Función para cargar los datos de entrenamiento
void load_training_data(double training_data[][vocab_size], int *labels, size_t *data_size, const char **vocab, int vocab_size) {
    *data_size = sizeof(text_definitions) / sizeof(TextDefinition);
    for (size_t i = 0; i < *data_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            training_data[i][j] = 0.0; // Inicializar a 0
            if (strstr(text_definitions[i].text, vocab[j]) != NULL) { // Verificar si la palabra está en el texto
                training_data[i][j] = 1.0; // Marcar como presente
            }
        }
        labels[i] = text_definitions[i].label;
    }
}

int main() {
    // Cargar datos de entrenamiento
    size_t data_size = sizeof(text_definitions) / sizeof(TextDefinition);
    double training_data[data_size][vocab_size];  // Ajustar el tamaño según el número de textos y el vocabulario
    int labels[data_size];
    load_training_data(training_data, labels, &data_size, vocab, vocab_size);

    // Crear y entrenar el modelo
    MLP *mlp = create_mlp(vocab_size, 10, 1);  // Más neuronas en la capa oculta
    train_mlp(mlp, training_data, labels, data_size, 0.005, 20000);  // Tasa de aprendizaje más baja y más épocas

    // Probar el modelo con un nuevo texto
    const char *test_text = "se implementó un protocolo especial de emergencia";
    double test_vector[vocab_size];
    for (int i = 0; i < vocab_size - 1; i++) {
        test_vector[i] = 0.0;
        if (strstr(test_text, vocab[i]) != NULL) {
            test_vector[i] = 1.0;
        }
    }
    test_vector[vocab_size - 1] = (strstr(test_text, "pago especial") != NULL) ? 1.0 : 0.0;

    double output = predict_mlp(mlp, test_vector);
    printf("Texto: %s -> Predicción: %s\n", test_text, output > 0.5 ? "Pago especial" : "Otro tipo");

    // Liberar memoria
    free_mlp(mlp);

    return 0;
}