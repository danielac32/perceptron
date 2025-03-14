#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Estructura para definir una muestra de texto y etiqueta
typedef struct {
    char text[250];  // Texto de entrada
    int label;       // Etiqueta (0: Deportes, 1: Tecnología, 2: Política)
} TextSample;

// Lista de datos de entrenamiento (global)
TextSample global_training_data[] = {
    // Deportes
    {"actividad física", 0},
    {"competencias", 0},



    {"El equipo ganó el partido de fútbol", 0},
    {"El jugador anotó un gol espectacular", 0},
    {"El tenista ganó el torneo en cinco sets", 0},
    {"El equipo de baloncesto avanzó a la final", 0},
    {"El maratonista rompió el récord mundial", 0},
    {"El equipo de hockey ganó el campeonato", 0},
    {"El ciclista completó la carrera en tiempo récord", 0},
    {"El boxeador ganó por knockout en el tercer round", 0},
    {"El nadador ganó la medalla de oro en los Juegos Olímpicos", 0},
    {"El equipo de rugby se clasificó para la final", 0},
    {"El golfista hizo un hoyo en uno en el torneo", 0},
    {"El equipo de béisbol ganó la serie mundial", 0},
    {"El atleta estableció un nuevo récord en salto de altura", 0},
    {"El equipo de fútbol americano ganó el Super Bowl", 0},
    {"El patinador ganó la medalla de oro en el campeonato mundial", 0},
    {"La actividad física es importante para la salud", 0},
    {"El ejercicio al aire libre mejora el bienestar", 0},
    {"El equipo siguió las reglas del juego al pie de la letra", 0},
    {"La competición deportiva fomenta el espíritu de equipo", 0},
    {"El entrenamiento físico es clave para el rendimiento", 0},
    {"El equipo de voleibol ganó el campeonato nacional", 0},
    {"El corredor completó la maratón en menos de tres horas", 0},
    {"El equipo de natación ganó la medalla de plata", 0},
    {"El jugador de balonmano anotó un gol decisivo", 0},
    {"El equipo de cricket ganó el partido por amplio margen", 0},
    {"El atleta ganó la medalla de bronce en los Juegos Panamericanos", 0},
    {"El equipo de waterpolo avanzó a la semifinal", 0},
    {"El jugador de bádminton ganó el torneo internacional", 0},
    {"El equipo de esgrima ganó la medalla de oro", 0},
    {"El atleta ganó la carrera de 100 metros lisos", 0},

    // Tecnología
    {"Nuevo lanzamiento de smartphone con cámara avanzada", 1},
    {"La empresa lanzó un nuevo sistema operativo", 1},
    {"Nueva tecnología de inteligencia artificial revoluciona la industria", 1},
    {"Nuevo avance en la computación cuántica", 1},
    {"Nuevo dispositivo wearable para monitorear la salud", 1},
    {"Nuevo software para la gestión de proyectos", 1},
    {"La compañía presentó un nuevo robot autónomo", 1},
    {"Nuevo chip de procesamiento con mayor eficiencia energética", 1},
    {"La startup desarrolló una aplicación revolucionaria", 1},
    {"Nuevo avance en la tecnología de baterías de litio", 1},
    {"La empresa anunció un nuevo servicio de streaming", 1},
    {"Nuevo sistema de reconocimiento facial para seguridad", 1},
    {"La compañía lanzó un nuevo dron con cámara 4K", 1},
    {"Nuevo avance en la realidad virtual inmersiva", 1},
    {"La empresa presentó un nuevo vehículo eléctrico", 1},
    {"Nuevo avance en la tecnología blockchain", 1},
    {"La empresa lanzó un nuevo asistente virtual", 1},
    {"Nuevo sistema de almacenamiento en la nube", 1},
    {"La compañía presentó un nuevo sistema de realidad aumentada", 1},
    {"Nuevo avance en la tecnología 5G", 1},
    {"La empresa desarrolló un nuevo sistema de inteligencia artificial", 1},
    {"Nuevo dispositivo para monitorear la calidad del aire", 1},
    {"La startup lanzó una nueva plataforma de comercio electrónico", 1},
    {"Nuevo avance en la tecnología de impresión 3D", 1},
    {"La empresa presentó un nuevo sistema de seguridad informática", 1},
    {"Nuevo avance en la tecnología de drones autónomos", 1},
    {"La compañía lanzó un nuevo sistema de gestión de datos", 1},
    {"Nuevo avance en la tecnología de energía solar", 1},
    {"La empresa presentó un nuevo sistema de transporte autónomo", 1},
    {"Nuevo avance en la tecnología de biometría", 1},

    // Política
    {"El presidente anunció nuevas reformas económicas", 2},
    {"El congreso aprobó la nueva ley de impuestos", 2},
    {"El gobierno firmó un acuerdo internacional", 2},
    {"El senado debatió la nueva política ambiental", 2},
    {"El primer ministro se reunió con líderes mundiales", 2},
    {"El parlamento discutió la reforma educativa", 2},
    {"El presidente firmó un tratado de libre comercio", 2},
    {"El gobierno anunció un plan de estímulo económico", 2},
    {"El congreso aprobó la nueva ley de seguridad social", 2},
    {"El senado discutió la reforma migratoria", 2},
    {"El primer ministro anunció un nuevo plan de infraestructura", 2},
    {"El parlamento aprobó la nueva ley de protección de datos", 2},
    {"El presidente se reunió con líderes de la Unión Europea", 2},
    {"El gobierno lanzó un nuevo programa de vivienda", 2},
    {"El congreso debatió la nueva ley de energía renovable", 2},
    {"El presidente anunció un nuevo plan de empleo", 2},
    {"El gobierno firmó un acuerdo de cooperación internacional", 2},
    {"El senado aprobó la nueva ley de transporte público", 2},
    {"El primer ministro anunció un nuevo plan de salud", 2},
    {"El parlamento discutió la reforma fiscal", 2},
    {"El presidente se reunió con líderes de América Latina", 2},
    {"El gobierno anunció un nuevo plan de seguridad ciudadana", 2},
    {"El congreso aprobó la nueva ley de educación superior", 2},
    {"El senado discutió la reforma laboral", 2},
    {"El primer ministro anunció un nuevo plan de desarrollo rural", 2},
    {"El parlamento aprobó la nueva ley de protección ambiental", 2},
    {"El presidente firmó un acuerdo de paz", 2},
    {"El gobierno lanzó un nuevo programa de apoyo a pequeñas empresas", 2},
    {"El congreso debatió la nueva ley de telecomunicaciones", 2},
    {"El senado aprobó la nueva ley de ciencia y tecnología", 2},
};
// Tamaño de la lista de datos de entrenamiento
const int global_training_data_size = sizeof(global_training_data) / sizeof(TextSample);

// Vocabulario de palabras únicas
const char *vocabulary[] = {
    "equipo", "ganó", "partido", "fútbol", "lanzamiento", "smartphone", "cámara", "avanzada", "presidente", "anunció",
    "reformas", "económicas", "jugador", "anotó", "gol", "espectacular", "empresa", "sistema", "operativo", "congreso",
    "aprobó", "ley", "impuestos", "tenista", "torneo", "sets", "tecnología", "inteligencia", "artificial", "revoluciona",
    "industria", "gobierno", "firmó", "acuerdo", "internacional", "baloncesto", "avanzó", "final", "avance", "computación",
    "cuántica", "senado", "debatió", "política", "ambiental", "maratonista", "rompió", "récord", "mundial", "dispositivo",
    "wearable", "monitorear", "salud", "primer", "ministro", "reunió", "líderes", "hockey", "campeonato", "software",
    "gestión", "proyectos", "parlamento", "discutió", "educativa", "voleibol", "corredor", "maratón", "natación", "balonmano",
    "cricket", "waterpolo", "bádminton", "esgrima", "carrera", "blockchain", "asistente", "virtual", "almacenamiento", "nube",
    "realidad", "aumentada", "5G", "inteligencia", "artificial", "calidad", "aire", "plataforma", "comercio", "electrónico",
    "impresión", "3D", "seguridad", "informática", "drones", "autónomos", "gestión", "datos", "energía", "solar", "transporte",
    "biometría", "empleo", "cooperación", "transporte", "público", "salud", "fiscal", "América", "Latina", "seguridad",
    "ciudadana", "educación", "superior", "laboral", "desarrollo", "rural", "protección", "ambiental", "paz", "apoyo",
    "pequeñas", "empresas", "telecomunicaciones", "ciencia", "tecnología","deporte","deportes","actividad"
};
const int vocabulary_size = sizeof(vocabulary) / sizeof(vocabulary[0]);

// Estructura de la Red Neuronal Multicapa (MLP)
typedef struct {
    double **weights_input_hidden;  // Pesos entre la capa de entrada y la oculta
    double **weights_hidden_output; // Pesos entre la capa oculta y la de salida
    double *bias_hidden;            // Bias de la capa oculta
    double *bias_output;            // Bias de la capa de salida
    int input_size;                 // Tamaño de la entrada (vocabulario_size)
    int hidden_size;                // Tamaño de la capa oculta
    int output_size;                // Tamaño de la salida (3 para clasificación multiclase)
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

// Función para convertir un texto en un vector one-hot
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
}

// Predecir la salida para un texto dado
void predict_mlp_text(MLP *mlp, const char *text, double *output) {
    double input[vocabulary_size];
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
            double input[vocabulary_size];
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

    int predicted_label = 0;
    double max_prob = output[0];
    for (int j = 1; j < mlp->output_size; j++) {
        if (output[j] > max_prob) {
            max_prob = output[j];
            predicted_label = j;
        }
    }

    const char *label_name;
    switch (predicted_label) {
        case 0:
            label_name = "Deportes";
            break;
        case 1:
            label_name = "Tecnología";
            break;
        case 2:
            label_name = "Política";
            break;
        default:
            label_name = "Desconocido";
            break;
    }

    printf("Texto: %s -> Predicción: %s\n", text, label_name);
}

int main() {
    // Crear el modelo
    MLP *mlp = create_mlp(vocabulary_size, 15, 3);  // vocabulary_size entradas, 10 neuronas ocultas, 3 salidas (multiclase)

    // Entrenar el modelo
    train_mlp_text(mlp, global_training_data, global_training_data_size, 0.01, 2000);

    // Probar el modelo con un texto
    const char *test_text = "aquella actividad física que involucra ";
    predict_and_print_text(mlp, test_text);

    // Liberar memoria
    free_mlp(mlp);

    return 0;
}