// 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>  // Biblioteca para expresiones regulares

// Función para verificar si un texto contiene "pago especial"
int contiene_pago_especial(const char *texto) {
    // Compilar la expresión regular
    regex_t regex;
    int resultado;
    char *patron = "pago especial";

    // Compilar la expresión regular
    resultado = regcomp(&regex, patron, REG_EXTENDED);
    if (resultado != 0) {
        printf("Error al compilar la expresión regular.\n");
        return -1;
    }

    // Ejecutar la expresión regular
    resultado = regexec(&regex, texto, 0, NULL, 0);

    // Liberar la memoria de la expresión regular
    regfree(&regex);

    // Devolver el resultado
    if (resultado == 0) {
        return 1;  // "pago especial" encontrado
    } else {
        return 0;  // "pago especial" no encontrado
    }
}

// Función para clasificar un texto
void clasificar_texto(const char *texto) {
    if (contiene_pago_especial(texto)) {
        printf("Texto: \"%s\" -> Clasificación: Pago especial\n", texto);
    } else {
        printf("Texto: \"%s\" -> Clasificación: Otro pago\n", texto);
    }
}

int main() {
    // Ejemplos de textos para clasificar
    const char *texto1 = "pago especial a un ente público";
    const char *texto2 = "pago para niño especial";
    const char *texto3 = "compra de puerta especial para el baño";
    const char *texto4 = "pago de servicios";

    // Clasificar los textos
    clasificar_texto(texto1);
    clasificar_texto(texto2);
    clasificar_texto(texto3);
    clasificar_texto(texto4);

    return 0;
}