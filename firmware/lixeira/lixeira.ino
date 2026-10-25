// Lixeira inteligente: abre a comporta da categoria recebida pela serial.
// Protocolo: um caractere + '\n' (P plástico, C papel, M metal, V vidro, O orgânico/rejeito).
// Arduino Uno/Nano: biblioteca Servo. ESP32: instale a biblioteca ESP32Servo.

#if defined(ESP32)
#include <ESP32Servo.h>
#else
#include <Servo.h>
#endif

const char COMMANDS[] = {'P', 'C', 'M', 'V', 'O'};
const int PINS[] = {3, 5, 6, 9, 10};  // ESP32: use pinos com PWM, ex. 13, 12, 14, 27, 26
const int N = sizeof(COMMANDS);
const int CLOSED_DEG = 0;
const int OPEN_DEG = 90;
const unsigned long OPEN_MS = 2000;

Servo servos[N];

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N; i++) {
    servos[i].attach(PINS[i]);
    servos[i].write(CLOSED_DEG);
  }
  Serial.println("PRONTO");
}

void loop() {
  if (!Serial.available()) return;
  char c = toupper(Serial.read());
  if (c == '\n' || c == '\r') return;
  for (int i = 0; i < N; i++) {
    if (COMMANDS[i] == c) {
      servos[i].write(OPEN_DEG);
      delay(OPEN_MS);
      servos[i].write(CLOSED_DEG);
      Serial.print("OK ");
      Serial.println(c);
      return;
    }
  }
  Serial.print("ERRO ");
  Serial.println(c);
}
