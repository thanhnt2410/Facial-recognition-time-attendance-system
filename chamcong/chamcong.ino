

#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN 15
#define RST_PIN 16

// | RC522    | ESP8266 |
// | -------- | ------- |
// | SDA (SS) | D8      |
// | SCK      | D5      |
// | MOSI     | D7      |
// | MISO     | D6      |
// | RST      | D0      |
// | GND      | GND     |
// | 3.3V     | 3.3V    |
#define BTN_A 4
#define BTN_B 5
#define BUZZER 2


MFRC522 rfid(SS_PIN, RST_PIN);

bool modeA = true;   // mặc định chế độ A

void setup() {

  Serial.begin(115200);

  SPI.begin();
  rfid.PCD_Init();

  pinMode(BTN_A, INPUT_PULLUP);
  pinMode(BTN_B, INPUT_PULLUP);

  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, HIGH);

  Serial.println("System ready");
}

void beep() {
  digitalWrite(BUZZER, LOW);
  delay(80);
  digitalWrite(BUZZER, HIGH);
}

void loop() {

  // kiểm tra nút nhấn
  if (digitalRead(BTN_A) == LOW) {
    modeA = true;
    Serial.println("ModeA");
    delay(200);
  }

  if (digitalRead(BTN_B) == LOW) {
    modeA = false;
    Serial.println("ModeB");
    delay(200);
  }

  // nếu chế độ B thì không đọc RFID
  if (!modeA) {
    return;
  }

  // đọc RFID
  if (!rfid.PICC_IsNewCardPresent()) {
    return;
  }

  if (!rfid.PICC_ReadCardSerial()) {
    return;
  }

// tạo chuỗi UID
String uidStr = "";

for (byte i = 0; i < rfid.uid.size; i++) {

  if (rfid.uid.uidByte[i] < 0x10) {
    uidStr += "0";
  }

  uidStr += String(rfid.uid.uidByte[i], HEX);
}

// chuyển thành chữ hoa
uidStr.toUpperCase();

// gửi dạng rfid:UID
Serial.print("rfid:");
Serial.println(uidStr);

beep();

rfid.PICC_HaltA();
}