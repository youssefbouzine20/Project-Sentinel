#include <HX711.h>
//DOUT: Sends Weight Numbers
const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN = 3;
//Creating the Object
HX711 scale;
//Should be changed untill the output matches the real weight
float calibration_factor = 420;

void setup() {
  Serial.begin(9600);
  Serial.println("Sentinel Load Cell test!");
  //Starting the HX711 chip
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  //Telling it which pins to use
  scale.set_scale(calibration_factor);
  //Showing the HX711 what 0 weight feels like
  Serial.println("Putting scale to 0, chill");
  scale.tare();
  Serial.println("Scale put to 0! Put anything on");
}

void loop() {
  //is_ready() checks if The HX711 picked out anything yet
  if (scale.is_ready()){
    float weight = scale.get_units(5);
    //Example
    if (weight > 2){
      Serial.print("Piggy Backing Detected!!");
      //Buzzer goes Off
      //Red LED goes Off
    }
    else {
      Serial.println("Access Granted");
      //Green LED ON
    }
  }
  else {
    Serial.println("HX711 not found or not ready.");
  }
}
