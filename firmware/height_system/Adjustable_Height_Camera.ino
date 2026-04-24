#include <Stepper.h>

//Pins for HC-SR04 and Button
#define buttonPin 4
#define trigPin 9
#define echoPin 10
//Stepper Motor Setup
const int stepsPerRevolution = 2048;
const int stepsPerCm = 50;
const int homeCameraHeight = 140;
//Creating motor object using Stepper Library
Stepper myCameraMotor(stepsPerRevolution,5,6,7,8);

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buttonPin,INPUT_PULLUP);
  //Motor speed in RPM
  myCameraMotor.setSpeed(60);
  Serial.println("Ready,Waiting for you!");
}

void loop() {
  //Check if Button is Pressed
  int buttonState = digitalRead(buttonPin);
  
  if (buttonState == LOW){
    Serial.println("Button Pressed! Measuring Height!!");
    //Measuring the Height
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10); 
    digitalWrite(trigPin, LOW);
    long duration = pulseIn(echoPin, HIGH);
    int distance = duration * (0.034 / 2);
    
    Serial.print("Distance to Head: ");
    Serial.print(distance);
    Serial.println(" cm");
    
    //Calculating Required Steps
    // BUG FIX APPLIED HERE: changed 'distanceToHead' to 'distance'
    int fanHeight = 200 - distance;
    
    //Calculate how far the camera needs to travel
    int distanceToMove = fanHeight - homeCameraHeight;
    //Calculating needed steps
    int requiredSteps = distanceToMove * stepsPerCm;
    
    Serial.print("Moving Camera by steps: ");
    Serial.println(requiredSteps);
    //Moving the Motor
    Serial.println("Adjusting Camera Position for you!");
    myCameraMotor.step(requiredSteps);
    Serial.println("Camera Locked. Ready for AI Scan!");
    delay(5000);
  }
}