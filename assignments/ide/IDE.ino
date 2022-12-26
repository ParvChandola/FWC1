#include<Arduino.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>

//Declaring all variables as integers
int X,W;
int A,B;

//Code released under GNU GPL.  Free to use for anything.
void disp_7447( int B, int A)
{
  digitalWrite(2, A); //LSB
  digitalWrite(3, B); //MSB
}
// the setup function runs once when you press reset or power the board
void setup() {
    pinMode(2, OUTPUT);  
    pinMode(3, OUTPUT);
    pinMode(6, INPUT);
    pinMode(7, INPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    
}

// the loop function runs over and over again forever
void loop() {


  digitalWrite(LED_BUILTIN,HIGH);
  delay(2000);
  W = digitalRead(6);
  X = digitalRead(7);
  A = (!W&&!X) || (!W&&X);
  B = (!W&&!X) || (W&&X);
  digitalWrite(LED_BUILTIN,LOW);
  delay(2000);
  
disp_7447(B,A);  


}
