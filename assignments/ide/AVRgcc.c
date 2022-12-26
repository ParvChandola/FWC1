//assignment to implement a 2 bit counter for 0->2>1>3->0 state transitions
#include <avr/io.h>

#include <util/delay.h>

#include <stdbool.h>




int main (void)

{
 bool A,B;

 bool W, X ;

//identifyinng pin13 as output 
 DDRB = 0b00100000;

//identifyinng pins 2,3 as ouputs and pins 6,7 as inputs  
 DDRD = 0b00001100;

 while(1){

   
  //Gnerate clock
  PORTB = ((0 <<  PB5));

  _delay_ms (2000L);

  //Read inputs
  W = (PIND & (1 << PIND6)) == (1 << PIND6);

  X = (PIND & (1 << PIND7)) == (1 << PIND7);
  
  //Implement combinational logicto havedesired state trnsitios
  B=(!w&&!X) || (W&&X);
  A=((!W&&!X)|| (!W&&X)) ;

//Write to the output
  PORTD = (A << 2);
  PORTD |= (B << 3);

  //Generate clock
  PORTB = ((1 <<  PB5));
  _delay_ms (2000L);

 }
return 0;

}
