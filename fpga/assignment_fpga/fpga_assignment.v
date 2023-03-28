/*Program for sequence counter 
January 16,2021
Code by G V V Sharma
Released under GNU GPL
*/

//declaring the display counter module
module fpga_assignment(
	output wire a,
	output wire b,
	output wire c,
	output wire d,
	output wire e,
	output wire f,
	output wire g
);
    wire clk;

   qlal4s3b_cell_macro u_qlal4s3b_cell_macro (
     .Sys_Clk0 (clk),
  );

    reg[26:0] delay;
    reg   A=0; 
    reg   B=0;
    reg clock1 = 0;
    reg W=0,X=0,Y=0,Z=0;
    reg a=0,b=0,c=0,d=0,e=0,f=0,g=1 ; //Start displaying with 0
    always@( posedge clk) 
    begin
         delay = delay+1; //incrementing the counter.
	    //counts from 0 to i20000000 n 1 second

	  // if(delay==27'b1001100010010110100000000) // blink delay in bits
	 if(delay > 20000000 ) //blink delay in decimal
	 begin
		delay=27'b0;
		//if (clock1 == 1)
        	//begin		
			W = !A;
			X = (!A&!B)|(A&B) ;
			Y = 0; 
        		Z = 0;
               
			//Display the number on 7-segment display connected to output
			//ports
			a=(!Z&!Y&!X&W)|(!Z&Y&!X&!W);
			b=(!Z&Y&!X&W)|(!Z&Y&X&!W);
			c=(!Z&!Y&X&!W);
			d=(!Z&!Y&!X&W)|(!Z&Y&!X&!W)|(!Z&Y&X&W);
			e=(!Z&!Y&!X&W)|(!Z&!Y&X&W)|(!Z&Y&!X&!W)|(!Z&Y&!X&W)|(!Z&Y&X&W)|(Z&!Y&!X&W);
			f=(!Z&!Y&!X&W)|(!Z&!Y&X&!W)|(!Z&!Y&X&W)|(!Z&Y&X&W);
			g=(!Z&!Y&!X&!W)|(!Z&!Y&!X&W)|(!Z&Y&X&W);
		
	        //end
	        clock1 = !clock1;
		
		if (clock1 == 0)
        	begin		
	 		A = W ;
	 		B = X ;
		end
     end
    end
endmodule
//end of the module

