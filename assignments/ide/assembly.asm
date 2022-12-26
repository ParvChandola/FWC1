.include "/home/annu/assembly/m328Pdef/m328Pdef.inc"

ldi r16, 0b00001100 ; defining pin 2,3 as output pins and pin 6,7 as input pins
out DDRD, r16

ldi r17, 0b00100000 ; defining pin 13 as output pin
out DDRB, r17

Start:

;generate clock on pin 13

ldi r17, 0b00100000
out PORTB, r17

;creat a delay
call wait 

in r16,PIND

ldi r21, 0b01000000 ;taking value of W
ldi r22, 0b10000000 ;taking value of X

and r21, r16 ;r21 has value of W
and r22, r16 ;r22 has value of X

;right shift W by 6 bits
mov r16, r21
ldi r20, 0b00000110
rcall loopr
mov r21, r16

;right shifting X by 7 bits
mov r16, r22
ldi r20, 0b00000111
rcall loopr
mov r22, r16

;W complement
mov r16, r21
rcall comp
mov r23, r16  ;storing W' in r23

;X complement
mov r16, r22
rcall comp
mov r24, r16  ;storing X' in r24


;calculation of A
mov r01, r23
and r01, r24
mov r02, r23
and r02, r22
or r01, r02
mov r25, r01   ;r25 has A

;calculation of B
mov r01, r23
and r01, r24
mov r02, r21
and r02, r22
or r01, r02
mov r26, r01   ;r26 has B


;left shift B by 3 bits
mov r16, r26
ldi r20, 0b00000011
rcall loopw
mov r26, r16

;left shift A by 2 bits
mov r16, r25
ldi r20, 0b00000010
rcall loopw
mov r25, r16

or r25, r26

;generate clock 0 on pin 13

ldi r17, 0b00000000
out PORTB, r17

;creat a delay
call wait

out PORTD, r25  ;writing the output 


rjmp Start

;loop for bit shifting to left
loopw:	lsl r16			;left shift
	dec r20			;counter --
	brne	loopw	;if counter != 0
	ret

;loop for bit shifting to right
loopr:	lsr r16			;right shift
	dec r20			;counter --
	brne	loopr	;if counter != 0
	ret

;comp routine begins
comp:
	mov r0,r16			;using r0 for computations
	ldi r16,0b00000001	;loading 1
	
	eor r16,r0			;A'=A XOR 1
	ret 				;returning from comp


;delay routine
wait:
  push r16		;save register contents
  push r17		
  push r18		

  ldi r16, 0x20		;loop 0x400000 times
  ldi r17, 0x00		;12 million cycles
  ldi r18, 0x00		;0.7s at 16 MHz

w0:
  
  dec r18
  brne w0
  dec r17
  brne w0
  dec r16
  brne w0

  pop r18		;restore register contents
  pop r17	
  pop r16

  ret






































