;program to complement a boolean

 
.include "/home/annu/assembly/m328Pdef/m328Pdef.inc"

ldi r29, 0b11000011; pins 10,11,12,13 as input
out DDRB,r29
ldi r29, 0b11111111
out PORTB,r29
in r29,PINB  

ldi r16, 0b00111100 ;identifying output pins 2,3,4,5
out DDRD,r16		;declaring pins as output

mov r0,r29
ldi r16,0b00100000  
and r16,r0  ;taking out Z
ldi r24,0b00000101 ;counter=5
rcall loopw5
mov r20,r16
rcall comp1

mov r0,r29
ldi r17,0b00010000
and r17,r0   ;taking out Y
ldi r24,0b00000100 ;counter=4
rcall loopw6
mov r21,r17
rcall comp2

mov r0,r29
ldi r18,0b00001000
and r18,r0   ;taking out X
ldi r24,0b00000011 ;counter=3
rcall loopw7
mov r22,r18
rcall comp3

mov r0,r29
ldi r19,0b00000100
and r19,r0   ;taking out W
ldi r24,0b00000010
rcall loopw8
mov r23,r19
rcall comp4

; for calculation the equations
;To calculate D

mov r1,r19
and r1,r18
and r1,r17
mov r2,r23
and r2,r16
or r1,r2
mov r25,r1

;to shift the value by 5 places

ldi r24,0b00000101 ;counter=5
rcall loopw1

;to calculate C
 
mov r1,r19
and r1,r18
and r1,r21
mov r2,r22
and r2,r17
mov r3,r23
and r3,r17
or r1,r2
or r1,r3
mov r26,r1

ldi r24,0b00000100  ;counter=4
rcall loopw2

;to calculate B

mov r1,r19
and r1,r22
and r1,r20
mov r2,r23
and r2,r18
or r1,r2
mov r27,r1

ldi r24,0b00000011  ;counter=3
rcall loopw3

;to calculate A

mov r28,r23

ldi r24,0b00000010  ;counter=2
rcall loopw4

or r25,r26
or r25,r27
or r25,r28


;following code is for displaying output

out PORTD,r25		;writing output to pins 2,3,4,5


Start:
rjmp Start

;loop for bit shifting
loopw1:	lsl r25			;left shift
	dec r24			;counter --
	brne	loopw1	;if counter != 0
	ret

loopw2: lsl r26
	dec r24
	brne	loopw2
	ret	

loopw3: lsl r27
	dec r24
	brne	loopw3
	ret

loopw4: lsl r28
	dec r24
	brne	loopw4
	ret

loopw5: lsr r16
	dec r24
	brne	loopw5
	ret

loopw6: lsr r17
	dec r24
	brne	loopw6
	ret

loopw7: lsr r18
	dec r24
	brne	loopw7
	ret

loopw8: lsr r19
	dec r24
	brne	loopw8
	ret


;comp routine begins
comp1:
	mov r0,r20			;using r0 for computations
	ldi r20,0b00000001	;loading 1
	
	eor r20,r0			;A'=A XOR 1
	ret 				;returning from comp
comp2:
        mov r0,r21 
	ldi r21,0b00000001
	eor r21,r0
	ret
comp3:
	mov r0,r22
	ldi r22,0b00000001
	eor r22,r0
	ret
comp4:
	mov r0,r23
	ldi r23,0b00000001
	eor r23,r0
	ret

