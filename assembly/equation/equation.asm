;program to complement a boolean

 
.include "/home/annu/assembly/m328Pdef/m328Pdef.inc"

ldi r16, 0b00111100 ;identifying output pins 2,3,4,5
out DDRD,r16		;declaring pins as output

ldi r16,0b00000001	;Z=0
ldi r20,0b00000001      ;storing Z'
rcall comp1	;jumpting to comp1 routine below
ldi r17,0b00000000      ;Y=0
ldi r21,0b00000000	;storing Y'
rcall comp2	;jumping to comp2 routine
ldi r18,0b00000000	;X=0
ldi r22,0b00000001	;storing X'
rcall comp3	;jumping to comp3 routine
ldi r19,0b00000000	;W=0
ldi r23,0b00000000	;storing W'
rcall comp4	;jumping to comp4 routine

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

