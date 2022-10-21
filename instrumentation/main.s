	.file	"main.c"
	.option nopic
	.attribute arch, "rv32i2p0_m2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	2
	.type	pcap_filter_with_aux_data.part.0, @function
pcap_filter_with_aux_data.part.0:
	lhu	t0,0(a0)
	addi	a4,a0,-8
	lui	a5,%hi(.L7)
	li	a7,177
	li	a0,0
	li	t3,0
	addi	t1,a5,%lo(.L7)
	addi	a6,a4,8
	bgtu	t0,a7,.L104
	addi	sp,sp,-64
.L97:
	slli	t2,t0,2
	add	t4,t2,t1
	lw	t5,0(t4)
	jr	t5
	.section	.rodata
	.align	2
	.align	2
.L7:
	.word	.L53
	.word	.L52
	.word	.L51
	.word	.L50
	.word	.L49
	.word	.L48
	.word	.L47
	.word	.L46
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L45
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L44
	.word	.L43
	.word	.L71
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L41
	.word	.L40
	.word	.L5
	.word	.L5
	.word	.L39
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L38
	.word	.L37
	.word	.L5
	.word	.L5
	.word	.L36
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L35
	.word	.L34
	.word	.L5
	.word	.L5
	.word	.L101
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L33
	.word	.L32
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L31
	.word	.L30
	.word	.L5
	.word	.L5
	.word	.L29
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L28
	.word	.L27
	.word	.L5
	.word	.L5
	.word	.L26
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L25
	.word	.L24
	.word	.L5
	.word	.L5
	.word	.L23
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L22
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L21
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L20
	.word	.L19
	.word	.L5
	.word	.L5
	.word	.L18
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L17
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L16
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L15
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L72
	.word	.L14
	.word	.L5
	.word	.L5
	.word	.L13
	.word	.L5
	.word	.L5
	.word	.L12
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L11
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L10
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L9
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L8
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L5
	.word	.L6
	.text
.L46:
	mv	a0,t3
.L101:
	mv	a4,a6
.L2:
	lhu	t0,8(a4)
	addi	a6,a4,8
	bleu	t0,a7,.L97
.L5:
	li	a0,-1
	li	a7,93
 #APP
# 25 "./example/main.c" 1
	scall
# 0 "" 2
 #NO_APP
.L47:
	lw	a0,12(a4)
.L1:
	addi	sp,sp,64
	jr	ra
.L30:
	bgtu	a0,t3,.L69
.L102:
	lbu	t5,10(a4)
	slli	t4,t5,3
	add	a4,a6,t4
	j	.L2
.L29:
	lw	t3,12(a4)
	add	t0,t3,a0
	bltu	a3,t3,.L85
	sub	a5,a3,t3
	bgtu	a0,a5,.L85
	sub	a4,a3,t0
	li	t2,3
	bleu	a4,t2,.L85
	add	t4,a1,t0
	lw	t3,0(t4)
	mv	a4,a6
	j	.L2
.L28:
	lw	t6,12(a4)
	mv	a4,a6
	or	t3,t3,t6
	j	.L2
.L27:
	lw	t4,12(a4)
	and	t5,t3,t4
	bne	t5,zero,.L102
.L69:
	lbu	a4,11(a4)
	slli	t4,a4,3
	add	a4,a6,t4
	j	.L2
.L26:
	lw	t5,12(a4)
	add	t6,a0,t5
	bltu	a3,a0,.L85
	sub	a5,a3,a0
	bgtu	t5,a5,.L85
	sub	t0,a3,t6
	li	a4,1
	bleu	t0,a4,.L85
	add	t2,a1,t6
	lhu	t3,0(t2)
	mv	a4,a6
	j	.L2
.L25:
	or	t3,t3,a0
	mv	a4,a6
	j	.L2
.L24:
	and	t4,a0,t3
	beq	t4,zero,.L69
	j	.L102
.L23:
	lw	t0,12(a4)
	bleu	a3,t0,.L85
	sub	a4,a3,t0
	bgeu	a0,a4,.L85
	add	t2,a1,t0
	add	t3,t2,a0
	lbu	t3,0(t3)
	mv	a4,a6
	j	.L2
.L37:
	lw	t2,12(a4)
	bleu	t3,t2,.L69
	j	.L102
.L36:
	lw	t0,12(a4)
	bltu	a3,t0,.L85
	sub	a5,a3,t0
	li	t3,1
	bleu	a5,t3,.L85
	add	a4,a1,t0
	lhu	t3,0(a4)
	mv	a4,a6
	j	.L2
.L35:
	mul	t3,t3,a0
	mv	a4,a6
	j	.L2
.L34:
	bgeu	a0,t3,.L69
	j	.L102
.L32:
	lw	t5,12(a4)
	bltu	t3,t5,.L69
	j	.L102
.L31:
	beq	a0,zero,.L1
	divu	t3,t3,a0
	mv	a4,a6
	j	.L2
.L6:
	lw	a0,12(a4)
	bleu	a3,a0,.L85
	add	t6,a1,a0
	lbu	a5,0(t6)
	mv	a4,a6
	slli	t0,a5,2
	andi	a0,t0,60
	j	.L2
.L45:
	add	t3,t3,a0
	mv	a4,a6
	j	.L2
.L44:
	lw	t6,12(a4)
	mv	a4,a6
	sub	t3,t3,t6
	j	.L2
.L43:
	lw	t2,12(a4)
	bne	t3,t2,.L69
	j	.L102
.L41:
	sub	t3,t3,a0
	mv	a4,a6
	j	.L2
.L40:
	bne	a0,t3,.L69
	j	.L102
.L39:
	lw	t5,12(a4)
	bgtu	t5,a3,.L85
	sub	t6,a3,t5
	li	t0,3
	bleu	t6,t0,.L85
	add	a5,a1,t5
	lw	t3,0(a5)
	mv	a4,a6
	j	.L2
.L38:
	lw	t4,12(a4)
	mv	a4,a6
	mul	t3,t3,t4
	j	.L2
.L53:
	lw	t3,12(a4)
	mv	a4,a6
	j	.L2
.L52:
	lw	a0,12(a4)
	mv	a4,a6
	j	.L2
.L51:
	lw	t0,12(a4)
	slli	a5,t0,2
	addi	t2,a5,64
	add	a4,t2,sp
	sw	t3,-64(a4)
	j	.L101
.L50:
	lw	a4,12(a4)
	slli	t4,a4,2
	addi	t5,t4,64
	add	t6,t5,sp
	sw	a0,-64(t6)
	j	.L101
.L49:
	lw	t2,12(a4)
	mv	a4,a6
	add	t3,t3,t2
	j	.L2
.L48:
	lw	t0,12(a4)
	slli	a5,t0,3
	add	a4,a6,a5
	j	.L2
.L33:
	lw	t6,12(a4)
	mv	a4,a6
	divu	t3,t3,t6
	j	.L2
.L22:
	lw	a5,12(a4)
	mv	a4,a6
	and	t3,t3,a5
	j	.L2
.L21:
	and	t3,t3,a0
	mv	a4,a6
	j	.L2
.L20:
	lw	t3,12(a4)
	mv	a4,a6
	slli	t4,t3,2
	addi	t5,t4,64
	add	t6,t5,sp
	lw	t3,-64(t6)
	j	.L2
.L19:
	lw	a0,12(a4)
	mv	a4,a6
	slli	a6,a0,2
	addi	t0,a6,64
	add	t2,t0,sp
	lw	a0,-64(t2)
	j	.L2
.L18:
	lw	a5,12(a4)
	mv	a4,a6
	sll	t3,t3,a5
	j	.L2
.L17:
	li	t6,31
	bgtu	a0,t6,.L87
	sll	t3,t3,a0
	mv	a4,a6
	j	.L2
.L16:
	lw	t5,12(a4)
	mv	a4,a6
	srl	t3,t3,t5
	j	.L2
.L15:
	li	a4,31
	bgtu	a0,a4,.L87
	srl	t3,t3,a0
	mv	a4,a6
	j	.L2
.L14:
	mv	a0,a2
	mv	a4,a6
	j	.L2
.L13:
	neg	t3,t3
	mv	a4,a6
	j	.L2
.L12:
	mv	t3,a0
	mv	a4,a6
	j	.L2
.L11:
	lw	t4,12(a4)
	mv	a4,a6
	remu	t3,t3,t4
	j	.L2
.L10:
	beq	a0,zero,.L1
	remu	t3,t3,a0
	mv	a4,a6
	j	.L2
.L9:
	lw	t2,12(a4)
	mv	a4,a6
	xor	t3,t3,t2
	j	.L2
.L8:
	xor	t3,t3,a0
	mv	a4,a6
	j	.L2
.L72:
	mv	t3,a2
	mv	a4,a6
	j	.L2
.L71:
	mv	a0,t3
	j	.L1
.L85:
	li	a0,0
	addi	sp,sp,64
	jr	ra
.L87:
	mv	a4,a6
	li	t3,0
	j	.L2
.L104:
	li	a0,-1
	li	a7,93
 #APP
# 25 "./example/main.c" 1
	scall
# 0 "" 2
 #NO_APP
	lw	a0,12(a4)
	ret
	.size	pcap_filter_with_aux_data.part.0, .-pcap_filter_with_aux_data.part.0
	.align	2
	.globl	pcap_filter_with_aux_data
	.type	pcap_filter_with_aux_data, @function
pcap_filter_with_aux_data:
	bne	a0,zero,.L107
	li	a0,-1
	ret
.L107:
	tail	pcap_filter_with_aux_data.part.0
	.size	pcap_filter_with_aux_data, .-pcap_filter_with_aux_data
	.align	2
	.globl	pcap_filter
	.type	pcap_filter, @function
pcap_filter:
	bne	a0,zero,.L110
	li	a0,-1
	ret
.L110:
	tail	pcap_filter_with_aux_data.part.0
	.size	pcap_filter, .-pcap_filter
	.align	2
	.globl	_start
	.type	_start, @function
_start:
	lui	a1,%hi(.LANCHOR0)
	lui	a0,%hi(.LANCHOR1)
	addi	sp,sp,-16
	li	a3,800
	li	a2,800
	addi	a1,a1,%lo(.LANCHOR0)
	addi	a0,a0,%lo(.LANCHOR1)
	sw	ra,12(sp)
	call	pcap_filter_with_aux_data.part.0
	li	a7,93
 #APP
# 25 "./example/main.c" 1
	scall
# 0 "" 2
 #NO_APP
	lw	ra,12(sp)
	addi	sp,sp,16
	jr	ra
	.size	_start, .-_start
	.globl	pkt
	.globl	bpfidata
	.section	.fdata,"a"
	.align	2
	.set	.LANCHOR1,. + 0
	.type	bpfidata, @object
	.size	bpfidata, 672
bpfidata:
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002"
	.string	"\b"
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\025"
	.string	"N"
	.string	"\004\003\002\001("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002\006\b"
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\025"
	.string	"J"
	.string	"\004\003\002\001("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\0025\200"
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\025"
	.string	"F"
	.string	"\004\003\002\001("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\006\335\206"
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\026"
	.string	""
	.string	""
	.string	"\025"
	.string	"B"
	.string	"\006"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\026"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002,"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"8"
	.string	""
	.string	""
	.string	"\025"
	.string	">"
	.string	"\006"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002"
	.string	"\b"
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\031"
	.string	""
	.string	""
	.string	"\025"
	.string	":"
	.string	"\006"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\006\335\206"
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\026"
	.string	""
	.string	""
	.string	"\025"
	.string	"6"
	.string	"\021"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\026"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002,"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"8"
	.string	""
	.string	""
	.string	"\025"
	.string	"2"
	.string	"\021"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002"
	.string	"\b"
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\031"
	.string	""
	.string	""
	.string	"\025"
	.string	"."
	.string	"\021"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	","
	.string	""
	.string	"\b"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"*"
	.string	"\335\206"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"("
	.string	"\006\b"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"&"
	.string	"5\200"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"$"
	.string	"\233\200"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\004\004"
	.string	""
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\024"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002\233\200\007"
	.string	" "
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	"\036"
	.string	"\b\003\252\252("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"\034"
	.string	"\363\200"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\004\004"
	.string	""
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\024"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002\363\200"
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	"\026"
	.string	""
	.string	"\003\252\252("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"\024"
	.string	"\003`"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002\004"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	"\020"
	.string	"\376\376"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\002\004"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	"\f"
	.string	"B"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"\n"
	.string	"7\201"
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	"\b"
	.string	"\001"
	.string	""
	.string	""
	.string	"("
	.string	""
	.string	""
	.string	"\016"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\007\004"
	.string	""
	.string	""
	.string	"0"
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	"\004"
	.string	"\340"
	.string	""
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\024"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\0037\201"
	.string	""
	.string	" "
	.string	""
	.string	""
	.string	"\020"
	.string	""
	.string	""
	.string	"\025"
	.string	""
	.string	"\001"
	.string	"\003\252\252\006"
	.string	""
	.string	""
	.string	""
	.string	" "
	.string	""
	.string	"\006"
	.string	""
	.string	""
	.string	""
	.string	""
	.string	""
	.string	""
	.section	.pkt,"a"
	.align	2
	.set	.LANCHOR0,. + 0
	.type	pkt, @object
	.size	pkt, 800
pkt:
	.string	"\001\002\003\004\005\006\007\b\t\020\021\022\023\024\025\026"
	.zero	783
	.ident	"GCC: (g1ea978e3066) 12.1.0"
