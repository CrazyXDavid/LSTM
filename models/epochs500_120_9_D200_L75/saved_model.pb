
×
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.11.02v2.11.0-rc2-15-g6290819256d8Æ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0

Adam/v/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_19/bias
y
(Adam/v/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/bias*
_output_shapes
:*
dtype0

Adam/m/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_19/bias
y
(Adam/m/dense_19/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/bias*
_output_shapes
:*
dtype0

Adam/v/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K*'
shared_nameAdam/v/dense_19/kernel

*Adam/v/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_19/kernel*
_output_shapes

:K*
dtype0

Adam/m/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K*'
shared_nameAdam/m/dense_19/kernel

*Adam/m/dense_19/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_19/kernel*
_output_shapes

:K*
dtype0

Adam/v/lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*/
shared_name Adam/v/lstm_9/lstm_cell_9/bias

2Adam/v/lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_9/lstm_cell_9/bias*
_output_shapes	
:¬*
dtype0

Adam/m/lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*/
shared_name Adam/m/lstm_9/lstm_cell_9/bias

2Adam/m/lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_9/lstm_cell_9/bias*
_output_shapes	
:¬*
dtype0
±
*Adam/v/lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K¬*;
shared_name,*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel
ª
>Adam/v/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	K¬*
dtype0
±
*Adam/m/lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K¬*;
shared_name,*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel
ª
>Adam/m/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	K¬*
dtype0

 Adam/v/lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È¬*1
shared_name" Adam/v/lstm_9/lstm_cell_9/kernel

4Adam/v/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOp Adam/v/lstm_9/lstm_cell_9/kernel* 
_output_shapes
:
È¬*
dtype0

 Adam/m/lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È¬*1
shared_name" Adam/m/lstm_9/lstm_cell_9/kernel

4Adam/m/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOp Adam/m/lstm_9/lstm_cell_9/kernel* 
_output_shapes
:
È¬*
dtype0

Adam/v/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*%
shared_nameAdam/v/dense_18/bias
z
(Adam/v/dense_18/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_18/bias*
_output_shapes	
:È*
dtype0

Adam/m/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*%
shared_nameAdam/m/dense_18/bias
z
(Adam/m/dense_18/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_18/bias*
_output_shapes	
:È*
dtype0

Adam/v/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		È*'
shared_nameAdam/v/dense_18/kernel

*Adam/v/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_18/kernel*
_output_shapes
:		È*
dtype0

Adam/m/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		È*'
shared_nameAdam/m/dense_18/kernel

*Adam/m/dense_18/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_18/kernel*
_output_shapes
:		È*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*(
shared_namelstm_9/lstm_cell_9/bias

+lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/bias*
_output_shapes	
:¬*
dtype0
£
#lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	K¬*4
shared_name%#lstm_9/lstm_cell_9/recurrent_kernel

7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	K¬*
dtype0

lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È¬**
shared_namelstm_9/lstm_cell_9/kernel

-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/kernel* 
_output_shapes
:
È¬*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:K* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:K*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:È*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		È* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:		È*
dtype0

serving_default_dense_18_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿx	
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_18_inputdense_18/kerneldense_18/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biasdense_19/kerneldense_19/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_643929

NoOpNoOp
Ì7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*7
valueý6Bú6 Bó6
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¦
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
5
0
1
&2
'3
(4
$5
%6*
5
0
1
&2
'3
(4
$5
%6*
* 
°
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
.trace_0
/trace_1
0trace_2
1trace_3* 
6
2trace_0
3trace_1
4trace_2
5trace_3* 
* 

6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla*

=serving_default* 

0
1*

0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
_Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1
(2*

&0
'1
(2*
* 


Estates
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
* 
ã
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator
Z
state_size

&kernel
'recurrent_kernel
(bias*
* 

$0
%1*

$0
%1*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

`trace_0* 

atrace_0* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_19/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_9/lstm_cell_9/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_9/lstm_cell_9/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_9/lstm_cell_9/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

b0
c1
d2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
r
70
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
e0
g1
i2
k3
m4
o5
q6*
5
f0
h1
j2
l3
n4
p5
r6*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1
(2*

&0
'1
(2*
* 

snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

xtrace_0
ytrace_1* 

ztrace_0
{trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
|	variables
}	keras_api
	~total
	count*
M
	variables
	keras_api

total

count

_fn_kwargs*
M
	variables
	keras_api

total

count

_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_18/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_18/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_18/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_18/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_9/lstm_cell_9/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_9/lstm_cell_9/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_9/lstm_cell_9/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_9/lstm_cell_9/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_19/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_19/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_19/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_19/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

~0
1*

|	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
õ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp+lstm_9/lstm_cell_9/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/dense_18/kernel/Read/ReadVariableOp*Adam/v/dense_18/kernel/Read/ReadVariableOp(Adam/m/dense_18/bias/Read/ReadVariableOp(Adam/v/dense_18/bias/Read/ReadVariableOp4Adam/m/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp4Adam/v/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp>Adam/m/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp>Adam/v/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp2Adam/m/lstm_9/lstm_cell_9/bias/Read/ReadVariableOp2Adam/v/lstm_9/lstm_cell_9/bias/Read/ReadVariableOp*Adam/m/dense_19/kernel/Read/ReadVariableOp*Adam/v/dense_19/kernel/Read/ReadVariableOp(Adam/m/dense_19/bias/Read/ReadVariableOp(Adam/v/dense_19/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_645211
¬
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/biasdense_19/kerneldense_19/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/bias	iterationlearning_rateAdam/m/dense_18/kernelAdam/v/dense_18/kernelAdam/m/dense_18/biasAdam/v/dense_18/bias Adam/m/lstm_9/lstm_cell_9/kernel Adam/v/lstm_9/lstm_cell_9/kernel*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel*Adam/v/lstm_9/lstm_cell_9/recurrent_kernelAdam/m/lstm_9/lstm_cell_9/biasAdam/v/lstm_9/lstm_cell_9/biasAdam/m/dense_19/kernelAdam/v/dense_19/kernelAdam/m/dense_19/biasAdam/v/dense_19/biastotal_2count_2total_1count_1totalcount*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_645308Ó»
à

G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645069

inputs
states_0
states_12
matmul_readvariableop_resource:
È¬3
 matmul_1_readvariableop_resource:	K¬.
biasadd_readvariableop_resource:	¬
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_1
ø
µ
'__inference_lstm_9_layer_call_fn_644393

inputs
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs
A
©

lstm_9_while_body_644230*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0M
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬N
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬I
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorK
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:
È¬L
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬G
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   Ê
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0ª
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0Í
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬­
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0´
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬°
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¹
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_split
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKª
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK}
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK®
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKy
7lstm_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1@lstm_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: n
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKé
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"Ä
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
ì
ö
,__inference_lstm_cell_9_layer_call_fn_645037

inputs
states_0
states_1
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity

identity_1

identity_2¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643231o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_1
è8
É
while_body_644464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
æ
Ó
%sequential_9_lstm_9_while_cond_642924D
@sequential_9_lstm_9_while_sequential_9_lstm_9_while_loop_counterJ
Fsequential_9_lstm_9_while_sequential_9_lstm_9_while_maximum_iterations)
%sequential_9_lstm_9_while_placeholder+
'sequential_9_lstm_9_while_placeholder_1+
'sequential_9_lstm_9_while_placeholder_2+
'sequential_9_lstm_9_while_placeholder_3F
Bsequential_9_lstm_9_while_less_sequential_9_lstm_9_strided_slice_1\
Xsequential_9_lstm_9_while_sequential_9_lstm_9_while_cond_642924___redundant_placeholder0\
Xsequential_9_lstm_9_while_sequential_9_lstm_9_while_cond_642924___redundant_placeholder1\
Xsequential_9_lstm_9_while_sequential_9_lstm_9_while_cond_642924___redundant_placeholder2\
Xsequential_9_lstm_9_while_sequential_9_lstm_9_while_cond_642924___redundant_placeholder3&
"sequential_9_lstm_9_while_identity
²
sequential_9/lstm_9/while/LessLess%sequential_9_lstm_9_while_placeholderBsequential_9_lstm_9_while_less_sequential_9_lstm_9_strided_slice_1*
T0*
_output_shapes
: s
"sequential_9/lstm_9/while/IdentityIdentity"sequential_9/lstm_9/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_9_lstm_9_while_identity+sequential_9/lstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
óJ

B__inference_lstm_9_layer_call_and_return_conditional_losses_643767

inputs>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_643682*
condR
while_cond_643681*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs
µ
Ã
while_cond_643097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_643097___redundant_placeholder04
0while_while_cond_643097___redundant_placeholder14
0while_while_cond_643097___redundant_placeholder24
0while_while_cond_643097___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_643681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_643681___redundant_placeholder04
0while_while_cond_643681___redundant_placeholder14
0while_while_cond_643681___redundant_placeholder24
0while_while_cond_643681___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
è8
É
while_body_644754
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
µ
·
H__inference_sequential_9_layer_call_and_return_conditional_losses_643582

inputs"
dense_18_643408:		È
dense_18_643410:	È!
lstm_9_643558:
È¬ 
lstm_9_643560:	K¬
lstm_9_643562:	¬!
dense_19_643576:K
dense_19_643578:
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCallõ
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputsdense_18_643408dense_18_643410*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_643407
lstm_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0lstm_9_643558lstm_9_643560lstm_9_643562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643557
 dense_19/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_19_643576dense_19_643578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_643575x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
K

B__inference_lstm_9_layer_call_and_return_conditional_losses_644694
inputs_0>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_644609*
condR
while_cond_644608*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs_0
è8
É
while_body_643472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_644608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_644608___redundant_placeholder04
0while_while_cond_644608___redundant_placeholder14
0while_while_cond_644608___redundant_placeholder24
0while_while_cond_644608___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_643290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_643290___redundant_placeholder04
0while_while_cond_643290___redundant_placeholder14
0while_while_cond_643290___redundant_placeholder24
0while_while_cond_643290___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
¬	
©
-__inference_sequential_9_layer_call_fn_643967

inputs
unknown:		È
	unknown_0:	È
	unknown_1:
È¬
	unknown_2:	K¬
	unknown_3:	¬
	unknown_4:K
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_643828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
óJ

B__inference_lstm_9_layer_call_and_return_conditional_losses_644984

inputs>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_644899*
condR
while_cond_644898*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs
í#
Þ
while_body_643098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_9_643122_0:
È¬-
while_lstm_cell_9_643124_0:	K¬)
while_lstm_cell_9_643126_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_9_643122:
È¬+
while_lstm_cell_9_643124:	K¬'
while_lstm_cell_9_643126:	¬¢)while/lstm_cell_9/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0®
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_643122_0while_lstm_cell_9_643124_0while_lstm_cell_9_643126_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643083r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_9_643122while_lstm_cell_9_643122_0"6
while_lstm_cell_9_643124while_lstm_cell_9_643124_0"6
while_lstm_cell_9_643126while_lstm_cell_9_643126_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
Ä	
±
-__inference_sequential_9_layer_call_fn_643599
dense_18_input
unknown:		È
	unknown_0:	È
	unknown_1:
È¬
	unknown_2:	K¬
	unknown_3:	¬
	unknown_4:K
	unknown_5:
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCalldense_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_643582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
¬	
©
-__inference_sequential_9_layer_call_fn_643948

inputs
unknown:		È
	unknown_0:	È
	unknown_1:
È¬
	unknown_2:	K¬
	unknown_3:	¬
	unknown_4:K
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_643582o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
ò8

B__inference_lstm_9_layer_call_and_return_conditional_losses_643168

inputs&
lstm_cell_9_643084:
È¬%
lstm_cell_9_643086:	K¬!
lstm_cell_9_643088:	¬
identity¢#lstm_cell_9/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskð
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_643084lstm_cell_9_643086lstm_cell_9_643088*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643083n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ´
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_643084lstm_cell_9_643086lstm_cell_9_643088*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_643098*
condR
while_cond_643097*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKt
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
óJ

B__inference_lstm_9_layer_call_and_return_conditional_losses_644839

inputs>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_644754*
condR
while_cond_644753*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs
Ûx
À
H__inference_sequential_9_layer_call_and_return_conditional_losses_644144

inputs=
*dense_18_tensordot_readvariableop_resource:		È7
(dense_18_biasadd_readvariableop_resource:	ÈE
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:
È¬F
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	K¬A
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	¬9
'dense_19_matmul_readvariableop_resource:K6
(dense_19_biasadd_readvariableop_resource:
identity¢dense_18/BiasAdd/ReadVariableOp¢!dense_18/Tensordot/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢(lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢lstm_9/while
!dense_18/Tensordot/ReadVariableOpReadVariableOp*dense_18_tensordot_readvariableop_resource*
_output_shapes
:		È*
dtype0a
dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_18/Tensordot/GatherV2GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/free:output:0)dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_18/Tensordot/GatherV2_1GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/axes:output:0+dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/ProdProd$dense_18/Tensordot/GatherV2:output:0!dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/Prod_1Prod&dense_18/Tensordot/GatherV2_1:output:0#dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_18/Tensordot/concatConcatV2 dense_18/Tensordot/free:output:0 dense_18/Tensordot/axes:output:0'dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/stackPack dense_18/Tensordot/Prod:output:0"dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/transpose	Transposeinputs"dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_18/Tensordot/ReshapeReshape dense_18/Tensordot/transpose:y:0!dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_18/Tensordot/MatMulMatMul#dense_18/Tensordot/Reshape:output:0)dense_18/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Èb
 dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_18/Tensordot/concat_1ConcatV2$dense_18/Tensordot/GatherV2:output:0#dense_18/Tensordot/Const_2:output:0)dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/TensordotReshape#dense_18/Tensordot/MatMul:product:0$dense_18/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
dense_18/BiasAddBiasAdddense_18/Tensordot:output:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈU
lstm_9/ShapeShapedense_18/BiasAdd:output:0*
T0*
_output_shapes
:d
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKY
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_9/transpose	Transposedense_18/BiasAdd:output:0lstm_9/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈR
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:f
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   õ
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0©
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0£
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0§
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitz
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK|
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKt
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK|
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   e
#lstm_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0,lstm_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : á
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_644053*$
condR
lstm_9_while_cond_644052*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ë
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementso
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maskl
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿKb
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:K*
dtype0
dense_19/MatMulMatMullstm_9/strided_slice_3:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/Tensordot/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/Tensordot/ReadVariableOp!dense_18/Tensordot/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
K

B__inference_lstm_9_layer_call_and_return_conditional_losses_644549
inputs_0>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_644464*
condR
while_cond_644463*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs_0
ò8

B__inference_lstm_9_layer_call_and_return_conditional_losses_643361

inputs&
lstm_cell_9_643277:
È¬%
lstm_cell_9_643279:	K¬!
lstm_cell_9_643281:	¬
identity¢#lstm_cell_9/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_maskð
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_643277lstm_cell_9_643279lstm_cell_9_643281*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643231n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ´
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_643277lstm_cell_9_643279lstm_cell_9_643281*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_643291*
condR
while_cond_643290*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKt
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ø

G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643231

inputs

states
states_12
matmul_readvariableop_resource:
È¬3
 matmul_1_readvariableop_resource:	K¬.
biasadd_readvariableop_resource:	¬
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_namestates
Ö

)__inference_dense_18_layer_call_fn_644330

inputs
unknown:		È
	unknown_0:	È
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_643407t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
Ûx
À
H__inference_sequential_9_layer_call_and_return_conditional_losses_644321

inputs=
*dense_18_tensordot_readvariableop_resource:		È7
(dense_18_biasadd_readvariableop_resource:	ÈE
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:
È¬F
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	K¬A
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	¬9
'dense_19_matmul_readvariableop_resource:K6
(dense_19_biasadd_readvariableop_resource:
identity¢dense_18/BiasAdd/ReadVariableOp¢!dense_18/Tensordot/ReadVariableOp¢dense_19/BiasAdd/ReadVariableOp¢dense_19/MatMul/ReadVariableOp¢)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢(lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢lstm_9/while
!dense_18/Tensordot/ReadVariableOpReadVariableOp*dense_18_tensordot_readvariableop_resource*
_output_shapes
:		È*
dtype0a
dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_18/Tensordot/GatherV2GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/free:output:0)dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_18/Tensordot/GatherV2_1GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/axes:output:0+dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/ProdProd$dense_18/Tensordot/GatherV2:output:0!dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/Prod_1Prod&dense_18/Tensordot/GatherV2_1:output:0#dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_18/Tensordot/concatConcatV2 dense_18/Tensordot/free:output:0 dense_18/Tensordot/axes:output:0'dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/stackPack dense_18/Tensordot/Prod:output:0"dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/transpose	Transposeinputs"dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_18/Tensordot/ReshapeReshape dense_18/Tensordot/transpose:y:0!dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_18/Tensordot/MatMulMatMul#dense_18/Tensordot/Reshape:output:0)dense_18/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈe
dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Èb
 dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_18/Tensordot/concat_1ConcatV2$dense_18/Tensordot/GatherV2:output:0#dense_18/Tensordot/Const_2:output:0)dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/TensordotReshape#dense_18/Tensordot/MatMul:product:0$dense_18/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0
dense_18/BiasAddBiasAdddense_18/Tensordot:output:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈU
lstm_9/ShapeShapedense_18/BiasAdd:output:0*
T0*
_output_shapes
:d
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKY
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKj
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_9/transpose	Transposedense_18/BiasAdd:output:0lstm_9/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈR
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:f
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   õ
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0©
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0£
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0§
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬d
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitz
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK|
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKt
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK|
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   e
#lstm_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0,lstm_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : á
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_644230*$
condR
lstm_9_while_cond_644229*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ë
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementso
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maskl
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿKb
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:K*
dtype0
dense_19/MatMulMatMullstm_9/strided_slice_3:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/Tensordot/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/Tensordot/ReadVariableOp!dense_18/Tensordot/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
Ä	
±
-__inference_sequential_9_layer_call_fn_643864
dense_18_input
unknown:		È
	unknown_0:	È
	unknown_1:
È¬
	unknown_2:	K¬
	unknown_3:	¬
	unknown_4:K
	unknown_5:
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCalldense_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_643828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
Û>
è
__inference__traced_save_645211
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop8
4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableopB
>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop6
2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_dense_18_kernel_read_readvariableop5
1savev2_adam_v_dense_18_kernel_read_readvariableop3
/savev2_adam_m_dense_18_bias_read_readvariableop3
/savev2_adam_v_dense_18_bias_read_readvariableop?
;savev2_adam_m_lstm_9_lstm_cell_9_kernel_read_readvariableop?
;savev2_adam_v_lstm_9_lstm_cell_9_kernel_read_readvariableopI
Esavev2_adam_m_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableopI
Esavev2_adam_v_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop=
9savev2_adam_m_lstm_9_lstm_cell_9_bias_read_readvariableop=
9savev2_adam_v_lstm_9_lstm_cell_9_bias_read_readvariableop5
1savev2_adam_m_dense_19_kernel_read_readvariableop5
1savev2_adam_v_dense_19_kernel_read_readvariableop3
/savev2_adam_m_dense_19_bias_read_readvariableop3
/savev2_adam_v_dense_19_bias_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ø
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value÷BôB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ý
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableop>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_dense_18_kernel_read_readvariableop1savev2_adam_v_dense_18_kernel_read_readvariableop/savev2_adam_m_dense_18_bias_read_readvariableop/savev2_adam_v_dense_18_bias_read_readvariableop;savev2_adam_m_lstm_9_lstm_cell_9_kernel_read_readvariableop;savev2_adam_v_lstm_9_lstm_cell_9_kernel_read_readvariableopEsavev2_adam_m_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableopEsavev2_adam_v_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop9savev2_adam_m_lstm_9_lstm_cell_9_bias_read_readvariableop9savev2_adam_v_lstm_9_lstm_cell_9_bias_read_readvariableop1savev2_adam_m_dense_19_kernel_read_readvariableop1savev2_adam_v_dense_19_kernel_read_readvariableop/savev2_adam_m_dense_19_bias_read_readvariableop/savev2_adam_v_dense_19_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *,
dtypes"
 2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*é
_input_shapes×
Ô: :		È:È:K::
È¬:	K¬:¬: : :		È:		È:È:È:
È¬:
È¬:	K¬:	K¬:¬:¬:K:K::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:		È:!

_output_shapes	
:È:$ 

_output_shapes

:K: 

_output_shapes
::&"
 
_output_shapes
:
È¬:%!

_output_shapes
:	K¬:!

_output_shapes	
:¬:

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:		È:%!

_output_shapes
:		È:!

_output_shapes	
:È:!

_output_shapes	
:È:&"
 
_output_shapes
:
È¬:&"
 
_output_shapes
:
È¬:%!

_output_shapes
:	K¬:%!

_output_shapes
:	K¬:!

_output_shapes	
:¬:!

_output_shapes	
:¬:$ 

_output_shapes

:K:$ 

_output_shapes

:K: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ç	
õ
D__inference_dense_19_layer_call_and_return_conditional_losses_645003

inputs0
matmul_readvariableop_resource:K-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:K*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
ù	
Ï
lstm_9_while_cond_644229*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_644229___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_644229___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_644229___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_644229___redundant_placeholder3
lstm_9_while_identity
~
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: Y
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
Ø

G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643083

inputs

states
states_12
matmul_readvariableop_resource:
È¬3
 matmul_1_readvariableop_resource:	K¬.
biasadd_readvariableop_resource:	¬
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_namestates
A
©

lstm_9_while_body_644053*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0M
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬N
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬I
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorK
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:
È¬L
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬G
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   Ê
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0ª
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0Í
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬­
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0´
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬°
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬§
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¹
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬j
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_split
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKª
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK}
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK®
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKy
7lstm_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1@lstm_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: n
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKé
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"Ä
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
ù	
Ï
lstm_9_while_cond_644052*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_644052___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_644052___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_644052___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_644052___redundant_placeholder3
lstm_9_while_identity
~
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: Y
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_643471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_643471___redundant_placeholder04
0while_while_cond_643471___redundant_placeholder14
0while_while_cond_643471___redundant_placeholder24
0while_while_cond_643471___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
Ô
ý
D__inference_dense_18_layer_call_and_return_conditional_losses_644360

inputs4
!tensordot_readvariableop_resource:		È.
biasadd_readvariableop_resource:	È
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:		È*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ÈY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
Ô
ý
D__inference_dense_18_layer_call_and_return_conditional_losses_643407

inputs4
!tensordot_readvariableop_resource:		È.
biasadd_readvariableop_resource:	È
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:		È*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ÈY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿx	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
µ
·
H__inference_sequential_9_layer_call_and_return_conditional_losses_643828

inputs"
dense_18_643810:		È
dense_18_643812:	È!
lstm_9_643815:
È¬ 
lstm_9_643817:	K¬
lstm_9_643819:	¬!
dense_19_643822:K
dense_19_643824:
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCallõ
 dense_18/StatefulPartitionedCallStatefulPartitionedCallinputsdense_18_643810dense_18_643812*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_643407
lstm_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0lstm_9_643815lstm_9_643817lstm_9_643819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643767
 dense_19/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_19_643822dense_19_643824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_643575x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
è8
É
while_body_644899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_644463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_644463___redundant_placeholder04
0while_while_cond_644463___redundant_placeholder14
0while_while_cond_644463___redundant_placeholder24
0while_while_cond_644463___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
ì
ö
,__inference_lstm_cell_9_layer_call_fn_645020

inputs
states_0
states_1
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity

identity_1

identity_2¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643083o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_1
í#
Þ
while_body_643291
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_9_643315_0:
È¬-
while_lstm_cell_9_643317_0:	K¬)
while_lstm_cell_9_643319_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_9_643315:
È¬+
while_lstm_cell_9_643317:	K¬'
while_lstm_cell_9_643319:	¬¢)while/lstm_cell_9/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0®
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_643315_0while_lstm_cell_9_643317_0while_lstm_cell_9_643319_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_643231r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_9_643315while_lstm_cell_9_643315_0"6
while_lstm_cell_9_643317while_lstm_cell_9_643317_0"6
while_lstm_cell_9_643319while_lstm_cell_9_643319_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
ø
µ
'__inference_lstm_9_layer_call_fn_644404

inputs
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs

·
'__inference_lstm_9_layer_call_fn_644371
inputs_0
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643168o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs_0
¤
ä
!__inference__wrapped_model_643016
dense_18_inputJ
7sequential_9_dense_18_tensordot_readvariableop_resource:		ÈD
5sequential_9_dense_18_biasadd_readvariableop_resource:	ÈR
>sequential_9_lstm_9_lstm_cell_9_matmul_readvariableop_resource:
È¬S
@sequential_9_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	K¬N
?sequential_9_lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	¬F
4sequential_9_dense_19_matmul_readvariableop_resource:KC
5sequential_9_dense_19_biasadd_readvariableop_resource:
identity¢,sequential_9/dense_18/BiasAdd/ReadVariableOp¢.sequential_9/dense_18/Tensordot/ReadVariableOp¢,sequential_9/dense_19/BiasAdd/ReadVariableOp¢+sequential_9/dense_19/MatMul/ReadVariableOp¢6sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢5sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢7sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢sequential_9/lstm_9/while§
.sequential_9/dense_18/Tensordot/ReadVariableOpReadVariableOp7sequential_9_dense_18_tensordot_readvariableop_resource*
_output_shapes
:		È*
dtype0n
$sequential_9/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_9/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
%sequential_9/dense_18/Tensordot/ShapeShapedense_18_input*
T0*
_output_shapes
:o
-sequential_9/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(sequential_9/dense_18/Tensordot/GatherV2GatherV2.sequential_9/dense_18/Tensordot/Shape:output:0-sequential_9/dense_18/Tensordot/free:output:06sequential_9/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_9/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
*sequential_9/dense_18/Tensordot/GatherV2_1GatherV2.sequential_9/dense_18/Tensordot/Shape:output:0-sequential_9/dense_18/Tensordot/axes:output:08sequential_9/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_9/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
$sequential_9/dense_18/Tensordot/ProdProd1sequential_9/dense_18/Tensordot/GatherV2:output:0.sequential_9/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_9/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¶
&sequential_9/dense_18/Tensordot/Prod_1Prod3sequential_9/dense_18/Tensordot/GatherV2_1:output:00sequential_9/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_9/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ô
&sequential_9/dense_18/Tensordot/concatConcatV2-sequential_9/dense_18/Tensordot/free:output:0-sequential_9/dense_18/Tensordot/axes:output:04sequential_9/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:»
%sequential_9/dense_18/Tensordot/stackPack-sequential_9/dense_18/Tensordot/Prod:output:0/sequential_9/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
)sequential_9/dense_18/Tensordot/transpose	Transposedense_18_input/sequential_9/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	Ì
'sequential_9/dense_18/Tensordot/ReshapeReshape-sequential_9/dense_18/Tensordot/transpose:y:0.sequential_9/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
&sequential_9/dense_18/Tensordot/MatMulMatMul0sequential_9/dense_18/Tensordot/Reshape:output:06sequential_9/dense_18/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈr
'sequential_9/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Èo
-sequential_9/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
(sequential_9/dense_18/Tensordot/concat_1ConcatV21sequential_9/dense_18/Tensordot/GatherV2:output:00sequential_9/dense_18/Tensordot/Const_2:output:06sequential_9/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Æ
sequential_9/dense_18/TensordotReshape0sequential_9/dense_18/Tensordot/MatMul:product:01sequential_9/dense_18/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
,sequential_9/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype0¿
sequential_9/dense_18/BiasAddBiasAdd(sequential_9/dense_18/Tensordot:output:04sequential_9/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈo
sequential_9/lstm_9/ShapeShape&sequential_9/dense_18/BiasAdd:output:0*
T0*
_output_shapes
:q
'sequential_9/lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_9/lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_9/lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_9/lstm_9/strided_sliceStridedSlice"sequential_9/lstm_9/Shape:output:00sequential_9/lstm_9/strided_slice/stack:output:02sequential_9/lstm_9/strided_slice/stack_1:output:02sequential_9/lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_9/lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K¯
 sequential_9/lstm_9/zeros/packedPack*sequential_9/lstm_9/strided_slice:output:0+sequential_9/lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_9/lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_9/lstm_9/zerosFill)sequential_9/lstm_9/zeros/packed:output:0(sequential_9/lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
$sequential_9/lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :K³
"sequential_9/lstm_9/zeros_1/packedPack*sequential_9/lstm_9/strided_slice:output:0-sequential_9/lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_9/lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_9/lstm_9/zeros_1Fill+sequential_9/lstm_9/zeros_1/packed:output:0*sequential_9/lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKw
"sequential_9/lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¶
sequential_9/lstm_9/transpose	Transpose&sequential_9/dense_18/BiasAdd:output:0+sequential_9/lstm_9/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈl
sequential_9/lstm_9/Shape_1Shape!sequential_9/lstm_9/transpose:y:0*
T0*
_output_shapes
:s
)sequential_9/lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_9/lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_9/lstm_9/strided_slice_1StridedSlice$sequential_9/lstm_9/Shape_1:output:02sequential_9/lstm_9/strided_slice_1/stack:output:04sequential_9/lstm_9/strided_slice_1/stack_1:output:04sequential_9/lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_9/lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_9/lstm_9/TensorArrayV2TensorListReserve8sequential_9/lstm_9/TensorArrayV2/element_shape:output:0,sequential_9/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_9/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   
;sequential_9/lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_9/lstm_9/transpose:y:0Rsequential_9/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_9/lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_9/lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
#sequential_9/lstm_9/strided_slice_2StridedSlice!sequential_9/lstm_9/transpose:y:02sequential_9/lstm_9/strided_slice_2/stack:output:04sequential_9/lstm_9/strided_slice_2/stack_1:output:04sequential_9/lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask¶
5sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp>sequential_9_lstm_9_lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0Ð
&sequential_9/lstm_9/lstm_cell_9/MatMulMatMul,sequential_9/lstm_9/strided_slice_2:output:0=sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬¹
7sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp@sequential_9_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0Ê
(sequential_9/lstm_9/lstm_cell_9/MatMul_1MatMul"sequential_9/lstm_9/zeros:output:0?sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Å
#sequential_9/lstm_9/lstm_cell_9/addAddV20sequential_9/lstm_9/lstm_cell_9/MatMul:product:02sequential_9/lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬³
6sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_9_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0Î
'sequential_9/lstm_9/lstm_cell_9/BiasAddBiasAdd'sequential_9/lstm_9/lstm_cell_9/add:z:0>sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬q
/sequential_9/lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_9/lstm_9/lstm_cell_9/splitSplit8sequential_9/lstm_9/lstm_cell_9/split/split_dim:output:00sequential_9/lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_split
'sequential_9/lstm_9/lstm_cell_9/SigmoidSigmoid.sequential_9/lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
)sequential_9/lstm_9/lstm_cell_9/Sigmoid_1Sigmoid.sequential_9/lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK±
#sequential_9/lstm_9/lstm_cell_9/mulMul-sequential_9/lstm_9/lstm_cell_9/Sigmoid_1:y:0$sequential_9/lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
$sequential_9/lstm_9/lstm_cell_9/ReluRelu.sequential_9/lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK¿
%sequential_9/lstm_9/lstm_cell_9/mul_1Mul+sequential_9/lstm_9/lstm_cell_9/Sigmoid:y:02sequential_9/lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK´
%sequential_9/lstm_9/lstm_cell_9/add_1AddV2'sequential_9/lstm_9/lstm_cell_9/mul:z:0)sequential_9/lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
)sequential_9/lstm_9/lstm_cell_9/Sigmoid_2Sigmoid.sequential_9/lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
&sequential_9/lstm_9/lstm_cell_9/Relu_1Relu)sequential_9/lstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÃ
%sequential_9/lstm_9/lstm_cell_9/mul_2Mul-sequential_9/lstm_9/lstm_cell_9/Sigmoid_2:y:04sequential_9/lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
1sequential_9/lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   r
0sequential_9/lstm_9/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
#sequential_9/lstm_9/TensorArrayV2_1TensorListReserve:sequential_9/lstm_9/TensorArrayV2_1/element_shape:output:09sequential_9/lstm_9/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_9/lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_9/lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_9/lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_9/lstm_9/whileWhile/sequential_9/lstm_9/while/loop_counter:output:05sequential_9/lstm_9/while/maximum_iterations:output:0!sequential_9/lstm_9/time:output:0,sequential_9/lstm_9/TensorArrayV2_1:handle:0"sequential_9/lstm_9/zeros:output:0$sequential_9/lstm_9/zeros_1:output:0,sequential_9/lstm_9/strided_slice_1:output:0Ksequential_9/lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_9_lstm_9_lstm_cell_9_matmul_readvariableop_resource@sequential_9_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource?sequential_9_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_9_lstm_9_while_body_642925*1
cond)R'
%sequential_9_lstm_9_while_cond_642924*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
Dsequential_9/lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   
6sequential_9/lstm_9/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_9/lstm_9/while:output:3Msequential_9/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elements|
)sequential_9/lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_9/lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_9/lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_9/lstm_9/strided_slice_3StridedSlice?sequential_9/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:02sequential_9/lstm_9/strided_slice_3/stack:output:04sequential_9/lstm_9/strided_slice_3/stack_1:output:04sequential_9/lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_masky
$sequential_9/lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_9/lstm_9/transpose_1	Transpose?sequential_9/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_9/lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
sequential_9/lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *     
+sequential_9/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_9_dense_19_matmul_readvariableop_resource*
_output_shapes

:K*
dtype0»
sequential_9/dense_19/MatMulMatMul,sequential_9/lstm_9/strided_slice_3:output:03sequential_9/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_9/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_9_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_9/dense_19/BiasAddBiasAdd&sequential_9/dense_19/MatMul:product:04sequential_9/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_9/dense_19/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp-^sequential_9/dense_18/BiasAdd/ReadVariableOp/^sequential_9/dense_18/Tensordot/ReadVariableOp-^sequential_9/dense_19/BiasAdd/ReadVariableOp,^sequential_9/dense_19/MatMul/ReadVariableOp7^sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6^sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOp8^sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^sequential_9/lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2\
,sequential_9/dense_18/BiasAdd/ReadVariableOp,sequential_9/dense_18/BiasAdd/ReadVariableOp2`
.sequential_9/dense_18/Tensordot/ReadVariableOp.sequential_9/dense_18/Tensordot/ReadVariableOp2\
,sequential_9/dense_19/BiasAdd/ReadVariableOp,sequential_9/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_9/dense_19/MatMul/ReadVariableOp+sequential_9/dense_19/MatMul/ReadVariableOp2p
6sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6sequential_9/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2n
5sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOp5sequential_9/lstm_9/lstm_cell_9/MatMul/ReadVariableOp2r
7sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp7sequential_9/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp26
sequential_9/lstm_9/whilesequential_9/lstm_9/while:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
Â

)__inference_dense_19_layer_call_fn_644993

inputs
unknown:K
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_643575o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
ÅP
É
%sequential_9_lstm_9_while_body_642925D
@sequential_9_lstm_9_while_sequential_9_lstm_9_while_loop_counterJ
Fsequential_9_lstm_9_while_sequential_9_lstm_9_while_maximum_iterations)
%sequential_9_lstm_9_while_placeholder+
'sequential_9_lstm_9_while_placeholder_1+
'sequential_9_lstm_9_while_placeholder_2+
'sequential_9_lstm_9_while_placeholder_3C
?sequential_9_lstm_9_while_sequential_9_lstm_9_strided_slice_1_0
{sequential_9_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_9_tensorarrayunstack_tensorlistfromtensor_0Z
Fsequential_9_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬[
Hsequential_9_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬V
Gsequential_9_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬&
"sequential_9_lstm_9_while_identity(
$sequential_9_lstm_9_while_identity_1(
$sequential_9_lstm_9_while_identity_2(
$sequential_9_lstm_9_while_identity_3(
$sequential_9_lstm_9_while_identity_4(
$sequential_9_lstm_9_while_identity_5A
=sequential_9_lstm_9_while_sequential_9_lstm_9_strided_slice_1}
ysequential_9_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_9_tensorarrayunstack_tensorlistfromtensorX
Dsequential_9_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:
È¬Y
Fsequential_9_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬T
Esequential_9_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢<sequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢;sequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢=sequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp
Ksequential_9/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   
=sequential_9/lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_9_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_9_tensorarrayunstack_tensorlistfromtensor_0%sequential_9_lstm_9_while_placeholderTsequential_9/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0Ä
;sequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOpFsequential_9_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0ô
,sequential_9/lstm_9/while/lstm_cell_9/MatMulMatMulDsequential_9/lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Ç
=sequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOpHsequential_9_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0Û
.sequential_9/lstm_9/while/lstm_cell_9/MatMul_1MatMul'sequential_9_lstm_9_while_placeholder_2Esequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬×
)sequential_9/lstm_9/while/lstm_cell_9/addAddV26sequential_9/lstm_9/while/lstm_cell_9/MatMul:product:08sequential_9/lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Á
<sequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOpGsequential_9_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0à
-sequential_9/lstm_9/while/lstm_cell_9/BiasAddBiasAdd-sequential_9/lstm_9/while/lstm_cell_9/add:z:0Dsequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬w
5sequential_9/lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
+sequential_9/lstm_9/while/lstm_cell_9/splitSplit>sequential_9/lstm_9/while/lstm_cell_9/split/split_dim:output:06sequential_9/lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_split 
-sequential_9/lstm_9/while/lstm_cell_9/SigmoidSigmoid4sequential_9/lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK¢
/sequential_9/lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid4sequential_9/lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÀ
)sequential_9/lstm_9/while/lstm_cell_9/mulMul3sequential_9/lstm_9/while/lstm_cell_9/Sigmoid_1:y:0'sequential_9_lstm_9_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
*sequential_9/lstm_9/while/lstm_cell_9/ReluRelu4sequential_9/lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÑ
+sequential_9/lstm_9/while/lstm_cell_9/mul_1Mul1sequential_9/lstm_9/while/lstm_cell_9/Sigmoid:y:08sequential_9/lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÆ
+sequential_9/lstm_9/while/lstm_cell_9/add_1AddV2-sequential_9/lstm_9/while/lstm_cell_9/mul:z:0/sequential_9/lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK¢
/sequential_9/lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid4sequential_9/lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
,sequential_9/lstm_9/while/lstm_cell_9/Relu_1Relu/sequential_9/lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÕ
+sequential_9/lstm_9/while/lstm_cell_9/mul_2Mul3sequential_9/lstm_9/while/lstm_cell_9/Sigmoid_2:y:0:sequential_9/lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Dsequential_9/lstm_9/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ¼
>sequential_9/lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_9_lstm_9_while_placeholder_1Msequential_9/lstm_9/while/TensorArrayV2Write/TensorListSetItem/index:output:0/sequential_9/lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_9/lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_9/lstm_9/while/addAddV2%sequential_9_lstm_9_while_placeholder(sequential_9/lstm_9/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_9/lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_9/lstm_9/while/add_1AddV2@sequential_9_lstm_9_while_sequential_9_lstm_9_while_loop_counter*sequential_9/lstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_9/lstm_9/while/IdentityIdentity#sequential_9/lstm_9/while/add_1:z:0^sequential_9/lstm_9/while/NoOp*
T0*
_output_shapes
: º
$sequential_9/lstm_9/while/Identity_1IdentityFsequential_9_lstm_9_while_sequential_9_lstm_9_while_maximum_iterations^sequential_9/lstm_9/while/NoOp*
T0*
_output_shapes
: 
$sequential_9/lstm_9/while/Identity_2Identity!sequential_9/lstm_9/while/add:z:0^sequential_9/lstm_9/while/NoOp*
T0*
_output_shapes
: Â
$sequential_9/lstm_9/while/Identity_3IdentityNsequential_9/lstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_9/lstm_9/while/NoOp*
T0*
_output_shapes
: ´
$sequential_9/lstm_9/while/Identity_4Identity/sequential_9/lstm_9/while/lstm_cell_9/mul_2:z:0^sequential_9/lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK´
$sequential_9/lstm_9/while/Identity_5Identity/sequential_9/lstm_9/while/lstm_cell_9/add_1:z:0^sequential_9/lstm_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
sequential_9/lstm_9/while/NoOpNoOp=^sequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<^sequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp>^sequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_9_lstm_9_while_identity+sequential_9/lstm_9/while/Identity:output:0"U
$sequential_9_lstm_9_while_identity_1-sequential_9/lstm_9/while/Identity_1:output:0"U
$sequential_9_lstm_9_while_identity_2-sequential_9/lstm_9/while/Identity_2:output:0"U
$sequential_9_lstm_9_while_identity_3-sequential_9/lstm_9/while/Identity_3:output:0"U
$sequential_9_lstm_9_while_identity_4-sequential_9/lstm_9/while/Identity_4:output:0"U
$sequential_9_lstm_9_while_identity_5-sequential_9/lstm_9/while/Identity_5:output:0"
Esequential_9_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resourceGsequential_9_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"
Fsequential_9_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resourceHsequential_9_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"
Dsequential_9_lstm_9_while_lstm_cell_9_matmul_readvariableop_resourceFsequential_9_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"
=sequential_9_lstm_9_while_sequential_9_lstm_9_strided_slice_1?sequential_9_lstm_9_while_sequential_9_lstm_9_strided_slice_1_0"ø
ysequential_9_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_9_tensorarrayunstack_tensorlistfromtensor{sequential_9_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2|
<sequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<sequential_9/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2z
;sequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp;sequential_9/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2~
=sequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp=sequential_9/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_644753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_644753___redundant_placeholder04
0while_while_cond_644753___redundant_placeholder14
0while_while_cond_644753___redundant_placeholder24
0while_while_cond_644753___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
è8
É
while_body_643682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
à

G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645101

inputs
states_0
states_12
matmul_readvariableop_resource:
È¬3
 matmul_1_readvariableop_resource:	K¬.
biasadd_readvariableop_resource:	¬
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
"
_user_specified_name
states_1

·
'__inference_lstm_9_layer_call_fn_644382
inputs_0
unknown:
È¬
	unknown_0:	K¬
	unknown_1:	¬
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643361o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
inputs_0
Ç	
õ
D__inference_dense_19_layer_call_and_return_conditional_losses_643575

inputs0
matmul_readvariableop_resource:K-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:K*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
 
_user_specified_nameinputs
è8
É
while_body_644609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
2while_lstm_cell_9_matmul_readvariableop_resource_0:
È¬G
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	K¬B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	¬
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
0while_lstm_cell_9_matmul_readvariableop_resource:
È¬E
2while_lstm_cell_9_matmul_1_readvariableop_resource:	K¬@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¬¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
element_dtype0
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0* 
_output_shapes
:
È¬*
dtype0¸
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	K¬*
dtype0
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:¬*
dtype0¤
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬c
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKÍ

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_644898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_644898___redundant_placeholder04
0while_while_cond_644898___redundant_placeholder14
0while_while_cond_644898___redundant_placeholder24
0while_while_cond_644898___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK:

_output_shapes
: :

_output_shapes
:
Í
¿
H__inference_sequential_9_layer_call_and_return_conditional_losses_643906
dense_18_input"
dense_18_643888:		È
dense_18_643890:	È!
lstm_9_643893:
È¬ 
lstm_9_643895:	K¬
lstm_9_643897:	¬!
dense_19_643900:K
dense_19_643902:
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCallý
 dense_18/StatefulPartitionedCallStatefulPartitionedCalldense_18_inputdense_18_643888dense_18_643890*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_643407
lstm_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0lstm_9_643893lstm_9_643895lstm_9_643897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643767
 dense_19/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_19_643900dense_19_643902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_643575x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
	
¨
$__inference_signature_wrapper_643929
dense_18_input
unknown:		È
	unknown_0:	È
	unknown_1:
È¬
	unknown_2:	K¬
	unknown_3:	¬
	unknown_4:K
	unknown_5:
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCalldense_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_643016o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
Í
¿
H__inference_sequential_9_layer_call_and_return_conditional_losses_643885
dense_18_input"
dense_18_643867:		È
dense_18_643869:	È!
lstm_9_643872:
È¬ 
lstm_9_643874:	K¬
lstm_9_643876:	¬!
dense_19_643879:K
dense_19_643881:
identity¢ dense_18/StatefulPartitionedCall¢ dense_19/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCallý
 dense_18/StatefulPartitionedCallStatefulPartitionedCalldense_18_inputdense_18_643867dense_18_643869*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_643407
lstm_9/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0lstm_9_643872lstm_9_643874lstm_9_643876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_643557
 dense_19/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0dense_19_643879dense_19_643881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_643575x
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_18_input
¾|
¤
"__inference__traced_restore_645308
file_prefix3
 assignvariableop_dense_18_kernel:		È/
 assignvariableop_1_dense_18_bias:	È4
"assignvariableop_2_dense_19_kernel:K.
 assignvariableop_3_dense_19_bias:@
,assignvariableop_4_lstm_9_lstm_cell_9_kernel:
È¬I
6assignvariableop_5_lstm_9_lstm_cell_9_recurrent_kernel:	K¬9
*assignvariableop_6_lstm_9_lstm_cell_9_bias:	¬&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: <
)assignvariableop_9_adam_m_dense_18_kernel:		È=
*assignvariableop_10_adam_v_dense_18_kernel:		È7
(assignvariableop_11_adam_m_dense_18_bias:	È7
(assignvariableop_12_adam_v_dense_18_bias:	ÈH
4assignvariableop_13_adam_m_lstm_9_lstm_cell_9_kernel:
È¬H
4assignvariableop_14_adam_v_lstm_9_lstm_cell_9_kernel:
È¬Q
>assignvariableop_15_adam_m_lstm_9_lstm_cell_9_recurrent_kernel:	K¬Q
>assignvariableop_16_adam_v_lstm_9_lstm_cell_9_recurrent_kernel:	K¬A
2assignvariableop_17_adam_m_lstm_9_lstm_cell_9_bias:	¬A
2assignvariableop_18_adam_v_lstm_9_lstm_cell_9_bias:	¬<
*assignvariableop_19_adam_m_dense_19_kernel:K<
*assignvariableop_20_adam_v_dense_19_kernel:K6
(assignvariableop_21_adam_m_dense_19_bias:6
(assignvariableop_22_adam_v_dense_19_bias:%
assignvariableop_23_total_2: %
assignvariableop_24_count_2: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: 
identity_30¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Û
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value÷BôB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_19_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_19_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_9_lstm_cell_9_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_5AssignVariableOp6assignvariableop_5_lstm_9_lstm_cell_9_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_6AssignVariableOp*assignvariableop_6_lstm_9_lstm_cell_9_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:³
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_m_dense_18_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_v_dense_18_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_11AssignVariableOp(assignvariableop_11_adam_m_dense_18_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_v_dense_18_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_m_lstm_9_lstm_cell_9_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_v_lstm_9_lstm_cell_9_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_m_lstm_9_lstm_cell_9_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_v_lstm_9_lstm_cell_9_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_m_lstm_9_lstm_cell_9_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_v_lstm_9_lstm_cell_9_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_19_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_19_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_19_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_19_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Í
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: º
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
óJ

B__inference_lstm_9_layer_call_and_return_conditional_losses_643557

inputs>
*lstm_cell_9_matmul_readvariableop_resource:
È¬?
,lstm_cell_9_matmul_1_readvariableop_resource:	K¬:
+lstm_cell_9_biasadd_readvariableop_resource:	¬
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ks
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Kw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿÈD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource* 
_output_shapes
:
È¬*
dtype0
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	K¬*
dtype0
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype0
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿKn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_643472*
condR
while_cond_643471*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿK:ÿÿÿÿÿÿÿÿÿK: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿK   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿK[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK½
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿxÈ: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxÈ
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*½
serving_default©
M
dense_18_input;
 serving_default_dense_18_input:0ÿÿÿÿÿÿÿÿÿx	<
dense_190
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:é´
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
»
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
Q
0
1
&2
'3
(4
$5
%6"
trackable_list_wrapper
Q
0
1
&2
'3
(4
$5
%6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
é
.trace_0
/trace_1
0trace_2
1trace_32þ
-__inference_sequential_9_layer_call_fn_643599
-__inference_sequential_9_layer_call_fn_643948
-__inference_sequential_9_layer_call_fn_643967
-__inference_sequential_9_layer_call_fn_643864¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z.trace_0z/trace_1z0trace_2z1trace_3
Õ
2trace_0
3trace_1
4trace_2
5trace_32ê
H__inference_sequential_9_layer_call_and_return_conditional_losses_644144
H__inference_sequential_9_layer_call_and_return_conditional_losses_644321
H__inference_sequential_9_layer_call_and_return_conditional_losses_643885
H__inference_sequential_9_layer_call_and_return_conditional_losses_643906¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z2trace_0z3trace_1z4trace_2z5trace_3
ÓBÐ
!__inference__wrapped_model_643016dense_18_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla"
experimentalOptimizer
,
=serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
í
Ctrace_02Ð
)__inference_dense_18_layer_call_fn_644330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zCtrace_0

Dtrace_02ë
D__inference_dense_18_layer_call_and_return_conditional_losses_644360¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zDtrace_0
": 		È2dense_18/kernel
:È2dense_18/bias
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Estates
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32û
'__inference_lstm_9_layer_call_fn_644371
'__inference_lstm_9_layer_call_fn_644382
'__inference_lstm_9_layer_call_fn_644393
'__inference_lstm_9_layer_call_fn_644404Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
Ò
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ç
B__inference_lstm_9_layer_call_and_return_conditional_losses_644549
B__inference_lstm_9_layer_call_and_return_conditional_losses_644694
B__inference_lstm_9_layer_call_and_return_conditional_losses_644839
B__inference_lstm_9_layer_call_and_return_conditional_losses_644984Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
"
_generic_user_object
ø
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Y_random_generator
Z
state_size

&kernel
'recurrent_kernel
(bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
í
`trace_02Ð
)__inference_dense_19_layer_call_fn_644993¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z`trace_0

atrace_02ë
D__inference_dense_19_layer_call_and_return_conditional_losses_645003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0
!:K2dense_19/kernel
:2dense_19/bias
-:+
È¬2lstm_9/lstm_cell_9/kernel
6:4	K¬2#lstm_9/lstm_cell_9/recurrent_kernel
&:$¬2lstm_9/lstm_cell_9/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
b0
c1
d2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_sequential_9_layer_call_fn_643599dense_18_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_9_layer_call_fn_643948inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_9_layer_call_fn_643967inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
-__inference_sequential_9_layer_call_fn_643864dense_18_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_9_layer_call_and_return_conditional_losses_644144inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_9_layer_call_and_return_conditional_losses_644321inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¡B
H__inference_sequential_9_layer_call_and_return_conditional_losses_643885dense_18_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¡B
H__inference_sequential_9_layer_call_and_return_conditional_losses_643906dense_18_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

70
e1
f2
g3
h4
i5
j6
k7
l8
m9
n10
o11
p12
q13
r14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
Q
e0
g1
i2
k3
m4
o5
q6"
trackable_list_wrapper
Q
f0
h1
j2
l3
n4
p5
r6"
trackable_list_wrapper
¿2¼¹
®²ª
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
ÒBÏ
$__inference_signature_wrapper_643929dense_18_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_18_layer_call_fn_644330inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_18_layer_call_and_return_conditional_losses_644360inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
'__inference_lstm_9_layer_call_fn_644371inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
'__inference_lstm_9_layer_call_fn_644382inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
'__inference_lstm_9_layer_call_fn_644393inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
'__inference_lstm_9_layer_call_fn_644404inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
B__inference_lstm_9_layer_call_and_return_conditional_losses_644549inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
B__inference_lstm_9_layer_call_and_return_conditional_losses_644694inputs_0"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
B__inference_lstm_9_layer_call_and_return_conditional_losses_644839inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨B¥
B__inference_lstm_9_layer_call_and_return_conditional_losses_644984inputs"Ô
Ë²Ç
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
Ó
xtrace_0
ytrace_12
,__inference_lstm_cell_9_layer_call_fn_645020
,__inference_lstm_cell_9_layer_call_fn_645037½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zxtrace_0zytrace_1

ztrace_0
{trace_12Ò
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645069
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645101½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zztrace_0z{trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_dense_19_layer_call_fn_644993inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_19_layer_call_and_return_conditional_losses_645003inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
|	variables
}	keras_api
	~total
	count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
':%		È2Adam/m/dense_18/kernel
':%		È2Adam/v/dense_18/kernel
!:È2Adam/m/dense_18/bias
!:È2Adam/v/dense_18/bias
2:0
È¬2 Adam/m/lstm_9/lstm_cell_9/kernel
2:0
È¬2 Adam/v/lstm_9/lstm_cell_9/kernel
;:9	K¬2*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel
;:9	K¬2*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel
+:)¬2Adam/m/lstm_9/lstm_cell_9/bias
+:)¬2Adam/v/lstm_9/lstm_cell_9/bias
&:$K2Adam/m/dense_19/kernel
&:$K2Adam/v/dense_19/kernel
 :2Adam/m/dense_19/bias
 :2Adam/v/dense_19/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_lstm_cell_9_layer_call_fn_645020inputsstates_0states_1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
,__inference_lstm_cell_9_layer_call_fn_645037inputsstates_0states_1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645069inputsstates_0states_1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645101inputsstates_0states_1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
~0
1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper 
!__inference__wrapped_model_643016{&'($%;¢8
1¢.
,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	
ª "3ª0
.
dense_19"
dense_19ÿÿÿÿÿÿÿÿÿ´
D__inference_dense_18_layer_call_and_return_conditional_losses_644360l3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿxÈ
 
)__inference_dense_18_layer_call_fn_644330a3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "&#
unknownÿÿÿÿÿÿÿÿÿxÈ«
D__inference_dense_19_layer_call_and_return_conditional_losses_645003c$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿK
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_19_layer_call_fn_644993X$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿK
ª "!
unknownÿÿÿÿÿÿÿÿÿÌ
B__inference_lstm_9_layer_call_and_return_conditional_losses_644549&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿK
 Ì
B__inference_lstm_9_layer_call_and_return_conditional_losses_644694&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿK
 »
B__inference_lstm_9_layer_call_and_return_conditional_losses_644839u&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿxÈ

 
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿK
 »
B__inference_lstm_9_layer_call_and_return_conditional_losses_644984u&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿxÈ

 
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿK
 ¥
'__inference_lstm_9_layer_call_fn_644371z&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿK¥
'__inference_lstm_9_layer_call_fn_644382z&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ

 
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿK
'__inference_lstm_9_layer_call_fn_644393j&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿxÈ

 
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿK
'__inference_lstm_9_layer_call_fn_644404j&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿxÈ

 
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿKá
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645069&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿK
"
states_1ÿÿÿÿÿÿÿÿÿK
p 
ª "¢
~¢{
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿK
SP
&#
tensor_0_1_0ÿÿÿÿÿÿÿÿÿK
&#
tensor_0_1_1ÿÿÿÿÿÿÿÿÿK
 á
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_645101&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿK
"
states_1ÿÿÿÿÿÿÿÿÿK
p
ª "¢
~¢{
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿK
SP
&#
tensor_0_1_0ÿÿÿÿÿÿÿÿÿK
&#
tensor_0_1_1ÿÿÿÿÿÿÿÿÿK
 ´
,__inference_lstm_cell_9_layer_call_fn_645020&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿK
"
states_1ÿÿÿÿÿÿÿÿÿK
p 
ª "x¢u
"
tensor_0ÿÿÿÿÿÿÿÿÿK
OL
$!

tensor_1_0ÿÿÿÿÿÿÿÿÿK
$!

tensor_1_1ÿÿÿÿÿÿÿÿÿK´
,__inference_lstm_cell_9_layer_call_fn_645037&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿÈ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿK
"
states_1ÿÿÿÿÿÿÿÿÿK
p
ª "x¢u
"
tensor_0ÿÿÿÿÿÿÿÿÿK
OL
$!

tensor_1_0ÿÿÿÿÿÿÿÿÿK
$!

tensor_1_1ÿÿÿÿÿÿÿÿÿKÈ
H__inference_sequential_9_layer_call_and_return_conditional_losses_643885|&'($%C¢@
9¢6
,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 È
H__inference_sequential_9_layer_call_and_return_conditional_losses_643906|&'($%C¢@
9¢6
,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_sequential_9_layer_call_and_return_conditional_losses_644144t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_sequential_9_layer_call_and_return_conditional_losses_644321t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¢
-__inference_sequential_9_layer_call_fn_643599q&'($%C¢@
9¢6
,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ¢
-__inference_sequential_9_layer_call_fn_643864q&'($%C¢@
9¢6
,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
-__inference_sequential_9_layer_call_fn_643948i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
-__inference_sequential_9_layer_call_fn_643967i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ¶
$__inference_signature_wrapper_643929&'($%M¢J
¢ 
Cª@
>
dense_18_input,)
dense_18_inputÿÿÿÿÿÿÿÿÿx	"3ª0
.
dense_19"
dense_19ÿÿÿÿÿÿÿÿÿ