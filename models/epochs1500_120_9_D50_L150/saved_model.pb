¼¡
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
"serve*2.11.02v2.11.0-rc2-15-g6290819256d8öË
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
Adam/v/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_17/bias
y
(Adam/v/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/bias*
_output_shapes
:*
dtype0

Adam/m/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_17/bias
y
(Adam/m/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/bias*
_output_shapes
:*
dtype0

Adam/v/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/v/dense_17/kernel

*Adam/v/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/kernel*
_output_shapes
:	*
dtype0

Adam/m/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/m/dense_17/kernel

*Adam/m/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/kernel*
_output_shapes
:	*
dtype0

Adam/v/lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*/
shared_name Adam/v/lstm_8/lstm_cell_8/bias

2Adam/v/lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_8/lstm_cell_8/bias*
_output_shapes	
:Ø*
dtype0

Adam/m/lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*/
shared_name Adam/m/lstm_8/lstm_cell_8/bias

2Adam/m/lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_8/lstm_cell_8/bias*
_output_shapes	
:Ø*
dtype0
²
*Adam/v/lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*;
shared_name,*Adam/v/lstm_8/lstm_cell_8/recurrent_kernel
«
>Adam/v/lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/v/lstm_8/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0
²
*Adam/m/lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*;
shared_name,*Adam/m/lstm_8/lstm_cell_8/recurrent_kernel
«
>Adam/m/lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/m/lstm_8/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0

 Adam/v/lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ø*1
shared_name" Adam/v/lstm_8/lstm_cell_8/kernel

4Adam/v/lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOp Adam/v/lstm_8/lstm_cell_8/kernel*
_output_shapes
:	2Ø*
dtype0

 Adam/m/lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ø*1
shared_name" Adam/m/lstm_8/lstm_cell_8/kernel

4Adam/m/lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOp Adam/m/lstm_8/lstm_cell_8/kernel*
_output_shapes
:	2Ø*
dtype0

Adam/v/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/v/dense_16/bias
y
(Adam/v/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/bias*
_output_shapes
:2*
dtype0

Adam/m/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdam/m/dense_16/bias
y
(Adam/m/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/bias*
_output_shapes
:2*
dtype0

Adam/v/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	2*'
shared_nameAdam/v/dense_16/kernel

*Adam/v/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/kernel*
_output_shapes

:	2*
dtype0

Adam/m/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	2*'
shared_nameAdam/m/dense_16/kernel

*Adam/m/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/kernel*
_output_shapes

:	2*
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
lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ø*(
shared_namelstm_8/lstm_cell_8/bias

+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/bias*
_output_shapes	
:Ø*
dtype0
¤
#lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ø*4
shared_name%#lstm_8/lstm_cell_8/recurrent_kernel

7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_8/lstm_cell_8/recurrent_kernel* 
_output_shapes
:
Ø*
dtype0

lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Ø**
shared_namelstm_8/lstm_cell_8/kernel

-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/kernel*
_output_shapes
:	2Ø*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:2*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	2* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:	2*
dtype0

serving_default_dense_16_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿx	
á
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_16_inputdense_16/kerneldense_16/biaslstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biasdense_17/kerneldense_17/bias*
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
GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1110314

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
VARIABLE_VALUEdense_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_8/lstm_cell_8/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_8/lstm_cell_8/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_8/lstm_cell_8/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/m/dense_16/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_16/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_16/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_16/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_8/lstm_cell_8/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_8/lstm_cell_8/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/lstm_8/lstm_cell_8/recurrent_kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/lstm_8/lstm_cell_8/recurrent_kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_8/lstm_cell_8/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_8/lstm_cell_8/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_17/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_17/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_17/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_17/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
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
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/dense_16/kernel/Read/ReadVariableOp*Adam/v/dense_16/kernel/Read/ReadVariableOp(Adam/m/dense_16/bias/Read/ReadVariableOp(Adam/v/dense_16/bias/Read/ReadVariableOp4Adam/m/lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp4Adam/v/lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp>Adam/m/lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp>Adam/v/lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp2Adam/m/lstm_8/lstm_cell_8/bias/Read/ReadVariableOp2Adam/v/lstm_8/lstm_cell_8/bias/Read/ReadVariableOp*Adam/m/dense_17/kernel/Read/ReadVariableOp*Adam/v/dense_17/kernel/Read/ReadVariableOp(Adam/m/dense_17/bias/Read/ReadVariableOp(Adam/v/dense_17/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst**
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1111596
­
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_16/kerneldense_16/biasdense_17/kerneldense_17/biaslstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/bias	iterationlearning_rateAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/bias Adam/m/lstm_8/lstm_cell_8/kernel Adam/v/lstm_8/lstm_cell_8/kernel*Adam/m/lstm_8/lstm_cell_8/recurrent_kernel*Adam/v/lstm_8/lstm_cell_8/recurrent_kernelAdam/m/lstm_8/lstm_cell_8/biasAdam/v/lstm_8/lstm_cell_8/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biastotal_2count_2total_1count_1totalcount*)
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1111693³Á
9

C__inference_lstm_8_layer_call_and_return_conditional_losses_1109553

inputs&
lstm_cell_8_1109469:	2Ø'
lstm_cell_8_1109471:
Ø"
lstm_cell_8_1109473:	Ø
identity¢#lstm_cell_8/StatefulPartitionedCall¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask÷
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_1109469lstm_cell_8_1109471lstm_cell_8_1109473*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109468n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_1109469lstm_cell_8_1109471lstm_cell_8_1109473*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1109483*
condR
while_cond_1109482*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
û8
Ê
while_body_1110067
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ØP
Ê
&sequential_8_lstm_8_while_body_1109310D
@sequential_8_lstm_8_while_sequential_8_lstm_8_while_loop_counterJ
Fsequential_8_lstm_8_while_sequential_8_lstm_8_while_maximum_iterations)
%sequential_8_lstm_8_while_placeholder+
'sequential_8_lstm_8_while_placeholder_1+
'sequential_8_lstm_8_while_placeholder_2+
'sequential_8_lstm_8_while_placeholder_3C
?sequential_8_lstm_8_while_sequential_8_lstm_8_strided_slice_1_0
{sequential_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	2Ø\
Hsequential_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØV
Gsequential_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø&
"sequential_8_lstm_8_while_identity(
$sequential_8_lstm_8_while_identity_1(
$sequential_8_lstm_8_while_identity_2(
$sequential_8_lstm_8_while_identity_3(
$sequential_8_lstm_8_while_identity_4(
$sequential_8_lstm_8_while_identity_5A
=sequential_8_lstm_8_while_sequential_8_lstm_8_strided_slice_1}
ysequential_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_8_tensorarrayunstack_tensorlistfromtensorW
Dsequential_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	2ØZ
Fsequential_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:
ØT
Esequential_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢<sequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢;sequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢=sequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp
Ksequential_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
=sequential_8/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0%sequential_8_lstm_8_while_placeholderTsequential_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0Ã
;sequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpFsequential_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0ô
,sequential_8/lstm_8/while/lstm_cell_8/MatMulMatMulDsequential_8/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØÈ
=sequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpHsequential_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0Û
.sequential_8/lstm_8/while/lstm_cell_8/MatMul_1MatMul'sequential_8_lstm_8_while_placeholder_2Esequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ×
)sequential_8/lstm_8/while/lstm_cell_8/addAddV26sequential_8/lstm_8/while/lstm_cell_8/MatMul:product:08sequential_8/lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØÁ
<sequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpGsequential_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0à
-sequential_8/lstm_8/while/lstm_cell_8/BiasAddBiasAdd-sequential_8/lstm_8/while/lstm_cell_8/add:z:0Dsequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØw
5sequential_8/lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¬
+sequential_8/lstm_8/while/lstm_cell_8/splitSplit>sequential_8/lstm_8/while/lstm_cell_8/split/split_dim:output:06sequential_8/lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¡
-sequential_8/lstm_8/while/lstm_cell_8/SigmoidSigmoid4sequential_8/lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
/sequential_8/lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid4sequential_8/lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
)sequential_8/lstm_8/while/lstm_cell_8/mulMul3sequential_8/lstm_8/while/lstm_cell_8/Sigmoid_1:y:0'sequential_8_lstm_8_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_8/lstm_8/while/lstm_cell_8/ReluRelu4sequential_8/lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
+sequential_8/lstm_8/while/lstm_cell_8/mul_1Mul1sequential_8/lstm_8/while/lstm_cell_8/Sigmoid:y:08sequential_8/lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
+sequential_8/lstm_8/while/lstm_cell_8/add_1AddV2-sequential_8/lstm_8/while/lstm_cell_8/mul:z:0/sequential_8/lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
/sequential_8/lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid4sequential_8/lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_8/lstm_8/while/lstm_cell_8/Relu_1Relu/sequential_8/lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
+sequential_8/lstm_8/while/lstm_cell_8/mul_2Mul3sequential_8/lstm_8/while/lstm_cell_8/Sigmoid_2:y:0:sequential_8/lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Dsequential_8/lstm_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ¼
>sequential_8/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_8_lstm_8_while_placeholder_1Msequential_8/lstm_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0/sequential_8/lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_8/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_8/lstm_8/while/addAddV2%sequential_8_lstm_8_while_placeholder(sequential_8/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_8/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_8/lstm_8/while/add_1AddV2@sequential_8_lstm_8_while_sequential_8_lstm_8_while_loop_counter*sequential_8/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_8/lstm_8/while/IdentityIdentity#sequential_8/lstm_8/while/add_1:z:0^sequential_8/lstm_8/while/NoOp*
T0*
_output_shapes
: º
$sequential_8/lstm_8/while/Identity_1IdentityFsequential_8_lstm_8_while_sequential_8_lstm_8_while_maximum_iterations^sequential_8/lstm_8/while/NoOp*
T0*
_output_shapes
: 
$sequential_8/lstm_8/while/Identity_2Identity!sequential_8/lstm_8/while/add:z:0^sequential_8/lstm_8/while/NoOp*
T0*
_output_shapes
: Â
$sequential_8/lstm_8/while/Identity_3IdentityNsequential_8/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_8/lstm_8/while/NoOp*
T0*
_output_shapes
: µ
$sequential_8/lstm_8/while/Identity_4Identity/sequential_8/lstm_8/while/lstm_cell_8/mul_2:z:0^sequential_8/lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
$sequential_8/lstm_8/while/Identity_5Identity/sequential_8/lstm_8/while/lstm_cell_8/add_1:z:0^sequential_8/lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_8/lstm_8/while/NoOpNoOp=^sequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<^sequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp>^sequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_8_lstm_8_while_identity+sequential_8/lstm_8/while/Identity:output:0"U
$sequential_8_lstm_8_while_identity_1-sequential_8/lstm_8/while/Identity_1:output:0"U
$sequential_8_lstm_8_while_identity_2-sequential_8/lstm_8/while/Identity_2:output:0"U
$sequential_8_lstm_8_while_identity_3-sequential_8/lstm_8/while/Identity_3:output:0"U
$sequential_8_lstm_8_while_identity_4-sequential_8/lstm_8/while/Identity_4:output:0"U
$sequential_8_lstm_8_while_identity_5-sequential_8/lstm_8/while/Identity_5:output:0"
Esequential_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resourceGsequential_8_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Fsequential_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resourceHsequential_8_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
Dsequential_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resourceFsequential_8_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"
=sequential_8_lstm_8_while_sequential_8_lstm_8_strided_slice_1?sequential_8_lstm_8_while_sequential_8_lstm_8_strided_slice_1_0"ø
ysequential_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_8_tensorarrayunstack_tensorlistfromtensor{sequential_8_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2|
<sequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<sequential_8/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2z
;sequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp;sequential_8/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2~
=sequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp=sequential_8/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¸
(__inference_lstm_8_layer_call_fn_1110756
inputs_0
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1109553p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs_0
ï
Ø
&sequential_8_lstm_8_while_cond_1109309D
@sequential_8_lstm_8_while_sequential_8_lstm_8_while_loop_counterJ
Fsequential_8_lstm_8_while_sequential_8_lstm_8_while_maximum_iterations)
%sequential_8_lstm_8_while_placeholder+
'sequential_8_lstm_8_while_placeholder_1+
'sequential_8_lstm_8_while_placeholder_2+
'sequential_8_lstm_8_while_placeholder_3F
Bsequential_8_lstm_8_while_less_sequential_8_lstm_8_strided_slice_1]
Ysequential_8_lstm_8_while_sequential_8_lstm_8_while_cond_1109309___redundant_placeholder0]
Ysequential_8_lstm_8_while_sequential_8_lstm_8_while_cond_1109309___redundant_placeholder1]
Ysequential_8_lstm_8_while_sequential_8_lstm_8_while_cond_1109309___redundant_placeholder2]
Ysequential_8_lstm_8_while_sequential_8_lstm_8_while_cond_1109309___redundant_placeholder3&
"sequential_8_lstm_8_while_identity
²
sequential_8/lstm_8/while/LessLess%sequential_8_lstm_8_while_placeholderBsequential_8_lstm_8_while_less_sequential_8_lstm_8_strided_slice_1*
T0*
_output_shapes
: s
"sequential_8/lstm_8/while/IdentityIdentity"sequential_8/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_8_lstm_8_while_identity+sequential_8/lstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Þ
Æ
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110291
dense_16_input"
dense_16_1110273:	2
dense_16_1110275:2!
lstm_8_1110278:	2Ø"
lstm_8_1110280:
Ø
lstm_8_1110282:	Ø#
dense_17_1110285:	
dense_17_1110287:
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCallÿ
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_1110273dense_16_1110275*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792¡
lstm_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0lstm_8_1110278lstm_8_1110280lstm_8_1110282*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110152
 dense_17/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_17_1110285dense_17_1110287*
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
GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_16_input
ó

H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111454

inputs
states_0
states_11
matmul_readvariableop_resource:	2Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
ö
÷
-__inference_lstm_cell_8_layer_call_fn_1111422

inputs
states_0
states_1
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109616p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
û8
Ê
while_body_1110849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ö
÷
-__inference_lstm_cell_8_layer_call_fn_1111405

inputs
states_0
states_1
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity

identity_1

identity_2¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109468p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1111224

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1111139*
condR
while_cond_1111138*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
îx
À
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110706

inputs<
*dense_16_tensordot_readvariableop_resource:	26
(dense_16_biasadd_readvariableop_resource:2D
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	2ØG
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:
ØA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	Ø:
'dense_17_matmul_readvariableop_resource:	6
(dense_17_biasadd_readvariableop_resource:
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_8/while
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:	2*
dtype0a
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2d
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2b
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2U
lstm_8/ShapeShapedense_16/BiasAdd:output:0*
T0*
_output_shapes
:d
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_8/transpose	Transposedense_16/BiasAdd:output:0lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2R
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:f
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   õ
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0©
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0£
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0§
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØd
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   e
#lstm_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0,lstm_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ç
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_8_while_body_1110615*%
condR
lstm_8_while_cond_1110614*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementso
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_17/MatMulMatMullstm_8/strided_slice_3:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
Å	
±
.__inference_sequential_8_layer_call_fn_1110249
dense_16_input
unknown:	2
	unknown_0:2
	unknown_1:	2Ø
	unknown_2:
Ø
	unknown_3:	Ø
	unknown_4:	
	unknown_5:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110213o
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
_user_specified_namedense_16_input
­	
©
.__inference_sequential_8_layer_call_fn_1110333

inputs
unknown:	2
	unknown_0:2
	unknown_1:	2Ø
	unknown_2:
Ø
	unknown_3:	Ø
	unknown_4:	
	unknown_5:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_1109967o
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
¼|
¢
#__inference__traced_restore_1111693
file_prefix2
 assignvariableop_dense_16_kernel:	2.
 assignvariableop_1_dense_16_bias:25
"assignvariableop_2_dense_17_kernel:	.
 assignvariableop_3_dense_17_bias:?
,assignvariableop_4_lstm_8_lstm_cell_8_kernel:	2ØJ
6assignvariableop_5_lstm_8_lstm_cell_8_recurrent_kernel:
Ø9
*assignvariableop_6_lstm_8_lstm_cell_8_bias:	Ø&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: ;
)assignvariableop_9_adam_m_dense_16_kernel:	2<
*assignvariableop_10_adam_v_dense_16_kernel:	26
(assignvariableop_11_adam_m_dense_16_bias:26
(assignvariableop_12_adam_v_dense_16_bias:2G
4assignvariableop_13_adam_m_lstm_8_lstm_cell_8_kernel:	2ØG
4assignvariableop_14_adam_v_lstm_8_lstm_cell_8_kernel:	2ØR
>assignvariableop_15_adam_m_lstm_8_lstm_cell_8_recurrent_kernel:
ØR
>assignvariableop_16_adam_v_lstm_8_lstm_cell_8_recurrent_kernel:
ØA
2assignvariableop_17_adam_m_lstm_8_lstm_cell_8_bias:	ØA
2assignvariableop_18_adam_v_lstm_8_lstm_cell_8_bias:	Ø=
*assignvariableop_19_adam_m_dense_17_kernel:	=
*assignvariableop_20_adam_v_dense_17_kernel:	6
(assignvariableop_21_adam_m_dense_17_bias:6
(assignvariableop_22_adam_v_dense_17_bias:%
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
AssignVariableOpAssignVariableOp assignvariableop_dense_16_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_16_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_17_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_17_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_8_lstm_cell_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_5AssignVariableOp6assignvariableop_5_lstm_8_lstm_cell_8_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_6AssignVariableOp*assignvariableop_6_lstm_8_lstm_cell_8_biasIdentity_6:output:0"/device:CPU:0*&
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
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_m_dense_16_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_v_dense_16_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_11AssignVariableOp(assignvariableop_11_adam_m_dense_16_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_v_dense_16_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_m_lstm_8_lstm_cell_8_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_v_lstm_8_lstm_cell_8_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_15AssignVariableOp>assignvariableop_15_adam_m_lstm_8_lstm_cell_8_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:×
AssignVariableOp_16AssignVariableOp>assignvariableop_16_adam_v_lstm_8_lstm_cell_8_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_m_lstm_8_lstm_cell_8_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_v_lstm_8_lstm_cell_8_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_17_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_17_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_17_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_17_biasIdentity_22:output:0"/device:CPU:0*&
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
ë

H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109468

inputs

states
states_11
matmul_readvariableop_resource:	2Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¤A
ª

lstm_8_while_body_1110615*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØO
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	2ØM
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:
ØG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   É
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0©
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0Í
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ®
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0´
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ°
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ§
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¹
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØj
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7lstm_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1@lstm_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: n
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¾
È
while_cond_1110066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1110066___redundant_placeholder05
1while_while_cond_1110066___redundant_placeholder15
1while_while_cond_1110066___redundant_placeholder25
1while_while_cond_1110066___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾
È
while_cond_1111138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1111138___redundant_placeholder05
1while_while_cond_1111138___redundant_placeholder15
1while_while_cond_1111138___redundant_placeholder25
1while_while_cond_1111138___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¾
È
while_cond_1109675
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1109675___redundant_placeholder05
1while_while_cond_1109675___redundant_placeholder15
1while_while_cond_1109675___redundant_placeholder25
1while_while_cond_1109675___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ì
ü
E__inference_dense_16_layer_call_and_return_conditional_losses_1110745

inputs3
!tensordot_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	2*
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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2z
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
¤A
ª

lstm_8_while_body_1110438*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØO
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	2ØM
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:
ØG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   É
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0©
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0Í
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ®
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0´
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ°
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ§
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¹
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØj
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7lstm_8/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1@lstm_8/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: n
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¾
È
while_cond_1110848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1110848___redundant_placeholder05
1while_while_cond_1110848___redundant_placeholder15
1while_while_cond_1110848___redundant_placeholder25
1while_while_cond_1110848___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ó

H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111486

inputs
states_0
states_11
matmul_readvariableop_resource:	2Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1110152

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1110067*
condR
while_cond_1110066*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
Ì
ü
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792

inputs3
!tensordot_readvariableop_resource:	2-
biasadd_readvariableop_resource:2
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	2*
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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2z
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
ú
¶
(__inference_lstm_8_layer_call_fn_1110778

inputs
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1109942p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
¾
È
while_cond_1110993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1110993___redundant_placeholder05
1while_while_cond_1110993___redundant_placeholder15
1while_while_cond_1110993___redundant_placeholder25
1while_while_cond_1110993___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ú
¶
(__inference_lstm_8_layer_call_fn_1110789

inputs
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110152p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
Ô

*__inference_dense_16_layer_call_fn_1110715

inputs
unknown:	2
	unknown_0:2
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2`
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
Å	
±
.__inference_sequential_8_layer_call_fn_1109984
dense_16_input
unknown:	2
	unknown_0:2
	unknown_1:	2Ø
	unknown_2:
Ø
	unknown_3:	Ø
	unknown_4:	
	unknown_5:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_1109967o
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
_user_specified_namedense_16_input
¾
È
while_cond_1109856
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1109856___redundant_placeholder05
1while_while_cond_1109856___redundant_placeholder15
1while_while_cond_1109856___redundant_placeholder25
1while_while_cond_1109856___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ì	
÷
E__inference_dense_17_layer_call_and_return_conditional_losses_1111388

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îx
À
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110529

inputs<
*dense_16_tensordot_readvariableop_resource:	26
(dense_16_biasadd_readvariableop_resource:2D
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	2ØG
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:
ØA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	Ø:
'dense_17_matmul_readvariableop_resource:	6
(dense_17_biasadd_readvariableop_resource:
identity¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOp¢)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_8/while
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:	2*
dtype0a
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2d
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2b
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2U
lstm_8/ShapeShapedense_16/BiasAdd:output:0*
T0*
_output_shapes
:d
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_8/transpose	Transposedense_16/BiasAdd:output:0lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2R
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:f
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   õ
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0©
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0£
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0§
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØd
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ó
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split{
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   e
#lstm_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0,lstm_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ç
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_8_while_body_1110438*%
condR
lstm_8_while_cond_1110437*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementso
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskl
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¬
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_17/MatMulMatMullstm_8/strided_slice_3:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿâ
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1109942

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1109857*
condR
while_cond_1109856*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
Æ
¾
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110213

inputs"
dense_16_1110195:	2
dense_16_1110197:2!
lstm_8_1110200:	2Ø"
lstm_8_1110202:
Ø
lstm_8_1110204:	Ø#
dense_17_1110207:	
dense_17_1110209:
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall÷
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_1110195dense_16_1110197*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792¡
lstm_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0lstm_8_1110200lstm_8_1110202lstm_8_1110204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110152
 dense_17/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_17_1110207dense_17_1110209*
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
GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
û8
Ê
while_body_1111284
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ì	
÷
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
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
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
å
while_body_1109676
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_8_1109700_0:	2Ø/
while_lstm_cell_8_1109702_0:
Ø*
while_lstm_cell_8_1109704_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_8_1109700:	2Ø-
while_lstm_cell_8_1109702:
Ø(
while_lstm_cell_8_1109704:	Ø¢)while/lstm_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0µ
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_1109700_0while_lstm_cell_8_1109702_0while_lstm_cell_8_1109704_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109616r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
: 
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_8_1109700while_lstm_cell_8_1109700_0"8
while_lstm_cell_8_1109702while_lstm_cell_8_1109702_0"8
while_lstm_cell_8_1109704while_lstm_cell_8_1109704_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
9

C__inference_lstm_8_layer_call_and_return_conditional_losses_1109746

inputs&
lstm_cell_8_1109662:	2Ø'
lstm_cell_8_1109664:
Ø"
lstm_cell_8_1109666:	Ø
identity¢#lstm_cell_8/StatefulPartitionedCall¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask÷
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_1109662lstm_cell_8_1109664lstm_cell_8_1109666*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109616n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : ½
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_1109662lstm_cell_8_1109664lstm_cell_8_1109666*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1109676*
condR
while_cond_1109675*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ë

H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109616

inputs

states
states_11
matmul_readvariableop_resource:	2Ø4
 matmul_1_readvariableop_resource:
Ø.
biasadd_readvariableop_resource:	Ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
­	
©
.__inference_sequential_8_layer_call_fn_1110352

inputs
unknown:	2
	unknown_0:2
	unknown_1:	2Ø
	unknown_2:
Ø
	unknown_3:	Ø
	unknown_4:	
	unknown_5:
identity¢StatefulPartitionedCall
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
GPU 2J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110213o
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
	
¨
%__inference_signature_wrapper_1110314
dense_16_input
unknown:	2
	unknown_0:2
	unknown_1:	2Ø
	unknown_2:
Ø
	unknown_3:	Ø
	unknown_4:	
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1109401o
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
_user_specified_namedense_16_input
¾
È
while_cond_1109482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1109482___redundant_placeholder05
1while_while_cond_1109482___redundant_placeholder15
1while_while_cond_1109482___redundant_placeholder25
1while_while_cond_1109482___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
û8
Ê
while_body_1111139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


Ô
lstm_8_while_cond_1110614*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1C
?lstm_8_while_lstm_8_while_cond_1110614___redundant_placeholder0C
?lstm_8_while_lstm_8_while_cond_1110614___redundant_placeholder1C
?lstm_8_while_lstm_8_while_cond_1110614___redundant_placeholder2C
?lstm_8_while_lstm_8_while_cond_1110614___redundant_placeholder3
lstm_8_while_identity
~
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: Y
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ö>
é
 __inference__traced_save_1111596
file_prefix.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop8
4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_dense_16_kernel_read_readvariableop5
1savev2_adam_v_dense_16_kernel_read_readvariableop3
/savev2_adam_m_dense_16_bias_read_readvariableop3
/savev2_adam_v_dense_16_bias_read_readvariableop?
;savev2_adam_m_lstm_8_lstm_cell_8_kernel_read_readvariableop?
;savev2_adam_v_lstm_8_lstm_cell_8_kernel_read_readvariableopI
Esavev2_adam_m_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableopI
Esavev2_adam_v_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop=
9savev2_adam_m_lstm_8_lstm_cell_8_bias_read_readvariableop=
9savev2_adam_v_lstm_8_lstm_cell_8_bias_read_readvariableop5
1savev2_adam_m_dense_17_kernel_read_readvariableop5
1savev2_adam_v_dense_17_kernel_read_readvariableop3
/savev2_adam_m_dense_17_bias_read_readvariableop3
/savev2_adam_v_dense_17_bias_read_readvariableop&
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_dense_16_kernel_read_readvariableop1savev2_adam_v_dense_16_kernel_read_readvariableop/savev2_adam_m_dense_16_bias_read_readvariableop/savev2_adam_v_dense_16_bias_read_readvariableop;savev2_adam_m_lstm_8_lstm_cell_8_kernel_read_readvariableop;savev2_adam_v_lstm_8_lstm_cell_8_kernel_read_readvariableopEsavev2_adam_m_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableopEsavev2_adam_v_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop9savev2_adam_m_lstm_8_lstm_cell_8_bias_read_readvariableop9savev2_adam_v_lstm_8_lstm_cell_8_bias_read_readvariableop1savev2_adam_m_dense_17_kernel_read_readvariableop1savev2_adam_v_dense_17_kernel_read_readvariableop/savev2_adam_m_dense_17_bias_read_readvariableop/savev2_adam_v_dense_17_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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

identity_1Identity_1:output:0*æ
_input_shapesÔ
Ñ: :	2:2:	::	2Ø:
Ø:Ø: : :	2:	2:2:2:	2Ø:	2Ø:
Ø:
Ø:Ø:Ø:	:	::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	2: 

_output_shapes
:2:%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	2Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

:	2:$ 

_output_shapes

:	2: 

_output_shapes
:2: 

_output_shapes
:2:%!

_output_shapes
:	2Ø:%!

_output_shapes
:	2Ø:&"
 
_output_shapes
:
Ø:&"
 
_output_shapes
:
Ø:!

_output_shapes	
:Ø:!

_output_shapes	
:Ø:%!

_output_shapes
:	:%!

_output_shapes
:	: 
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


Ô
lstm_8_while_cond_1110437*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1C
?lstm_8_while_lstm_8_while_cond_1110437___redundant_placeholder0C
?lstm_8_while_lstm_8_while_cond_1110437___redundant_placeholder1C
?lstm_8_while_lstm_8_while_cond_1110437___redundant_placeholder2C
?lstm_8_while_lstm_8_while_cond_1110437___redundant_placeholder3
lstm_8_while_identity
~
lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: Y
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
û8
Ê
while_body_1110994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
·
ä
"__inference__wrapped_model_1109401
dense_16_inputI
7sequential_8_dense_16_tensordot_readvariableop_resource:	2C
5sequential_8_dense_16_biasadd_readvariableop_resource:2Q
>sequential_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource:	2ØT
@sequential_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:
ØN
?sequential_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	ØG
4sequential_8_dense_17_matmul_readvariableop_resource:	C
5sequential_8_dense_17_biasadd_readvariableop_resource:
identity¢,sequential_8/dense_16/BiasAdd/ReadVariableOp¢.sequential_8/dense_16/Tensordot/ReadVariableOp¢,sequential_8/dense_17/BiasAdd/ReadVariableOp¢+sequential_8/dense_17/MatMul/ReadVariableOp¢6sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢5sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢7sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢sequential_8/lstm_8/while¦
.sequential_8/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_16_tensordot_readvariableop_resource*
_output_shapes

:	2*
dtype0n
$sequential_8/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_8/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
%sequential_8/dense_16/Tensordot/ShapeShapedense_16_input*
T0*
_output_shapes
:o
-sequential_8/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(sequential_8/dense_16/Tensordot/GatherV2GatherV2.sequential_8/dense_16/Tensordot/Shape:output:0-sequential_8/dense_16/Tensordot/free:output:06sequential_8/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_8/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
*sequential_8/dense_16/Tensordot/GatherV2_1GatherV2.sequential_8/dense_16/Tensordot/Shape:output:0-sequential_8/dense_16/Tensordot/axes:output:08sequential_8/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_8/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: °
$sequential_8/dense_16/Tensordot/ProdProd1sequential_8/dense_16/Tensordot/GatherV2:output:0.sequential_8/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_8/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¶
&sequential_8/dense_16/Tensordot/Prod_1Prod3sequential_8/dense_16/Tensordot/GatherV2_1:output:00sequential_8/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_8/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ô
&sequential_8/dense_16/Tensordot/concatConcatV2-sequential_8/dense_16/Tensordot/free:output:0-sequential_8/dense_16/Tensordot/axes:output:04sequential_8/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:»
%sequential_8/dense_16/Tensordot/stackPack-sequential_8/dense_16/Tensordot/Prod:output:0/sequential_8/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:­
)sequential_8/dense_16/Tensordot/transpose	Transposedense_16_input/sequential_8/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	Ì
'sequential_8/dense_16/Tensordot/ReshapeReshape-sequential_8/dense_16/Tensordot/transpose:y:0.sequential_8/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
&sequential_8/dense_16/Tensordot/MatMulMatMul0sequential_8/dense_16/Tensordot/Reshape:output:06sequential_8/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
'sequential_8/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2o
-sequential_8/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
(sequential_8/dense_16/Tensordot/concat_1ConcatV21sequential_8/dense_16/Tensordot/GatherV2:output:00sequential_8/dense_16/Tensordot/Const_2:output:06sequential_8/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Å
sequential_8/dense_16/TensordotReshape0sequential_8/dense_16/Tensordot/MatMul:product:01sequential_8/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
,sequential_8/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_16_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0¾
sequential_8/dense_16/BiasAddBiasAdd(sequential_8/dense_16/Tensordot:output:04sequential_8/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2o
sequential_8/lstm_8/ShapeShape&sequential_8/dense_16/BiasAdd:output:0*
T0*
_output_shapes
:q
'sequential_8/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_8/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_8/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_8/lstm_8/strided_sliceStridedSlice"sequential_8/lstm_8/Shape:output:00sequential_8/lstm_8/strided_slice/stack:output:02sequential_8/lstm_8/strided_slice/stack_1:output:02sequential_8/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_8/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¯
 sequential_8/lstm_8/zeros/packedPack*sequential_8/lstm_8/strided_slice:output:0+sequential_8/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_8/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ©
sequential_8/lstm_8/zerosFill)sequential_8/lstm_8/zeros/packed:output:0(sequential_8/lstm_8/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$sequential_8/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :³
"sequential_8/lstm_8/zeros_1/packedPack*sequential_8/lstm_8/strided_slice:output:0-sequential_8/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_8/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¯
sequential_8/lstm_8/zeros_1Fill+sequential_8/lstm_8/zeros_1/packed:output:0*sequential_8/lstm_8/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"sequential_8/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
sequential_8/lstm_8/transpose	Transpose&sequential_8/dense_16/BiasAdd:output:0+sequential_8/lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2l
sequential_8/lstm_8/Shape_1Shape!sequential_8/lstm_8/transpose:y:0*
T0*
_output_shapes
:s
)sequential_8/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_8/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_8/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_8/lstm_8/strided_slice_1StridedSlice$sequential_8/lstm_8/Shape_1:output:02sequential_8/lstm_8/strided_slice_1/stack:output:04sequential_8/lstm_8/strided_slice_1/stack_1:output:04sequential_8/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_8/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_8/lstm_8/TensorArrayV2TensorListReserve8sequential_8/lstm_8/TensorArrayV2/element_shape:output:0,sequential_8/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_8/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
;sequential_8/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_8/lstm_8/transpose:y:0Rsequential_8/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_8/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_8/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_8/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_8/lstm_8/strided_slice_2StridedSlice!sequential_8/lstm_8/transpose:y:02sequential_8/lstm_8/strided_slice_2/stack:output:04sequential_8/lstm_8/strided_slice_2/stack_1:output:04sequential_8/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskµ
5sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp>sequential_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0Ð
&sequential_8/lstm_8/lstm_cell_8/MatMulMatMul,sequential_8/lstm_8/strided_slice_2:output:0=sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØº
7sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp@sequential_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0Ê
(sequential_8/lstm_8/lstm_cell_8/MatMul_1MatMul"sequential_8/lstm_8/zeros:output:0?sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØÅ
#sequential_8/lstm_8/lstm_cell_8/addAddV20sequential_8/lstm_8/lstm_cell_8/MatMul:product:02sequential_8/lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ³
6sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0Î
'sequential_8/lstm_8/lstm_cell_8/BiasAddBiasAdd'sequential_8/lstm_8/lstm_cell_8/add:z:0>sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØq
/sequential_8/lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_8/lstm_8/lstm_cell_8/splitSplit8sequential_8/lstm_8/lstm_cell_8/split/split_dim:output:00sequential_8/lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
'sequential_8/lstm_8/lstm_cell_8/SigmoidSigmoid.sequential_8/lstm_8/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_8/lstm_8/lstm_cell_8/Sigmoid_1Sigmoid.sequential_8/lstm_8/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
#sequential_8/lstm_8/lstm_cell_8/mulMul-sequential_8/lstm_8/lstm_cell_8/Sigmoid_1:y:0$sequential_8/lstm_8/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_8/lstm_8/lstm_cell_8/ReluRelu.sequential_8/lstm_8/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
%sequential_8/lstm_8/lstm_cell_8/mul_1Mul+sequential_8/lstm_8/lstm_cell_8/Sigmoid:y:02sequential_8/lstm_8/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
%sequential_8/lstm_8/lstm_cell_8/add_1AddV2'sequential_8/lstm_8/lstm_cell_8/mul:z:0)sequential_8/lstm_8/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_8/lstm_8/lstm_cell_8/Sigmoid_2Sigmoid.sequential_8/lstm_8/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_8/lstm_8/lstm_cell_8/Relu_1Relu)sequential_8/lstm_8/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
%sequential_8/lstm_8/lstm_cell_8/mul_2Mul-sequential_8/lstm_8/lstm_cell_8/Sigmoid_2:y:04sequential_8/lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
1sequential_8/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   r
0sequential_8/lstm_8/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
#sequential_8/lstm_8/TensorArrayV2_1TensorListReserve:sequential_8/lstm_8/TensorArrayV2_1/element_shape:output:09sequential_8/lstm_8/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_8/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_8/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_8/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_8/lstm_8/whileWhile/sequential_8/lstm_8/while/loop_counter:output:05sequential_8/lstm_8/while/maximum_iterations:output:0!sequential_8/lstm_8/time:output:0,sequential_8/lstm_8/TensorArrayV2_1:handle:0"sequential_8/lstm_8/zeros:output:0$sequential_8/lstm_8/zeros_1:output:0,sequential_8/lstm_8/strided_slice_1:output:0Ksequential_8/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_8_lstm_8_lstm_cell_8_matmul_readvariableop_resource@sequential_8_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource?sequential_8_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_8_lstm_8_while_body_1109310*2
cond*R(
&sequential_8_lstm_8_while_cond_1109309*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Dsequential_8/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
6sequential_8/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_8/lstm_8/while:output:3Msequential_8/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elements|
)sequential_8/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_8/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_8/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
#sequential_8/lstm_8/strided_slice_3StridedSlice?sequential_8/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:02sequential_8/lstm_8/strided_slice_3/stack:output:04sequential_8/lstm_8/strided_slice_3/stack_1:output:04sequential_8/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_masky
$sequential_8/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ó
sequential_8/lstm_8/transpose_1	Transpose?sequential_8/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_8/lstm_8/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
sequential_8/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¡
+sequential_8/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_17_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0»
sequential_8/dense_17/MatMulMatMul,sequential_8/lstm_8/strided_slice_3:output:03sequential_8/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_8/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_8/dense_17/BiasAddBiasAdd&sequential_8/dense_17/MatMul:product:04sequential_8/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_8/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
NoOpNoOp-^sequential_8/dense_16/BiasAdd/ReadVariableOp/^sequential_8/dense_16/Tensordot/ReadVariableOp-^sequential_8/dense_17/BiasAdd/ReadVariableOp,^sequential_8/dense_17/MatMul/ReadVariableOp7^sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6^sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp8^sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^sequential_8/lstm_8/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2\
,sequential_8/dense_16/BiasAdd/ReadVariableOp,sequential_8/dense_16/BiasAdd/ReadVariableOp2`
.sequential_8/dense_16/Tensordot/ReadVariableOp.sequential_8/dense_16/Tensordot/ReadVariableOp2\
,sequential_8/dense_17/BiasAdd/ReadVariableOp,sequential_8/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_17/MatMul/ReadVariableOp+sequential_8/dense_17/MatMul/ReadVariableOp2p
6sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6sequential_8/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2n
5sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp5sequential_8/lstm_8/lstm_cell_8/MatMul/ReadVariableOp2r
7sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp7sequential_8/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp26
sequential_8/lstm_8/whilesequential_8/lstm_8/while:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_16_input

¸
(__inference_lstm_8_layer_call_fn_1110767
inputs_0
unknown:	2Ø
	unknown_0:
Ø
	unknown_1:	Ø
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1109746p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs_0
K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1111369

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while;
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:xÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1111284*
condR
while_cond_1111283*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿx2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2
 
_user_specified_nameinputs
®K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1110934
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1110849*
condR
while_cond_1110848*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs_0
Þ
Æ
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110270
dense_16_input"
dense_16_1110252:	2
dense_16_1110254:2!
lstm_8_1110257:	2Ø"
lstm_8_1110259:
Ø
lstm_8_1110261:	Ø#
dense_17_1110264:	
dense_17_1110266:
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCallÿ
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_1110252dense_16_1110254*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792¡
lstm_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0lstm_8_1110257lstm_8_1110259lstm_8_1110261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1109942
 dense_17/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_17_1110264dense_17_1110266*
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
GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_16_input
Ç

*__inference_dense_17_layer_call_fn_1111378

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
È
while_cond_1111283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1111283___redundant_placeholder05
1while_while_cond_1111283___redundant_placeholder15
1while_while_cond_1111283___redundant_placeholder25
1while_while_cond_1111283___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
$
å
while_body_1109483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_8_1109507_0:	2Ø/
while_lstm_cell_8_1109509_0:
Ø*
while_lstm_cell_8_1109511_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_8_1109507:	2Ø-
while_lstm_cell_8_1109509:
Ø(
while_lstm_cell_8_1109511:	Ø¢)while/lstm_cell_8/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0µ
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_1109507_0while_lstm_cell_8_1109509_0while_lstm_cell_8_1109511_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1109468r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
: 
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_8_1109507while_lstm_cell_8_1109507_0"8
while_lstm_cell_8_1109509while_lstm_cell_8_1109509_0"8
while_lstm_cell_8_1109511while_lstm_cell_8_1109511_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
®K

C__inference_lstm_8_layer_call_and_return_conditional_losses_1111079
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	2Ø@
,lstm_cell_8_matmul_1_readvariableop_resource:
Ø:
+lstm_cell_8_biasadd_readvariableop_resource:	Ø
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢while=
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
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
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
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
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
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
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
valueB"ÿÿÿÿ2   à
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	2Ø*
dtype0
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource* 
_output_shapes
:
Ø*
dtype0
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:Ø*
dtype0
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ]
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Þ
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitm
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1110994*
condR
while_cond_1110993*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ×
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs_0
Æ
¾
I__inference_sequential_8_layer_call_and_return_conditional_losses_1109967

inputs"
dense_16_1109793:	2
dense_16_1109795:2!
lstm_8_1109943:	2Ø"
lstm_8_1109945:
Ø
lstm_8_1109947:	Ø#
dense_17_1109961:	
dense_17_1109963:
identity¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall÷
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_1109793dense_16_1109795*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_16_layer_call_and_return_conditional_losses_1109792¡
lstm_8/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0lstm_8_1109943lstm_8_1109945lstm_8_1109947*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_8_layer_call_and_return_conditional_losses_1109942
 dense_17/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0dense_17_1109961dense_17_1109963*
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
GPU 2J 8 *N
fIRG
E__inference_dense_17_layer_call_and_return_conditional_losses_1109960x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
û8
Ê
while_body_1109857
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	2ØH
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:
ØB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	Ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	2ØF
2while_lstm_cell_8_matmul_1_readvariableop_resource:
Ø@
1while_lstm_cell_8_biasadd_readvariableop_resource:	Ø¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	2Ø*
dtype0¸
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ 
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ø*
dtype0
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:Ø*
dtype0¤
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØc
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ð
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splity
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_8/mul_2:z:0*
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
: y
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: "
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
dense_16_input;
 serving_default_dense_16_input:0ÿÿÿÿÿÿÿÿÿx	<
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Äµ
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
í
.trace_0
/trace_1
0trace_2
1trace_32
.__inference_sequential_8_layer_call_fn_1109984
.__inference_sequential_8_layer_call_fn_1110333
.__inference_sequential_8_layer_call_fn_1110352
.__inference_sequential_8_layer_call_fn_1110249¿
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
Ù
2trace_0
3trace_1
4trace_2
5trace_32î
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110529
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110706
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110270
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110291¿
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
ÔBÑ
"__inference__wrapped_model_1109401dense_16_input"
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
î
Ctrace_02Ñ
*__inference_dense_16_layer_call_fn_1110715¢
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

Dtrace_02ì
E__inference_dense_16_layer_call_and_return_conditional_losses_1110745¢
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
!:	22dense_16/kernel
:22dense_16/bias
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
ê
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32ÿ
(__inference_lstm_8_layer_call_fn_1110756
(__inference_lstm_8_layer_call_fn_1110767
(__inference_lstm_8_layer_call_fn_1110778
(__inference_lstm_8_layer_call_fn_1110789Ô
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
Ö
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ë
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110934
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111079
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111224
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111369Ô
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
î
`trace_02Ñ
*__inference_dense_17_layer_call_fn_1111378¢
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

atrace_02ì
E__inference_dense_17_layer_call_and_return_conditional_losses_1111388¢
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
": 	2dense_17/kernel
:2dense_17/bias
,:*	2Ø2lstm_8/lstm_cell_8/kernel
7:5
Ø2#lstm_8/lstm_cell_8/recurrent_kernel
&:$Ø2lstm_8/lstm_cell_8/bias
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
B
.__inference_sequential_8_layer_call_fn_1109984dense_16_input"¿
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
ÿBü
.__inference_sequential_8_layer_call_fn_1110333inputs"¿
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
ÿBü
.__inference_sequential_8_layer_call_fn_1110352inputs"¿
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
B
.__inference_sequential_8_layer_call_fn_1110249dense_16_input"¿
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
B
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110529inputs"¿
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
B
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110706inputs"¿
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
¢B
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110270dense_16_input"¿
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
¢B
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110291dense_16_input"¿
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
ÓBÐ
%__inference_signature_wrapper_1110314dense_16_input"
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
ÞBÛ
*__inference_dense_16_layer_call_fn_1110715inputs"¢
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
ùBö
E__inference_dense_16_layer_call_and_return_conditional_losses_1110745inputs"¢
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
B
(__inference_lstm_8_layer_call_fn_1110756inputs_0"Ô
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
B
(__inference_lstm_8_layer_call_fn_1110767inputs_0"Ô
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
B
(__inference_lstm_8_layer_call_fn_1110778inputs"Ô
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
B
(__inference_lstm_8_layer_call_fn_1110789inputs"Ô
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
«B¨
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110934inputs_0"Ô
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
«B¨
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111079inputs_0"Ô
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
©B¦
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111224inputs"Ô
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
©B¦
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111369inputs"Ô
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
Õ
xtrace_0
ytrace_12
-__inference_lstm_cell_8_layer_call_fn_1111405
-__inference_lstm_cell_8_layer_call_fn_1111422½
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

ztrace_0
{trace_12Ô
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111454
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111486½
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
ÞBÛ
*__inference_dense_17_layer_call_fn_1111378inputs"¢
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
ùBö
E__inference_dense_17_layer_call_and_return_conditional_losses_1111388inputs"¢
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
&:$	22Adam/m/dense_16/kernel
&:$	22Adam/v/dense_16/kernel
 :22Adam/m/dense_16/bias
 :22Adam/v/dense_16/bias
1:/	2Ø2 Adam/m/lstm_8/lstm_cell_8/kernel
1:/	2Ø2 Adam/v/lstm_8/lstm_cell_8/kernel
<::
Ø2*Adam/m/lstm_8/lstm_cell_8/recurrent_kernel
<::
Ø2*Adam/v/lstm_8/lstm_cell_8/recurrent_kernel
+:)Ø2Adam/m/lstm_8/lstm_cell_8/bias
+:)Ø2Adam/v/lstm_8/lstm_cell_8/bias
':%	2Adam/m/dense_17/kernel
':%	2Adam/v/dense_17/kernel
 :2Adam/m/dense_17/bias
 :2Adam/v/dense_17/bias
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
B
-__inference_lstm_cell_8_layer_call_fn_1111405inputsstates_0states_1"½
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
B
-__inference_lstm_cell_8_layer_call_fn_1111422inputsstates_0states_1"½
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
«B¨
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111454inputsstates_0states_1"½
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
«B¨
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111486inputsstates_0states_1"½
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
trackable_dict_wrapper¡
"__inference__wrapped_model_1109401{&'($%;¢8
1¢.
,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ´
E__inference_dense_16_layer_call_and_return_conditional_losses_1110745k3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "0¢-
&#
tensor_0ÿÿÿÿÿÿÿÿÿx2
 
*__inference_dense_16_layer_call_fn_1110715`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "%"
unknownÿÿÿÿÿÿÿÿÿx2­
E__inference_dense_17_layer_call_and_return_conditional_losses_1111388d$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_17_layer_call_fn_1111378Y$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿÍ
C__inference_lstm_8_layer_call_and_return_conditional_losses_1110934&'(O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 Í
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111079&'(O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¼
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111224u&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿx2

 
p 

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¼
C__inference_lstm_8_layer_call_and_return_conditional_losses_1111369u&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿx2

 
p

 
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¦
(__inference_lstm_8_layer_call_fn_1110756z&'(O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ¦
(__inference_lstm_8_layer_call_fn_1110767z&'(O¢L
E¢B
41
/,
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ
(__inference_lstm_8_layer_call_fn_1110778j&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿx2

 
p 

 
ª ""
unknownÿÿÿÿÿÿÿÿÿ
(__inference_lstm_8_layer_call_fn_1110789j&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿx2

 
p

 
ª ""
unknownÿÿÿÿÿÿÿÿÿç
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111454&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ2
M¢J
# 
states_0ÿÿÿÿÿÿÿÿÿ
# 
states_1ÿÿÿÿÿÿÿÿÿ
p 
ª "¢
¢~
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
UR
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
'$
tensor_0_1_1ÿÿÿÿÿÿÿÿÿ
 ç
H__inference_lstm_cell_8_layer_call_and_return_conditional_losses_1111486&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ2
M¢J
# 
states_0ÿÿÿÿÿÿÿÿÿ
# 
states_1ÿÿÿÿÿÿÿÿÿ
p
ª "¢
¢~
%"

tensor_0_0ÿÿÿÿÿÿÿÿÿ
UR
'$
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
'$
tensor_0_1_1ÿÿÿÿÿÿÿÿÿ
 ¹
-__inference_lstm_cell_8_layer_call_fn_1111405&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ2
M¢J
# 
states_0ÿÿÿÿÿÿÿÿÿ
# 
states_1ÿÿÿÿÿÿÿÿÿ
p 
ª "{¢x
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
QN
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿ
%"

tensor_1_1ÿÿÿÿÿÿÿÿÿ¹
-__inference_lstm_cell_8_layer_call_fn_1111422&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ2
M¢J
# 
states_0ÿÿÿÿÿÿÿÿÿ
# 
states_1ÿÿÿÿÿÿÿÿÿ
p
ª "{¢x
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
QN
%"

tensor_1_0ÿÿÿÿÿÿÿÿÿ
%"

tensor_1_1ÿÿÿÿÿÿÿÿÿÉ
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110270|&'($%C¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 É
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110291|&'($%C¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Á
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110529t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Á
I__inference_sequential_8_layer_call_and_return_conditional_losses_1110706t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 £
.__inference_sequential_8_layer_call_fn_1109984q&'($%C¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ£
.__inference_sequential_8_layer_call_fn_1110249q&'($%C¢@
9¢6
,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
.__inference_sequential_8_layer_call_fn_1110333i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
.__inference_sequential_8_layer_call_fn_1110352i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ·
%__inference_signature_wrapper_1110314&'($%M¢J
¢ 
Cª@
>
dense_16_input,)
dense_16_inputÿÿÿÿÿÿÿÿÿx	"3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ