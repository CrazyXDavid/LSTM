Å²
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
"serve*2.11.02v2.11.0-rc2-15-g6290819256d8ÉÜ
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
Adam/v/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_41/bias
y
(Adam/v/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/bias*
_output_shapes
:*
dtype0

Adam/m/dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_41/bias
y
(Adam/m/dense_41/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/bias*
_output_shapes
:*
dtype0

Adam/v/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_41/kernel

*Adam/v/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_41/kernel*
_output_shapes

:*
dtype0

Adam/m/dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_41/kernel

*Adam/m/dense_41/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_41/kernel*
_output_shapes

:*
dtype0

 Adam/v/lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adam/v/lstm_20/lstm_cell_20/bias

4Adam/v/lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_20/lstm_cell_20/bias*
_output_shapes
:d*
dtype0

 Adam/m/lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*1
shared_name" Adam/m/lstm_20/lstm_cell_20/bias

4Adam/m/lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_20/lstm_cell_20/bias*
_output_shapes
:d*
dtype0
´
,Adam/v/lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*=
shared_name.,Adam/v/lstm_20/lstm_cell_20/recurrent_kernel
­
@Adam/v/lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_20/lstm_cell_20/recurrent_kernel*
_output_shapes

:d*
dtype0
´
,Adam/m/lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*=
shared_name.,Adam/m/lstm_20/lstm_cell_20/recurrent_kernel
­
@Adam/m/lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_20/lstm_cell_20/recurrent_kernel*
_output_shapes

:d*
dtype0
¡
"Adam/v/lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*3
shared_name$"Adam/v/lstm_20/lstm_cell_20/kernel

6Adam/v/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_20/lstm_cell_20/kernel*
_output_shapes
:	d*
dtype0
¡
"Adam/m/lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*3
shared_name$"Adam/m/lstm_20/lstm_cell_20/kernel

6Adam/m/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_20/lstm_cell_20/kernel*
_output_shapes
:	d*
dtype0

Adam/v/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_40/bias
z
(Adam/v/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/bias*
_output_shapes	
:*
dtype0

Adam/m/dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_40/bias
z
(Adam/m/dense_40/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/bias*
_output_shapes	
:*
dtype0

Adam/v/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*'
shared_nameAdam/v/dense_40/kernel

*Adam/v/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_40/kernel*
_output_shapes
:		*
dtype0

Adam/m/dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*'
shared_nameAdam/m/dense_40/kernel

*Adam/m/dense_40/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_40/kernel*
_output_shapes
:		*
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

lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_namelstm_20/lstm_cell_20/bias

-lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/bias*
_output_shapes
:d*
dtype0
¦
%lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*6
shared_name'%lstm_20/lstm_cell_20/recurrent_kernel

9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_20/lstm_cell_20/recurrent_kernel*
_output_shapes

:d*
dtype0

lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*,
shared_namelstm_20/lstm_cell_20/kernel

/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/kernel*
_output_shapes
:	d*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes
:*
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

:*
dtype0
s
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_40/bias
l
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes	
:*
dtype0
{
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		* 
shared_namedense_40/kernel
t
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes
:		*
dtype0

serving_default_dense_40_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿx	
ç
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_40_inputdense_40/kerneldense_40/biaslstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biasdense_41/kerneldense_41/bias*
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
%__inference_signature_wrapper_1590694

NoOpNoOp
Þ7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*7
value7B7 B7
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
VARIABLE_VALUEdense_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_20/lstm_cell_20/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_20/lstm_cell_20/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_20/lstm_cell_20/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/m/dense_40/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_40/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_40/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_40/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/lstm_20/lstm_cell_20/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_20/lstm_cell_20/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_20/lstm_cell_20/recurrent_kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/v/lstm_20/lstm_cell_20/recurrent_kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/lstm_20/lstm_cell_20/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/lstm_20/lstm_cell_20/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_41/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_41/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_41/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_41/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOp/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOp9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOp-lstm_20/lstm_cell_20/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp*Adam/m/dense_40/kernel/Read/ReadVariableOp*Adam/v/dense_40/kernel/Read/ReadVariableOp(Adam/m/dense_40/bias/Read/ReadVariableOp(Adam/v/dense_40/bias/Read/ReadVariableOp6Adam/m/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOp6Adam/v/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOp@Adam/m/lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_20/lstm_cell_20/bias/Read/ReadVariableOp4Adam/v/lstm_20/lstm_cell_20/bias/Read/ReadVariableOp*Adam/m/dense_41/kernel/Read/ReadVariableOp*Adam/v/dense_41/kernel/Read/ReadVariableOp(Adam/m/dense_41/bias/Read/ReadVariableOp(Adam/v/dense_41/bias/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst**
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
 __inference__traced_save_1591976
¿
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasdense_41/kerneldense_41/biaslstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/bias	iterationlearning_rateAdam/m/dense_40/kernelAdam/v/dense_40/kernelAdam/m/dense_40/biasAdam/v/dense_40/bias"Adam/m/lstm_20/lstm_cell_20/kernel"Adam/v/lstm_20/lstm_cell_20/kernel,Adam/m/lstm_20/lstm_cell_20/recurrent_kernel,Adam/v/lstm_20/lstm_cell_20/recurrent_kernel Adam/m/lstm_20/lstm_cell_20/bias Adam/v/lstm_20/lstm_cell_20/biasAdam/m/dense_41/kernelAdam/v/dense_41/kernelAdam/m/dense_41/biasAdam/v/dense_41/biastotal_2count_2total_1count_1totalcount*)
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
#__inference__traced_restore_1592073Ñ
ËK

D__inference_lstm_20_layer_call_and_return_conditional_losses_1591314
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while=
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1591229*
condR
while_cond_1591228*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
í
õ
.__inference_lstm_cell_20_layer_call_fn_1591785

inputs
states_0
states_1
unknown:	d
	unknown_0:d
	unknown_1:d
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
¨K

D__inference_lstm_20_layer_call_and_return_conditional_losses_1590322

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1590237*
condR
while_cond_1590236*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
°S

(sequential_20_lstm_20_while_body_1589690H
Dsequential_20_lstm_20_while_sequential_20_lstm_20_while_loop_counterN
Jsequential_20_lstm_20_while_sequential_20_lstm_20_while_maximum_iterations+
'sequential_20_lstm_20_while_placeholder-
)sequential_20_lstm_20_while_placeholder_1-
)sequential_20_lstm_20_while_placeholder_2-
)sequential_20_lstm_20_while_placeholder_3G
Csequential_20_lstm_20_while_sequential_20_lstm_20_strided_slice_1_0
sequential_20_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_20_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_20_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	d]
Ksequential_20_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:dX
Jsequential_20_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:d(
$sequential_20_lstm_20_while_identity*
&sequential_20_lstm_20_while_identity_1*
&sequential_20_lstm_20_while_identity_2*
&sequential_20_lstm_20_while_identity_3*
&sequential_20_lstm_20_while_identity_4*
&sequential_20_lstm_20_while_identity_5E
Asequential_20_lstm_20_while_sequential_20_lstm_20_strided_slice_1
}sequential_20_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_20_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_20_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	d[
Isequential_20_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:dV
Hsequential_20_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:d¢?sequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢>sequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢@sequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp
Msequential_20/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
?sequential_20/lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_20_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_20_tensorarrayunstack_tensorlistfromtensor_0'sequential_20_lstm_20_while_placeholderVsequential_20/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0É
>sequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpIsequential_20_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0û
/sequential_20/lstm_20/while/lstm_cell_20/MatMulMatMulFsequential_20/lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÌ
@sequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpKsequential_20_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0â
1sequential_20/lstm_20/while/lstm_cell_20/MatMul_1MatMul)sequential_20_lstm_20_while_placeholder_2Hsequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdß
,sequential_20/lstm_20/while/lstm_cell_20/addAddV29sequential_20/lstm_20/while/lstm_cell_20/MatMul:product:0;sequential_20/lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
?sequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpJsequential_20_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0è
0sequential_20/lstm_20/while/lstm_cell_20/BiasAddBiasAdd0sequential_20/lstm_20/while/lstm_cell_20/add:z:0Gsequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
8sequential_20/lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :±
.sequential_20/lstm_20/while/lstm_cell_20/splitSplitAsequential_20/lstm_20/while/lstm_cell_20/split/split_dim:output:09sequential_20/lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¦
0sequential_20/lstm_20/while/lstm_cell_20/SigmoidSigmoid7sequential_20/lstm_20/while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_20/lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid7sequential_20/lstm_20/while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
,sequential_20/lstm_20/while/lstm_cell_20/mulMul6sequential_20/lstm_20/while/lstm_cell_20/Sigmoid_1:y:0)sequential_20_lstm_20_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_20/lstm_20/while/lstm_cell_20/ReluRelu7sequential_20/lstm_20/while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
.sequential_20/lstm_20/while/lstm_cell_20/mul_1Mul4sequential_20/lstm_20/while/lstm_cell_20/Sigmoid:y:0;sequential_20/lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
.sequential_20/lstm_20/while/lstm_cell_20/add_1AddV20sequential_20/lstm_20/while/lstm_cell_20/mul:z:02sequential_20/lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
2sequential_20/lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid7sequential_20/lstm_20/while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/sequential_20/lstm_20/while/lstm_cell_20/Relu_1Relu2sequential_20/lstm_20/while/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
.sequential_20/lstm_20/while/lstm_cell_20/mul_2Mul6sequential_20/lstm_20/while/lstm_cell_20/Sigmoid_2:y:0=sequential_20/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_20/lstm_20/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Å
@sequential_20/lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_20_lstm_20_while_placeholder_1Osequential_20/lstm_20/while/TensorArrayV2Write/TensorListSetItem/index:output:02sequential_20/lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒc
!sequential_20/lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_20/lstm_20/while/addAddV2'sequential_20_lstm_20_while_placeholder*sequential_20/lstm_20/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_20/lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!sequential_20/lstm_20/while/add_1AddV2Dsequential_20_lstm_20_while_sequential_20_lstm_20_while_loop_counter,sequential_20/lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_20/lstm_20/while/IdentityIdentity%sequential_20/lstm_20/while/add_1:z:0!^sequential_20/lstm_20/while/NoOp*
T0*
_output_shapes
: Â
&sequential_20/lstm_20/while/Identity_1IdentityJsequential_20_lstm_20_while_sequential_20_lstm_20_while_maximum_iterations!^sequential_20/lstm_20/while/NoOp*
T0*
_output_shapes
: 
&sequential_20/lstm_20/while/Identity_2Identity#sequential_20/lstm_20/while/add:z:0!^sequential_20/lstm_20/while/NoOp*
T0*
_output_shapes
: È
&sequential_20/lstm_20/while/Identity_3IdentityPsequential_20/lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_20/lstm_20/while/NoOp*
T0*
_output_shapes
: »
&sequential_20/lstm_20/while/Identity_4Identity2sequential_20/lstm_20/while/lstm_cell_20/mul_2:z:0!^sequential_20/lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&sequential_20/lstm_20/while/Identity_5Identity2sequential_20/lstm_20/while/lstm_cell_20/add_1:z:0!^sequential_20/lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 sequential_20/lstm_20/while/NoOpNoOp@^sequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?^sequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpA^sequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_20_lstm_20_while_identity-sequential_20/lstm_20/while/Identity:output:0"Y
&sequential_20_lstm_20_while_identity_1/sequential_20/lstm_20/while/Identity_1:output:0"Y
&sequential_20_lstm_20_while_identity_2/sequential_20/lstm_20/while/Identity_2:output:0"Y
&sequential_20_lstm_20_while_identity_3/sequential_20/lstm_20/while/Identity_3:output:0"Y
&sequential_20_lstm_20_while_identity_4/sequential_20/lstm_20/while/Identity_4:output:0"Y
&sequential_20_lstm_20_while_identity_5/sequential_20/lstm_20/while/Identity_5:output:0"
Hsequential_20_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resourceJsequential_20_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"
Isequential_20_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resourceKsequential_20_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"
Gsequential_20_lstm_20_while_lstm_cell_20_matmul_readvariableop_resourceIsequential_20_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"
Asequential_20_lstm_20_while_sequential_20_lstm_20_strided_slice_1Csequential_20_lstm_20_while_sequential_20_lstm_20_strided_slice_1_0"
}sequential_20_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_20_tensorarrayunstack_tensorlistfromtensorsequential_20_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_20_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?sequential_20/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2
>sequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp>sequential_20/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2
@sequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp@sequential_20/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


è
lstm_20_while_cond_1590994,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1E
Alstm_20_while_lstm_20_while_cond_1590994___redundant_placeholder0E
Alstm_20_while_lstm_20_while_cond_1590994___redundant_placeholder1E
Alstm_20_while_lstm_20_while_cond_1590994___redundant_placeholder2E
Alstm_20_while_lstm_20_while_cond_1590994___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: [
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ù
´
)__inference_lstm_20_layer_call_fn_1591158

inputs
unknown:	d
	unknown_0:d
	unknown_1:d
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
È	
ö
E__inference_dense_41_layer_call_and_return_conditional_losses_1591768

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589848

inputs

states
states_11
matmul_readvariableop_resource:	d2
 matmul_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
È	
ö
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­	
¨
/__inference_sequential_20_layer_call_fn_1590732

inputs
unknown:		
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:d
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall 
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
GPU 2J 8 *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590593o
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
Ä

*__inference_dense_41_layer_call_fn_1591758

inputs
unknown:
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
È
while_cond_1590055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1590055___redundant_placeholder05
1while_while_cond_1590055___redundant_placeholder15
1while_while_cond_1590055___redundant_placeholder25
1while_while_cond_1590055___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
 9
Í
while_body_1590447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
í
õ
.__inference_lstm_cell_20_layer_call_fn_1591802

inputs
states_0
states_1
unknown:	d
	unknown_0:d
	unknown_1:d
identity

identity_1

identity_2¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
Õ
þ
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172

inputs4
!tensordot_readvariableop_resource:		.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:		*
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
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
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
:ÿÿÿÿÿÿÿÿÿxs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxz
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
¯z
Ì
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590909

inputs=
*dense_40_tensordot_readvariableop_resource:		7
(dense_40_biasadd_readvariableop_resource:	F
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	dG
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:dB
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:d9
'dense_41_matmul_readvariableop_resource:6
(dense_41_biasadd_readvariableop_resource:
identity¢dense_40/BiasAdd/ReadVariableOp¢!dense_40/Tensordot/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes
:		*
dtype0a
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_40/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_40/Tensordot/GatherV2_1GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/axes:output:0+dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_40/Tensordot/transpose	Transposeinputs"dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxV
lstm_20/ShapeShapedense_40/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_20/transpose	Transposedense_40/BiasAdd:output:0lstm_20/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿT
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:g
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0­
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0§
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¬
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_20/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0-lstm_20/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_20_while_body_1590818*&
condR
lstm_20_while_cond_1590817*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   î
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_41/MatMulMatMul lstm_20/strided_slice_3:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_41/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
NoOpNoOp ^dense_40/BiasAdd/ReadVariableOp"^dense_40/Tensordot/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2F
!dense_40/Tensordot/ReadVariableOp!dense_40/Tensordot/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
ù
´
)__inference_lstm_20_layer_call_fn_1591169

inputs
unknown:	d
	unknown_0:d
	unknown_1:d
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ê
É
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590671
dense_40_input#
dense_40_1590653:		
dense_40_1590655:	"
lstm_20_1590658:	d!
lstm_20_1590660:d
lstm_20_1590662:d"
dense_41_1590665:
dense_41_1590667:
identity¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall
 dense_40/StatefulPartitionedCallStatefulPartitionedCalldense_40_inputdense_40_1590653dense_40_1590655*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172¥
lstm_20/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_20_1590658lstm_20_1590660lstm_20_1590662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590532
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0dense_41_1590665dense_41_1590667*
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340x
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_40_input
º
È
while_cond_1590446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1590446___redundant_placeholder05
1while_while_cond_1590446___redundant_placeholder15
1while_while_cond_1590446___redundant_placeholder25
1while_while_cond_1590446___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ËK

D__inference_lstm_20_layer_call_and_return_conditional_losses_1591459
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while=
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1591374*
condR
while_cond_1591373*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
$
æ
while_body_1589863
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_20_1589887_0:	d.
while_lstm_cell_20_1589889_0:d*
while_lstm_cell_20_1589891_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_20_1589887:	d,
while_lstm_cell_20_1589889:d(
while_lstm_cell_20_1589891:d¢*while/lstm_cell_20/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_1589887_0while_lstm_cell_20_1589889_0while_lstm_cell_20_1589891_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589848r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_20_1589887while_lstm_cell_20_1589887_0":
while_lstm_cell_20_1589889while_lstm_cell_20_1589889_0":
while_lstm_cell_20_1589891while_lstm_cell_20_1589891_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
 9
Í
while_body_1591229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ê
É
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590650
dense_40_input#
dense_40_1590632:		
dense_40_1590634:	"
lstm_20_1590637:	d!
lstm_20_1590639:d
lstm_20_1590641:d"
dense_41_1590644:
dense_41_1590646:
identity¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall
 dense_40/StatefulPartitionedCallStatefulPartitionedCalldense_40_inputdense_40_1590632dense_40_1590634*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172¥
lstm_20/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_20_1590637lstm_20_1590639lstm_20_1590641*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590322
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0dense_41_1590644dense_41_1590646*
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340x
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_40_input
9

D__inference_lstm_20_layer_call_and_return_conditional_losses_1589933

inputs'
lstm_cell_20_1589849:	d&
lstm_cell_20_1589851:d"
lstm_cell_20_1589853:d
identity¢$lstm_cell_20/StatefulPartitionedCall¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_1589849lstm_cell_20_1589851lstm_cell_20_1589853*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589848n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_1589849lstm_cell_20_1589851lstm_cell_20_1589853*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1589863*
condR
while_cond_1589862*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
$
æ
while_body_1590056
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_20_1590080_0:	d.
while_lstm_cell_20_1590082_0:d*
while_lstm_cell_20_1590084_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_20_1590080:	d,
while_lstm_cell_20_1590082:d(
while_lstm_cell_20_1590084:d¢*while/lstm_cell_20/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0·
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_1590080_0while_lstm_cell_20_1590082_0while_lstm_cell_20_1590084_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589996r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_20_1590080while_lstm_cell_20_1590080_0":
while_lstm_cell_20_1590082while_lstm_cell_20_1590082_0":
while_lstm_cell_20_1590084while_lstm_cell_20_1590084_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¨K

D__inference_lstm_20_layer_call_and_return_conditional_losses_1591604

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1591519*
condR
while_cond_1591518*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
ç
þ
"__inference__wrapped_model_1589781
dense_40_inputK
8sequential_20_dense_40_tensordot_readvariableop_resource:		E
6sequential_20_dense_40_biasadd_readvariableop_resource:	T
Asequential_20_lstm_20_lstm_cell_20_matmul_readvariableop_resource:	dU
Csequential_20_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:dP
Bsequential_20_lstm_20_lstm_cell_20_biasadd_readvariableop_resource:dG
5sequential_20_dense_41_matmul_readvariableop_resource:D
6sequential_20_dense_41_biasadd_readvariableop_resource:
identity¢-sequential_20/dense_40/BiasAdd/ReadVariableOp¢/sequential_20/dense_40/Tensordot/ReadVariableOp¢-sequential_20/dense_41/BiasAdd/ReadVariableOp¢,sequential_20/dense_41/MatMul/ReadVariableOp¢9sequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢8sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢:sequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢sequential_20/lstm_20/while©
/sequential_20/dense_40/Tensordot/ReadVariableOpReadVariableOp8sequential_20_dense_40_tensordot_readvariableop_resource*
_output_shapes
:		*
dtype0o
%sequential_20/dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_20/dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
&sequential_20/dense_40/Tensordot/ShapeShapedense_40_input*
T0*
_output_shapes
:p
.sequential_20/dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_20/dense_40/Tensordot/GatherV2GatherV2/sequential_20/dense_40/Tensordot/Shape:output:0.sequential_20/dense_40/Tensordot/free:output:07sequential_20/dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_20/dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+sequential_20/dense_40/Tensordot/GatherV2_1GatherV2/sequential_20/dense_40/Tensordot/Shape:output:0.sequential_20/dense_40/Tensordot/axes:output:09sequential_20/dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_20/dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ³
%sequential_20/dense_40/Tensordot/ProdProd2sequential_20/dense_40/Tensordot/GatherV2:output:0/sequential_20/dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_20/dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
'sequential_20/dense_40/Tensordot/Prod_1Prod4sequential_20/dense_40/Tensordot/GatherV2_1:output:01sequential_20/dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_20/dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ø
'sequential_20/dense_40/Tensordot/concatConcatV2.sequential_20/dense_40/Tensordot/free:output:0.sequential_20/dense_40/Tensordot/axes:output:05sequential_20/dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¾
&sequential_20/dense_40/Tensordot/stackPack.sequential_20/dense_40/Tensordot/Prod:output:00sequential_20/dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
*sequential_20/dense_40/Tensordot/transpose	Transposedense_40_input0sequential_20/dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	Ï
(sequential_20/dense_40/Tensordot/ReshapeReshape.sequential_20/dense_40/Tensordot/transpose:y:0/sequential_20/dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
'sequential_20/dense_40/Tensordot/MatMulMatMul1sequential_20/dense_40/Tensordot/Reshape:output:07sequential_20/dense_40/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
(sequential_20/dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:p
.sequential_20/dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_20/dense_40/Tensordot/concat_1ConcatV22sequential_20/dense_40/Tensordot/GatherV2:output:01sequential_20/dense_40/Tensordot/Const_2:output:07sequential_20/dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:É
 sequential_20/dense_40/TensordotReshape1sequential_20/dense_40/Tensordot/MatMul:product:02sequential_20/dense_40/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx¡
-sequential_20/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
sequential_20/dense_40/BiasAddBiasAdd)sequential_20/dense_40/Tensordot:output:05sequential_20/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxr
sequential_20/lstm_20/ShapeShape'sequential_20/dense_40/BiasAdd:output:0*
T0*
_output_shapes
:s
)sequential_20/lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_20/lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_20/lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_20/lstm_20/strided_sliceStridedSlice$sequential_20/lstm_20/Shape:output:02sequential_20/lstm_20/strided_slice/stack:output:04sequential_20/lstm_20/strided_slice/stack_1:output:04sequential_20/lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_20/lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :µ
"sequential_20/lstm_20/zeros/packedPack,sequential_20/lstm_20/strided_slice:output:0-sequential_20/lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_20/lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_20/lstm_20/zerosFill+sequential_20/lstm_20/zeros/packed:output:0*sequential_20/lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&sequential_20/lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¹
$sequential_20/lstm_20/zeros_1/packedPack,sequential_20/lstm_20/strided_slice:output:0/sequential_20/lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_20/lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
sequential_20/lstm_20/zeros_1Fill-sequential_20/lstm_20/zeros_1/packed:output:0,sequential_20/lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_20/lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          »
sequential_20/lstm_20/transpose	Transpose'sequential_20/dense_40/BiasAdd:output:0-sequential_20/lstm_20/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿp
sequential_20/lstm_20/Shape_1Shape#sequential_20/lstm_20/transpose:y:0*
T0*
_output_shapes
:u
+sequential_20/lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%sequential_20/lstm_20/strided_slice_1StridedSlice&sequential_20/lstm_20/Shape_1:output:04sequential_20/lstm_20/strided_slice_1/stack:output:06sequential_20/lstm_20/strided_slice_1/stack_1:output:06sequential_20/lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_20/lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#sequential_20/lstm_20/TensorArrayV2TensorListReserve:sequential_20/lstm_20/TensorArrayV2/element_shape:output:0.sequential_20/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_20/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¢
=sequential_20/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_20/lstm_20/transpose:y:0Tsequential_20/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+sequential_20/lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_20/lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
%sequential_20/lstm_20/strided_slice_2StridedSlice#sequential_20/lstm_20/transpose:y:04sequential_20/lstm_20/strided_slice_2/stack:output:06sequential_20/lstm_20/strided_slice_2/stack_1:output:06sequential_20/lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask»
8sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpAsequential_20_lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0×
)sequential_20/lstm_20/lstm_cell_20/MatMulMatMul.sequential_20/lstm_20/strided_slice_2:output:0@sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¾
:sequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpCsequential_20_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0Ñ
+sequential_20/lstm_20/lstm_cell_20/MatMul_1MatMul$sequential_20/lstm_20/zeros:output:0Bsequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ
&sequential_20/lstm_20/lstm_cell_20/addAddV23sequential_20/lstm_20/lstm_cell_20/MatMul:product:05sequential_20/lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¸
9sequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpBsequential_20_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Ö
*sequential_20/lstm_20/lstm_cell_20/BiasAddBiasAdd*sequential_20/lstm_20/lstm_cell_20/add:z:0Asequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
2sequential_20/lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
(sequential_20/lstm_20/lstm_cell_20/splitSplit;sequential_20/lstm_20/lstm_cell_20/split/split_dim:output:03sequential_20/lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
*sequential_20/lstm_20/lstm_cell_20/SigmoidSigmoid1sequential_20/lstm_20/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_20/lstm_20/lstm_cell_20/Sigmoid_1Sigmoid1sequential_20/lstm_20/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
&sequential_20/lstm_20/lstm_cell_20/mulMul0sequential_20/lstm_20/lstm_cell_20/Sigmoid_1:y:0&sequential_20/lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential_20/lstm_20/lstm_cell_20/ReluRelu1sequential_20/lstm_20/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_20/lstm_20/lstm_cell_20/mul_1Mul.sequential_20/lstm_20/lstm_cell_20/Sigmoid:y:05sequential_20/lstm_20/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
(sequential_20/lstm_20/lstm_cell_20/add_1AddV2*sequential_20/lstm_20/lstm_cell_20/mul:z:0,sequential_20/lstm_20/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_20/lstm_20/lstm_cell_20/Sigmoid_2Sigmoid1sequential_20/lstm_20/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_20/lstm_20/lstm_cell_20/Relu_1Relu,sequential_20/lstm_20/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
(sequential_20/lstm_20/lstm_cell_20/mul_2Mul0sequential_20/lstm_20/lstm_cell_20/Sigmoid_2:y:07sequential_20/lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_20/lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   t
2sequential_20/lstm_20/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_20/lstm_20/TensorArrayV2_1TensorListReserve<sequential_20/lstm_20/TensorArrayV2_1/element_shape:output:0;sequential_20/lstm_20/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
sequential_20/lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_20/lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(sequential_20/lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¸
sequential_20/lstm_20/whileWhile1sequential_20/lstm_20/while/loop_counter:output:07sequential_20/lstm_20/while/maximum_iterations:output:0#sequential_20/lstm_20/time:output:0.sequential_20/lstm_20/TensorArrayV2_1:handle:0$sequential_20/lstm_20/zeros:output:0&sequential_20/lstm_20/zeros_1:output:0.sequential_20/lstm_20/strided_slice_1:output:0Msequential_20/lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_20_lstm_20_lstm_cell_20_matmul_readvariableop_resourceCsequential_20_lstm_20_lstm_cell_20_matmul_1_readvariableop_resourceBsequential_20_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_20_lstm_20_while_body_1589690*4
cond,R*
(sequential_20_lstm_20_while_cond_1589689*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Fsequential_20/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8sequential_20/lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_20/lstm_20/while:output:3Osequential_20/lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elements~
+sequential_20/lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-sequential_20/lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_20/lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%sequential_20/lstm_20/strided_slice_3StridedSliceAsequential_20/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:04sequential_20/lstm_20/strided_slice_3/stack:output:06sequential_20/lstm_20/strided_slice_3/stack_1:output:06sequential_20/lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask{
&sequential_20/lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!sequential_20/lstm_20/transpose_1	TransposeAsequential_20/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_20/lstm_20/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_20/lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¢
,sequential_20/dense_41/MatMul/ReadVariableOpReadVariableOp5sequential_20_dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¿
sequential_20/dense_41/MatMulMatMul.sequential_20/lstm_20/strided_slice_3:output:04sequential_20/dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_20/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_20/dense_41/BiasAddBiasAdd'sequential_20/dense_41/MatMul:product:05sequential_20/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_20/dense_41/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
NoOpNoOp.^sequential_20/dense_40/BiasAdd/ReadVariableOp0^sequential_20/dense_40/Tensordot/ReadVariableOp.^sequential_20/dense_41/BiasAdd/ReadVariableOp-^sequential_20/dense_41/MatMul/ReadVariableOp:^sequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9^sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOp;^sequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^sequential_20/lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2^
-sequential_20/dense_40/BiasAdd/ReadVariableOp-sequential_20/dense_40/BiasAdd/ReadVariableOp2b
/sequential_20/dense_40/Tensordot/ReadVariableOp/sequential_20/dense_40/Tensordot/ReadVariableOp2^
-sequential_20/dense_41/BiasAdd/ReadVariableOp-sequential_20/dense_41/BiasAdd/ReadVariableOp2\
,sequential_20/dense_41/MatMul/ReadVariableOp,sequential_20/dense_41/MatMul/ReadVariableOp2v
9sequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9sequential_20/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2t
8sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOp8sequential_20/lstm_20/lstm_cell_20/MatMul/ReadVariableOp2x
:sequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:sequential_20/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2:
sequential_20/lstm_20/whilesequential_20/lstm_20/while:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
(
_user_specified_namedense_40_input
­	
¨
/__inference_sequential_20_layer_call_fn_1590713

inputs
unknown:		
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:d
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall 
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
GPU 2J 8 *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590347o
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
 9
Í
while_body_1590237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø

*__inference_dense_40_layer_call_fn_1591095

inputs
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx`
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
Ø

I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591866

inputs
states_0
states_11
matmul_readvariableop_resource:	d2
 matmul_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
Ò
Á
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590347

inputs#
dense_40_1590173:		
dense_40_1590175:	"
lstm_20_1590323:	d!
lstm_20_1590325:d
lstm_20_1590327:d"
dense_41_1590341:
dense_41_1590343:
identity¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCallø
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_1590173dense_40_1590175*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172¥
lstm_20/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_20_1590323lstm_20_1590325lstm_20_1590327*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590322
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0dense_41_1590341dense_41_1590343*
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340x
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
º
È
while_cond_1591228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1591228___redundant_placeholder05
1while_while_cond_1591228___redundant_placeholder15
1while_while_cond_1591228___redundant_placeholder25
1while_while_cond_1591228___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ø

I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591834

inputs
states_0
states_11
matmul_readvariableop_resource:	d2
 matmul_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states_1
º
È
while_cond_1589862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1589862___redundant_placeholder05
1while_while_cond_1589862___redundant_placeholder15
1while_while_cond_1589862___redundant_placeholder25
1while_while_cond_1589862___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
	
¦
%__inference_signature_wrapper_1590694
dense_40_input
unknown:		
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:d
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
"__inference__wrapped_model_1589781o
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
_user_specified_namedense_40_input
 9
Í
while_body_1591664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_1591373
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1591373___redundant_placeholder05
1while_while_cond_1591373___redundant_placeholder15
1while_while_cond_1591373___redundant_placeholder25
1while_while_cond_1591373___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ð

I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589996

inputs

states
states_11
matmul_readvariableop_resource:	d2
 matmul_1_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates

¶
)__inference_lstm_20_layer_call_fn_1591136
inputs_0
unknown:	d
	unknown_0:d
	unknown_1:d
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1589933o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
º
È
while_cond_1591663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1591663___redundant_placeholder05
1while_while_cond_1591663___redundant_placeholder15
1while_while_cond_1591663___redundant_placeholder25
1while_while_cond_1591663___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

¶
)__inference_lstm_20_layer_call_fn_1591147
inputs_0
unknown:	d
	unknown_0:d
	unknown_1:d
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590126o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs_0
áB
Í

lstm_20_while_body_1590818,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	dO
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:dJ
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:d
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	dM
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:dH
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:d¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ñ
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0¸
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¾
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_20/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1Alstm_20/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: q
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
áB
Í

lstm_20_while_body_1590995,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	dO
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:dJ
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:d
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	dM
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:dH
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:d¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ï
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0­
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Ñ
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd°
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0¸
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdµ
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¾
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdl
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8lstm_20/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1Alstm_20/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: q
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ò
Á
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590593

inputs#
dense_40_1590575:		
dense_40_1590577:	"
lstm_20_1590580:	d!
lstm_20_1590582:d
lstm_20_1590584:d"
dense_41_1590587:
dense_41_1590589:
identity¢ dense_40/StatefulPartitionedCall¢ dense_41/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCallø
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_1590575dense_40_1590577*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_1590172¥
lstm_20/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_20_1590580lstm_20_1590582lstm_20_1590584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_20_layer_call_and_return_conditional_losses_1590532
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0dense_41_1590587dense_41_1590589*
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1590340x
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
NoOpNoOp!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
_user_specified_nameinputs
 9
Í
while_body_1591374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_1591518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1591518___redundant_placeholder05
1while_while_cond_1591518___redundant_placeholder15
1while_while_cond_1591518___redundant_placeholder25
1while_while_cond_1591518___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Õ
þ
E__inference_dense_40_layer_call_and_return_conditional_losses_1591125

inputs4
!tensordot_readvariableop_resource:		.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:		*
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
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
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
:ÿÿÿÿÿÿÿÿÿxs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxd
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxz
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
¨K

D__inference_lstm_20_layer_call_and_return_conditional_losses_1591749

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1591664*
condR
while_cond_1591663*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
Ú|
®
#__inference__traced_restore_1592073
file_prefix3
 assignvariableop_dense_40_kernel:		/
 assignvariableop_1_dense_40_bias:	4
"assignvariableop_2_dense_41_kernel:.
 assignvariableop_3_dense_41_bias:A
.assignvariableop_4_lstm_20_lstm_cell_20_kernel:	dJ
8assignvariableop_5_lstm_20_lstm_cell_20_recurrent_kernel:d:
,assignvariableop_6_lstm_20_lstm_cell_20_bias:d&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: <
)assignvariableop_9_adam_m_dense_40_kernel:		=
*assignvariableop_10_adam_v_dense_40_kernel:		7
(assignvariableop_11_adam_m_dense_40_bias:	7
(assignvariableop_12_adam_v_dense_40_bias:	I
6assignvariableop_13_adam_m_lstm_20_lstm_cell_20_kernel:	dI
6assignvariableop_14_adam_v_lstm_20_lstm_cell_20_kernel:	dR
@assignvariableop_15_adam_m_lstm_20_lstm_cell_20_recurrent_kernel:dR
@assignvariableop_16_adam_v_lstm_20_lstm_cell_20_recurrent_kernel:dB
4assignvariableop_17_adam_m_lstm_20_lstm_cell_20_bias:dB
4assignvariableop_18_adam_v_lstm_20_lstm_cell_20_bias:d<
*assignvariableop_19_adam_m_dense_41_kernel:<
*assignvariableop_20_adam_v_dense_41_kernel:6
(assignvariableop_21_adam_m_dense_41_bias:6
(assignvariableop_22_adam_v_dense_41_bias:%
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
AssignVariableOpAssignVariableOp assignvariableop_dense_40_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_40_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_41_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_41_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Å
AssignVariableOp_4AssignVariableOp.assignvariableop_4_lstm_20_lstm_cell_20_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_5AssignVariableOp8assignvariableop_5_lstm_20_lstm_cell_20_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_6AssignVariableOp,assignvariableop_6_lstm_20_lstm_cell_20_biasIdentity_6:output:0"/device:CPU:0*&
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
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_m_dense_40_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_v_dense_40_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_11AssignVariableOp(assignvariableop_11_adam_m_dense_40_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_v_dense_40_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_13AssignVariableOp6assignvariableop_13_adam_m_lstm_20_lstm_cell_20_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_v_lstm_20_lstm_cell_20_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ù
AssignVariableOp_15AssignVariableOp@assignvariableop_15_adam_m_lstm_20_lstm_cell_20_recurrent_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ù
AssignVariableOp_16AssignVariableOp@assignvariableop_16_adam_v_lstm_20_lstm_cell_20_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_m_lstm_20_lstm_cell_20_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Í
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_v_lstm_20_lstm_cell_20_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_41_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ã
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_41_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_41_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_41_biasIdentity_22:output:0"/device:CPU:0*&
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
 9
Í
while_body_1591519
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	dG
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:dB
4while_lstm_cell_20_biasadd_readvariableop_resource_0:d
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	dE
3while_lstm_cell_20_matmul_1_readvariableop_resource:d@
2while_lstm_cell_20_biasadd_readvariableop_resource:d¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	d*
dtype0¹
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd 
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0*
_output_shapes

:d*
dtype0 
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0¦
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitz
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : í
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
º
È
while_cond_1590236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_15
1while_while_cond_1590236___redundant_placeholder05
1while_while_cond_1590236___redundant_placeholder15
1while_while_cond_1590236___redundant_placeholder25
1while_while_cond_1590236___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


è
lstm_20_while_cond_1590817,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1E
Alstm_20_while_lstm_20_while_cond_1590817___redundant_placeholder0E
Alstm_20_while_lstm_20_while_cond_1590817___redundant_placeholder1E
Alstm_20_while_lstm_20_while_cond_1590817___redundant_placeholder2E
Alstm_20_while_lstm_20_while_cond_1590817___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: [
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Å	
°
/__inference_sequential_20_layer_call_fn_1590364
dense_40_input
unknown:		
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:d
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590347o
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
_user_specified_namedense_40_input
Å	
°
/__inference_sequential_20_layer_call_fn_1590629
dense_40_input
unknown:		
	unknown_0:	
	unknown_1:	d
	unknown_2:d
	unknown_3:d
	unknown_4:
	unknown_5:
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCalldense_40_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590593o
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
_user_specified_namedense_40_input
î>
û
 __inference__traced_save_1591976
file_prefix.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop:
6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableopD
@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop8
4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop5
1savev2_adam_m_dense_40_kernel_read_readvariableop5
1savev2_adam_v_dense_40_kernel_read_readvariableop3
/savev2_adam_m_dense_40_bias_read_readvariableop3
/savev2_adam_v_dense_40_bias_read_readvariableopA
=savev2_adam_m_lstm_20_lstm_cell_20_kernel_read_readvariableopA
=savev2_adam_v_lstm_20_lstm_cell_20_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_20_lstm_cell_20_bias_read_readvariableop?
;savev2_adam_v_lstm_20_lstm_cell_20_bias_read_readvariableop5
1savev2_adam_m_dense_41_kernel_read_readvariableop5
1savev2_adam_v_dense_41_kernel_read_readvariableop3
/savev2_adam_m_dense_41_bias_read_readvariableop3
/savev2_adam_v_dense_41_bias_read_readvariableop&
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
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableop@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop1savev2_adam_m_dense_40_kernel_read_readvariableop1savev2_adam_v_dense_40_kernel_read_readvariableop/savev2_adam_m_dense_40_bias_read_readvariableop/savev2_adam_v_dense_40_bias_read_readvariableop=savev2_adam_m_lstm_20_lstm_cell_20_kernel_read_readvariableop=savev2_adam_v_lstm_20_lstm_cell_20_kernel_read_readvariableopGsavev2_adam_m_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_20_lstm_cell_20_bias_read_readvariableop;savev2_adam_v_lstm_20_lstm_cell_20_bias_read_readvariableop1savev2_adam_m_dense_41_kernel_read_readvariableop1savev2_adam_v_dense_41_kernel_read_readvariableop/savev2_adam_m_dense_41_bias_read_readvariableop/savev2_adam_v_dense_41_bias_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
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

identity_1Identity_1:output:0*à
_input_shapesÎ
Ë: :		::::	d:d:d: : :		:		:::	d:	d:d:d:d:d::::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:		:!

_output_shapes	
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	d:$ 

_output_shapes

:d: 

_output_shapes
:d:

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:		:%!

_output_shapes
:		:!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	d:%!

_output_shapes
:	d:$ 

_output_shapes

:d:$ 

_output_shapes

:d: 

_output_shapes
:d: 

_output_shapes
:d:$ 

_output_shapes

::$ 

_output_shapes

:: 
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
£

(sequential_20_lstm_20_while_cond_1589689H
Dsequential_20_lstm_20_while_sequential_20_lstm_20_while_loop_counterN
Jsequential_20_lstm_20_while_sequential_20_lstm_20_while_maximum_iterations+
'sequential_20_lstm_20_while_placeholder-
)sequential_20_lstm_20_while_placeholder_1-
)sequential_20_lstm_20_while_placeholder_2-
)sequential_20_lstm_20_while_placeholder_3J
Fsequential_20_lstm_20_while_less_sequential_20_lstm_20_strided_slice_1a
]sequential_20_lstm_20_while_sequential_20_lstm_20_while_cond_1589689___redundant_placeholder0a
]sequential_20_lstm_20_while_sequential_20_lstm_20_while_cond_1589689___redundant_placeholder1a
]sequential_20_lstm_20_while_sequential_20_lstm_20_while_cond_1589689___redundant_placeholder2a
]sequential_20_lstm_20_while_sequential_20_lstm_20_while_cond_1589689___redundant_placeholder3(
$sequential_20_lstm_20_while_identity
º
 sequential_20/lstm_20/while/LessLess'sequential_20_lstm_20_while_placeholderFsequential_20_lstm_20_while_less_sequential_20_lstm_20_strided_slice_1*
T0*
_output_shapes
: w
$sequential_20/lstm_20/while/IdentityIdentity$sequential_20/lstm_20/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_20_lstm_20_while_identity-sequential_20/lstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
9

D__inference_lstm_20_layer_call_and_return_conditional_losses_1590126

inputs'
lstm_cell_20_1590042:	d&
lstm_cell_20_1590044:d"
lstm_cell_20_1590046:d
identity¢$lstm_cell_20/StatefulPartitionedCall¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskù
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_1590042lstm_cell_20_1590044lstm_cell_20_1590046*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1589996n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_1590042lstm_cell_20_1590044lstm_cell_20_1590046*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1590056*
condR
while_cond_1590055*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨K

D__inference_lstm_20_layer_call_and_return_conditional_losses_1590532

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	d?
-lstm_cell_20_matmul_1_readvariableop_resource:d:
,lstm_cell_20_biasadd_readvariableop_resource:d
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢while;
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
value	B :s
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
:ÿÿÿÿÿÿÿÿÿR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿD
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
valueB"ÿÿÿÿ   à
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd^
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitn
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_1590447*
condR
while_cond_1590446*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿx: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
 
_user_specified_nameinputs
¯z
Ì
J__inference_sequential_20_layer_call_and_return_conditional_losses_1591086

inputs=
*dense_40_tensordot_readvariableop_resource:		7
(dense_40_biasadd_readvariableop_resource:	F
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	dG
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:dB
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:d9
'dense_41_matmul_readvariableop_resource:6
(dense_41_biasadd_readvariableop_resource:
identity¢dense_40/BiasAdd/ReadVariableOp¢!dense_40/Tensordot/ReadVariableOp¢dense_41/BiasAdd/ReadVariableOp¢dense_41/MatMul/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes
:		*
dtype0a
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_40/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_40/Tensordot/GatherV2_1GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/axes:output:0+dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_40/Tensordot/transpose	Transposeinputs"dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	¥
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿxV
lstm_20/ShapeShapedense_40/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_20/transpose	Transposedense_40/BiasAdd:output:0lstm_20/transpose/perm:output:0*
T0*,
_output_shapes
:xÿÿÿÿÿÿÿÿÿT
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:g
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	d*
dtype0­
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource*
_output_shapes

:d*
dtype0§
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd£
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0¬
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :õ
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split~
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   f
$lstm_20/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ý
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0-lstm_20/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ô
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_20_while_body_1590995*&
condR
lstm_20_while_cond_1590994*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   î
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*
num_elementsp
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_41/MatMulMatMul lstm_20/strided_slice_3:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_41/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿé
NoOpNoOp ^dense_40/BiasAdd/ReadVariableOp"^dense_40/Tensordot/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿx	: : : : : : : 2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2F
!dense_40/Tensordot/ReadVariableOp!dense_40/Tensordot/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿx	
 
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
dense_40_input;
 serving_default_dense_40_input:0ÿÿÿÿÿÿÿÿÿx	<
dense_410
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:úµ
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
ñ
.trace_0
/trace_1
0trace_2
1trace_32
/__inference_sequential_20_layer_call_fn_1590364
/__inference_sequential_20_layer_call_fn_1590713
/__inference_sequential_20_layer_call_fn_1590732
/__inference_sequential_20_layer_call_fn_1590629¿
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
Ý
2trace_0
3trace_1
4trace_2
5trace_32ò
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590909
J__inference_sequential_20_layer_call_and_return_conditional_losses_1591086
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590650
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590671¿
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
"__inference__wrapped_model_1589781dense_40_input"
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
*__inference_dense_40_layer_call_fn_1591095¢
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
E__inference_dense_40_layer_call_and_return_conditional_losses_1591125¢
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
": 		2dense_40/kernel
:2dense_40/bias
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
î
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32
)__inference_lstm_20_layer_call_fn_1591136
)__inference_lstm_20_layer_call_fn_1591147
)__inference_lstm_20_layer_call_fn_1591158
)__inference_lstm_20_layer_call_fn_1591169Ô
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
Ú
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32ï
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591314
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591459
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591604
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591749Ô
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
*__inference_dense_41_layer_call_fn_1591758¢
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1591768¢
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
!:2dense_41/kernel
:2dense_41/bias
.:,	d2lstm_20/lstm_cell_20/kernel
7:5d2%lstm_20/lstm_cell_20/recurrent_kernel
':%d2lstm_20/lstm_cell_20/bias
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
B
/__inference_sequential_20_layer_call_fn_1590364dense_40_input"¿
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
Bý
/__inference_sequential_20_layer_call_fn_1590713inputs"¿
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
Bý
/__inference_sequential_20_layer_call_fn_1590732inputs"¿
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
B
/__inference_sequential_20_layer_call_fn_1590629dense_40_input"¿
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
B
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590909inputs"¿
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
B
J__inference_sequential_20_layer_call_and_return_conditional_losses_1591086inputs"¿
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
£B 
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590650dense_40_input"¿
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
£B 
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590671dense_40_input"¿
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
%__inference_signature_wrapper_1590694dense_40_input"
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
*__inference_dense_40_layer_call_fn_1591095inputs"¢
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
E__inference_dense_40_layer_call_and_return_conditional_losses_1591125inputs"¢
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
B
)__inference_lstm_20_layer_call_fn_1591136inputs_0"Ô
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
B
)__inference_lstm_20_layer_call_fn_1591147inputs_0"Ô
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
)__inference_lstm_20_layer_call_fn_1591158inputs"Ô
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
)__inference_lstm_20_layer_call_fn_1591169inputs"Ô
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
¬B©
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591314inputs_0"Ô
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
¬B©
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591459inputs_0"Ô
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
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591604inputs"Ô
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
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591749inputs"Ô
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
×
xtrace_0
ytrace_12 
.__inference_lstm_cell_20_layer_call_fn_1591785
.__inference_lstm_cell_20_layer_call_fn_1591802½
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

ztrace_0
{trace_12Ö
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591834
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591866½
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
*__inference_dense_41_layer_call_fn_1591758inputs"¢
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
E__inference_dense_41_layer_call_and_return_conditional_losses_1591768inputs"¢
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
':%		2Adam/m/dense_40/kernel
':%		2Adam/v/dense_40/kernel
!:2Adam/m/dense_40/bias
!:2Adam/v/dense_40/bias
3:1	d2"Adam/m/lstm_20/lstm_cell_20/kernel
3:1	d2"Adam/v/lstm_20/lstm_cell_20/kernel
<::d2,Adam/m/lstm_20/lstm_cell_20/recurrent_kernel
<::d2,Adam/v/lstm_20/lstm_cell_20/recurrent_kernel
,:*d2 Adam/m/lstm_20/lstm_cell_20/bias
,:*d2 Adam/v/lstm_20/lstm_cell_20/bias
&:$2Adam/m/dense_41/kernel
&:$2Adam/v/dense_41/kernel
 :2Adam/m/dense_41/bias
 :2Adam/v/dense_41/bias
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
B
.__inference_lstm_cell_20_layer_call_fn_1591785inputsstates_0states_1"½
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
B
.__inference_lstm_cell_20_layer_call_fn_1591802inputsstates_0states_1"½
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
¬B©
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591834inputsstates_0states_1"½
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
¬B©
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591866inputsstates_0states_1"½
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
"__inference__wrapped_model_1589781{&'($%;¢8
1¢.
,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	
ª "3ª0
.
dense_41"
dense_41ÿÿÿÿÿÿÿÿÿµ
E__inference_dense_40_layer_call_and_return_conditional_losses_1591125l3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "1¢.
'$
tensor_0ÿÿÿÿÿÿÿÿÿx
 
*__inference_dense_40_layer_call_fn_1591095a3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿx	
ª "&#
unknownÿÿÿÿÿÿÿÿÿx¬
E__inference_dense_41_layer_call_and_return_conditional_losses_1591768c$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_41_layer_call_fn_1591758X$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿÎ
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591314&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Î
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591459&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ½
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591604u&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿx

 
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ½
D__inference_lstm_20_layer_call_and_return_conditional_losses_1591749u&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿx

 
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 §
)__inference_lstm_20_layer_call_fn_1591136z&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ§
)__inference_lstm_20_layer_call_fn_1591147z&'(P¢M
F¢C
52
0-
inputs_0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
)__inference_lstm_20_layer_call_fn_1591158j&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿx

 
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
)__inference_lstm_20_layer_call_fn_1591169j&'(@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿx

 
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿã
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591834&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿ
"
states_1ÿÿÿÿÿÿÿÿÿ
p 
ª "¢
~¢{
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
SP
&#
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
&#
tensor_0_1_1ÿÿÿÿÿÿÿÿÿ
 ã
I__inference_lstm_cell_20_layer_call_and_return_conditional_losses_1591866&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿ
"
states_1ÿÿÿÿÿÿÿÿÿ
p
ª "¢
~¢{
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
SP
&#
tensor_0_1_0ÿÿÿÿÿÿÿÿÿ
&#
tensor_0_1_1ÿÿÿÿÿÿÿÿÿ
 ¶
.__inference_lstm_cell_20_layer_call_fn_1591785&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿ
"
states_1ÿÿÿÿÿÿÿÿÿ
p 
ª "x¢u
"
tensor_0ÿÿÿÿÿÿÿÿÿ
OL
$!

tensor_1_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_1_1ÿÿÿÿÿÿÿÿÿ¶
.__inference_lstm_cell_20_layer_call_fn_1591802&'(¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states_0ÿÿÿÿÿÿÿÿÿ
"
states_1ÿÿÿÿÿÿÿÿÿ
p
ª "x¢u
"
tensor_0ÿÿÿÿÿÿÿÿÿ
OL
$!

tensor_1_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_1_1ÿÿÿÿÿÿÿÿÿÊ
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590650|&'($%C¢@
9¢6
,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Ê
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590671|&'($%C¢@
9¢6
,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_sequential_20_layer_call_and_return_conditional_losses_1590909t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Â
J__inference_sequential_20_layer_call_and_return_conditional_losses_1591086t&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ¤
/__inference_sequential_20_layer_call_fn_1590364q&'($%C¢@
9¢6
,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ¤
/__inference_sequential_20_layer_call_fn_1590629q&'($%C¢@
9¢6
,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
/__inference_sequential_20_layer_call_fn_1590713i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
/__inference_sequential_20_layer_call_fn_1590732i&'($%;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿx	
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ·
%__inference_signature_wrapper_1590694&'($%M¢J
¢ 
Cª@
>
dense_40_input,)
dense_40_inputÿÿÿÿÿÿÿÿÿx	"3ª0
.
dense_41"
dense_41ÿÿÿÿÿÿÿÿÿ