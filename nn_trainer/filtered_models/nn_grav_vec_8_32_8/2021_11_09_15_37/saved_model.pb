í
ÿÓ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
H
ShardedFilename
basename	
shard

num_shards
filename
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.12v2.6.0-101-g3aa40c3ce9d8ª

sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namesequential/dense/kernel

+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes

:*
dtype0

sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:*
dtype0

sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namesequential/dense_1/kernel

-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

: *
dtype0

sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
: *
dtype0

sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namesequential/dense_2/kernel

-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes

: *
dtype0

sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:*
dtype0

sequential/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namesequential/dense_3/kernel

-sequential/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_3/kernel*
_output_shapes

:*
dtype0

sequential/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_3/bias

+sequential/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential/dense_3/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0

training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
ª
'training/Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'training/Adam/sequential/dense/kernel/m
£
;training/Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense/kernel/m*
_output_shapes

:*
dtype0
¢
%training/Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%training/Adam/sequential/dense/bias/m

9training/Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOp%training/Adam/sequential/dense/bias/m*
_output_shapes
:*
dtype0
®
)training/Adam/sequential/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)training/Adam/sequential/dense_1/kernel/m
§
=training/Adam/sequential/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_1/kernel/m*
_output_shapes

: *
dtype0
¦
'training/Adam/sequential/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'training/Adam/sequential/dense_1/bias/m

;training/Adam/sequential/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_1/bias/m*
_output_shapes
: *
dtype0
®
)training/Adam/sequential/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)training/Adam/sequential/dense_2/kernel/m
§
=training/Adam/sequential/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_2/kernel/m*
_output_shapes

: *
dtype0
¦
'training/Adam/sequential/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_2/bias/m

;training/Adam/sequential/dense_2/bias/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_2/bias/m*
_output_shapes
:*
dtype0
®
)training/Adam/sequential/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)training/Adam/sequential/dense_3/kernel/m
§
=training/Adam/sequential/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_3/kernel/m*
_output_shapes

:*
dtype0
¦
'training/Adam/sequential/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_3/bias/m

;training/Adam/sequential/dense_3/bias/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_3/bias/m*
_output_shapes
:*
dtype0
ª
'training/Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'training/Adam/sequential/dense/kernel/v
£
;training/Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense/kernel/v*
_output_shapes

:*
dtype0
¢
%training/Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%training/Adam/sequential/dense/bias/v

9training/Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOp%training/Adam/sequential/dense/bias/v*
_output_shapes
:*
dtype0
®
)training/Adam/sequential/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)training/Adam/sequential/dense_1/kernel/v
§
=training/Adam/sequential/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_1/kernel/v*
_output_shapes

: *
dtype0
¦
'training/Adam/sequential/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'training/Adam/sequential/dense_1/bias/v

;training/Adam/sequential/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_1/bias/v*
_output_shapes
: *
dtype0
®
)training/Adam/sequential/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)training/Adam/sequential/dense_2/kernel/v
§
=training/Adam/sequential/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_2/kernel/v*
_output_shapes

: *
dtype0
¦
'training/Adam/sequential/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_2/bias/v

;training/Adam/sequential/dense_2/bias/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_2/bias/v*
_output_shapes
:*
dtype0
®
)training/Adam/sequential/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)training/Adam/sequential/dense_3/kernel/v
§
=training/Adam/sequential/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_3/kernel/v*
_output_shapes

:*
dtype0
¦
'training/Adam/sequential/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_3/bias/v

;training/Adam/sequential/dense_3/bias/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
©-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ä,
valueÚ,B×, BÐ,

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
Ð
#iter

$beta_1

%beta_2
	&decay
'learning_ratemAmBmCmDmEmFmGmHvIvJvKvLvMvNvOvP
8
0
1
2
3
4
5
6
7
 
8
0
1
2
3
4
5
6
7
­
(layer_regularization_losses
)metrics
trainable_variables

*layers
regularization_losses
+non_trainable_variables
	variables
,layer_metrics
 
ca
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
-layer_regularization_losses
.metrics
trainable_variables

/layers
regularization_losses
0non_trainable_variables
	variables
1layer_metrics
ec
VARIABLE_VALUEsequential/dense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
2layer_regularization_losses
3metrics
trainable_variables

4layers
regularization_losses
5non_trainable_variables
	variables
6layer_metrics
ec
VARIABLE_VALUEsequential/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
7layer_regularization_losses
8metrics
trainable_variables

9layers
regularization_losses
:non_trainable_variables
	variables
;layer_metrics
ec
VARIABLE_VALUEsequential/dense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential/dense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
<layer_regularization_losses
=metrics
trainable_variables

>layers
 regularization_losses
?non_trainable_variables
!	variables
@layer_metrics
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

VARIABLE_VALUE'training/Adam/sequential/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%training/Adam/sequential/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%training/Adam/sequential/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE)training/Adam/sequential/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'training/Adam/sequential/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_signature_wrapper_661
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOp-sequential/dense_3/kernel/Read/ReadVariableOp+sequential/dense_3/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp;training/Adam/sequential/dense/kernel/m/Read/ReadVariableOp9training/Adam/sequential/dense/bias/m/Read/ReadVariableOp=training/Adam/sequential/dense_1/kernel/m/Read/ReadVariableOp;training/Adam/sequential/dense_1/bias/m/Read/ReadVariableOp=training/Adam/sequential/dense_2/kernel/m/Read/ReadVariableOp;training/Adam/sequential/dense_2/bias/m/Read/ReadVariableOp=training/Adam/sequential/dense_3/kernel/m/Read/ReadVariableOp;training/Adam/sequential/dense_3/bias/m/Read/ReadVariableOp;training/Adam/sequential/dense/kernel/v/Read/ReadVariableOp9training/Adam/sequential/dense/bias/v/Read/ReadVariableOp=training/Adam/sequential/dense_1/kernel/v/Read/ReadVariableOp;training/Adam/sequential/dense_1/bias/v/Read/ReadVariableOp=training/Adam/sequential/dense_2/kernel/v/Read/ReadVariableOp;training/Adam/sequential/dense_2/bias/v/Read/ReadVariableOp=training/Adam/sequential/dense_3/kernel/v/Read/ReadVariableOp;training/Adam/sequential/dense_3/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_1022


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_rate'training/Adam/sequential/dense/kernel/m%training/Adam/sequential/dense/bias/m)training/Adam/sequential/dense_1/kernel/m'training/Adam/sequential/dense_1/bias/m)training/Adam/sequential/dense_2/kernel/m'training/Adam/sequential/dense_2/bias/m)training/Adam/sequential/dense_3/kernel/m'training/Adam/sequential/dense_3/bias/m'training/Adam/sequential/dense/kernel/v%training/Adam/sequential/dense/bias/v)training/Adam/sequential/dense_1/kernel/v'training/Adam/sequential/dense_1/bias/v)training/Adam/sequential/dense_2/kernel/v'training/Adam/sequential/dense_2/bias/v)training/Adam/sequential/dense_3/kernel/v'training/Adam/sequential/dense_3/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_1119µ
ø1
ª
__inference__wrapped_model_368
input_1P
>sequential_dense_matmul_readvariableop_sequential_dense_kernel:K
=sequential_dense_biasadd_readvariableop_sequential_dense_bias:T
Bsequential_dense_1_matmul_readvariableop_sequential_dense_1_kernel: O
Asequential_dense_1_biasadd_readvariableop_sequential_dense_1_bias: T
Bsequential_dense_2_matmul_readvariableop_sequential_dense_2_kernel: O
Asequential_dense_2_biasadd_readvariableop_sequential_dense_2_bias:T
Bsequential_dense_3_matmul_readvariableop_sequential_dense_3_kernel:O
Asequential_dense_3_biasadd_readvariableop_sequential_dense_3_bias:
identity¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp¢)sequential/dense_1/BiasAdd/ReadVariableOp¢(sequential/dense_1/MatMul/ReadVariableOp¢)sequential/dense_2/BiasAdd/ReadVariableOp¢(sequential/dense_2/MatMul/ReadVariableOp¢)sequential/dense_3/BiasAdd/ReadVariableOp¢(sequential/dense_3/MatMul/ReadVariableOp
sequential/dense/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/CastÏ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp>sequential_dense_matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02(
&sequential/dense/MatMul/ReadVariableOp¹
sequential/dense/MatMulMatMulsequential/dense/Cast:y:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/MatMulÌ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=sequential_dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense/Relu×
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOpBsequential_dense_1_matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpÉ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dense_1/MatMulÔ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÍ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dense_1/BiasAdd
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/dense_1/Relu×
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOpBsequential_dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpË
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_2/MatMulÔ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOpAsequential_dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpÍ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_2/BiasAdd
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_2/Relu×
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOpBsequential_dense_3_matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02*
(sequential/dense_3/MatMul/ReadVariableOpË
sequential/dense_3/MatMulMatMul%sequential/dense_2/Relu:activations:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_3/MatMulÔ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOpAsequential_dense_3_biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOpÍ
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/dense_3/BiasAdd~
IdentityIdentity#sequential/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¦
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ï
ð
C__inference_sequential_layer_call_and_return_conditional_losses_436

inputs/
dense_sequential_dense_kernel:)
dense_sequential_dense_bias:3
!dense_1_sequential_dense_1_kernel: -
dense_1_sequential_dense_1_bias: 3
!dense_2_sequential_dense_2_kernel: -
dense_2_sequential_dense_2_bias:3
!dense_3_sequential_dense_3_kernel:-
dense_3_sequential_dense_3_bias:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCalli

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast­
dense/StatefulPartitionedCallStatefulPartitionedCalldense/Cast:y:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3872
dense/StatefulPartitionedCallÓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_4022!
dense_1/StatefulPartitionedCallÕ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_4172!
dense_2/StatefulPartitionedCallÕ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0!dense_3_sequential_dense_3_kerneldense_3_sequential_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_4312!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²
%__inference_dense_2_layer_call_fn_884

inputs+
sequential_dense_2_kernel: %
sequential_dense_2_bias:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_2_kernelsequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_4172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


@__inference_dense_2_layer_call_and_return_conditional_losses_895

inputsA
/matmul_readvariableop_sequential_dense_2_kernel: <
.biasadd_readvariableop_sequential_dense_2_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³
®
(__inference_sequential_layer_call_fn_674
input_1)
sequential_dense_kernel:#
sequential_dense_bias:+
sequential_dense_1_kernel: %
sequential_dense_1_bias: +
sequential_dense_2_kernel: %
sequential_dense_2_bias:+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_4362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ò(

C__inference_sequential_layer_call_and_return_conditional_losses_777

inputsE
3dense_matmul_readvariableop_sequential_dense_kernel:@
2dense_biasadd_readvariableop_sequential_dense_bias:I
7dense_1_matmul_readvariableop_sequential_dense_1_kernel: D
6dense_1_biasadd_readvariableop_sequential_dense_1_bias: I
7dense_2_matmul_readvariableop_sequential_dense_2_kernel: D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:I
7dense_3_matmul_readvariableop_sequential_dense_3_kernel:D
6dense_3_biasadd_readvariableop_sequential_dense_3_bias:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpi

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast®
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¶
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul³
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¶
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul³
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÎ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ(

C__inference_sequential_layer_call_and_return_conditional_losses_809
input_1E
3dense_matmul_readvariableop_sequential_dense_kernel:@
2dense_biasadd_readvariableop_sequential_dense_bias:I
7dense_1_matmul_readvariableop_sequential_dense_1_kernel: D
6dense_1_biasadd_readvariableop_sequential_dense_1_bias: I
7dense_2_matmul_readvariableop_sequential_dense_2_kernel: D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:I
7dense_3_matmul_readvariableop_sequential_dense_3_kernel:D
6dense_3_biasadd_readvariableop_sequential_dense_3_bias:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpj

dense/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast®
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¶
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul³
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¶
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul³
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÎ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
â


@__inference_dense_3_layer_call_and_return_conditional_losses_431

inputsA
/matmul_readvariableop_sequential_dense_3_kernel:<
.biasadd_readvariableop_sequential_dense_3_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


@__inference_dense_1_layer_call_and_return_conditional_losses_877

inputsA
/matmul_readvariableop_sequential_dense_1_kernel: <
.biasadd_readvariableop_sequential_dense_1_bias: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
ð
C__inference_sequential_layer_call_and_return_conditional_losses_554

inputs/
dense_sequential_dense_kernel:)
dense_sequential_dense_bias:3
!dense_1_sequential_dense_1_kernel: -
dense_1_sequential_dense_1_bias: 3
!dense_2_sequential_dense_2_kernel: -
dense_2_sequential_dense_2_bias:3
!dense_3_sequential_dense_3_kernel:-
dense_3_sequential_dense_3_bias:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCalli

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast­
dense/StatefulPartitionedCallStatefulPartitionedCalldense/Cast:y:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3872
dense/StatefulPartitionedCallÓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_4022!
dense_1/StatefulPartitionedCallÕ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_4172!
dense_2/StatefulPartitionedCallÕ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0!dense_3_sequential_dense_3_kerneldense_3_sequential_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_4312!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

@__inference_dense_1_layer_call_and_return_conditional_losses_402

inputsA
/matmul_readvariableop_sequential_dense_1_kernel: <
.biasadd_readvariableop_sequential_dense_1_bias: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

§
!__inference_signature_wrapper_661
input_1)
sequential_dense_kernel:#
sequential_dense_bias:+
sequential_dense_1_kernel: %
sequential_dense_1_bias: +
sequential_dense_2_kernel: %
sequential_dense_2_bias:+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__wrapped_model_3682
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

²
%__inference_dense_1_layer_call_fn_866

inputs+
sequential_dense_1_kernel: %
sequential_dense_1_bias: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_1_kernelsequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_4022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


>__inference_dense_layer_call_and_return_conditional_losses_859

inputs?
-matmul_readvariableop_sequential_dense_kernel::
,biasadd_readvariableop_sequential_dense_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò(

C__inference_sequential_layer_call_and_return_conditional_losses_745

inputsE
3dense_matmul_readvariableop_sequential_dense_kernel:@
2dense_biasadd_readvariableop_sequential_dense_bias:I
7dense_1_matmul_readvariableop_sequential_dense_1_kernel: D
6dense_1_biasadd_readvariableop_sequential_dense_1_bias: I
7dense_2_matmul_readvariableop_sequential_dense_2_kernel: D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:I
7dense_3_matmul_readvariableop_sequential_dense_3_kernel:D
6dense_3_biasadd_readvariableop_sequential_dense_3_bias:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpi

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast®
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¶
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul³
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¶
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul³
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÎ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
®
(__inference_sequential_layer_call_fn_713
input_1)
sequential_dense_kernel:#
sequential_dense_bias:+
sequential_dense_1_kernel: %
sequential_dense_1_bias: +
sequential_dense_2_kernel: %
sequential_dense_2_bias:+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_5542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
õ(

C__inference_sequential_layer_call_and_return_conditional_losses_841
input_1E
3dense_matmul_readvariableop_sequential_dense_kernel:@
2dense_biasadd_readvariableop_sequential_dense_bias:I
7dense_1_matmul_readvariableop_sequential_dense_1_kernel: D
6dense_1_biasadd_readvariableop_sequential_dense_1_bias: I
7dense_2_matmul_readvariableop_sequential_dense_2_kernel: D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:I
7dense_3_matmul_readvariableop_sequential_dense_3_kernel:D
6dense_3_biasadd_readvariableop_sequential_dense_3_bias:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢dense_3/MatMul/ReadVariableOpj

dense/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Cast®
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul«
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu¶
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/MatMul³
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_1/Relu¶
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul³
dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu¶
dense_3/MatMul/ReadVariableOpReadVariableOp7dense_3_matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul³
dense_3/BiasAdd/ReadVariableOpReadVariableOp6dense_3_biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¡
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÎ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ÓI
Ã
__inference__traced_save_1022
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop8
4savev2_sequential_dense_3_kernel_read_readvariableop6
2savev2_sequential_dense_3_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableopF
Bsavev2_training_adam_sequential_dense_kernel_m_read_readvariableopD
@savev2_training_adam_sequential_dense_bias_m_read_readvariableopH
Dsavev2_training_adam_sequential_dense_1_kernel_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_1_bias_m_read_readvariableopH
Dsavev2_training_adam_sequential_dense_2_kernel_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_2_bias_m_read_readvariableopH
Dsavev2_training_adam_sequential_dense_3_kernel_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_3_bias_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_kernel_v_read_readvariableopD
@savev2_training_adam_sequential_dense_bias_v_read_readvariableopH
Dsavev2_training_adam_sequential_dense_1_kernel_v_read_readvariableopF
Bsavev2_training_adam_sequential_dense_1_bias_v_read_readvariableopH
Dsavev2_training_adam_sequential_dense_2_kernel_v_read_readvariableopF
Bsavev2_training_adam_sequential_dense_2_bias_v_read_readvariableopH
Dsavev2_training_adam_sequential_dense_3_kernel_v_read_readvariableopF
Bsavev2_training_adam_sequential_dense_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameî
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueöBóB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÄ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableop4savev2_sequential_dense_3_kernel_read_readvariableop2savev2_sequential_dense_3_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableopBsavev2_training_adam_sequential_dense_kernel_m_read_readvariableop@savev2_training_adam_sequential_dense_bias_m_read_readvariableopDsavev2_training_adam_sequential_dense_1_kernel_m_read_readvariableopBsavev2_training_adam_sequential_dense_1_bias_m_read_readvariableopDsavev2_training_adam_sequential_dense_2_kernel_m_read_readvariableopBsavev2_training_adam_sequential_dense_2_bias_m_read_readvariableopDsavev2_training_adam_sequential_dense_3_kernel_m_read_readvariableopBsavev2_training_adam_sequential_dense_3_bias_m_read_readvariableopBsavev2_training_adam_sequential_dense_kernel_v_read_readvariableop@savev2_training_adam_sequential_dense_bias_v_read_readvariableopDsavev2_training_adam_sequential_dense_1_kernel_v_read_readvariableopBsavev2_training_adam_sequential_dense_1_bias_v_read_readvariableopDsavev2_training_adam_sequential_dense_2_kernel_v_read_readvariableopBsavev2_training_adam_sequential_dense_2_bias_v_read_readvariableopDsavev2_training_adam_sequential_dense_3_kernel_v_read_readvariableopBsavev2_training_adam_sequential_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*ã
_input_shapesÑ
Î: ::: : : :::: : : : : ::: : : :::::: : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
¾

@__inference_dense_2_layer_call_and_return_conditional_losses_417

inputsA
/matmul_readvariableop_sequential_dense_2_kernel: <
.biasadd_readvariableop_sequential_dense_2_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
°
­
(__inference_sequential_layer_call_fn_687

inputs)
sequential_dense_kernel:#
sequential_dense_bias:+
sequential_dense_1_kernel: %
sequential_dense_1_bias: +
sequential_dense_2_kernel: %
sequential_dense_2_bias:+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_4362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

²
%__inference_dense_3_layer_call_fn_902

inputs+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_3_kernelsequential_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_3_layer_call_and_return_conditional_losses_4312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸


@__inference_dense_3_layer_call_and_return_conditional_losses_912

inputsA
/matmul_readvariableop_sequential_dense_3_kernel:<
.biasadd_readvariableop_sequential_dense_3_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_3_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
¬
#__inference_dense_layer_call_fn_848

inputs)
sequential_dense_kernel:#
sequential_dense_bias:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_3872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
­
(__inference_sequential_layer_call_fn_700

inputs)
sequential_dense_kernel:#
sequential_dense_bias:+
sequential_dense_1_kernel: %
sequential_dense_1_bias: +
sequential_dense_2_kernel: %
sequential_dense_2_bias:+
sequential_dense_3_kernel:%
sequential_dense_3_bias:
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_5542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

>__inference_dense_layer_call_and_return_conditional_losses_387

inputs?
-matmul_readvariableop_sequential_dense_kernel::
,biasadd_readvariableop_sequential_dense_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_sequential_dense_kernel*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_sequential_dense_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
ù
 __inference__traced_restore_1119
file_prefix:
(assignvariableop_sequential_dense_kernel:6
(assignvariableop_1_sequential_dense_bias:>
,assignvariableop_2_sequential_dense_1_kernel: 8
*assignvariableop_3_sequential_dense_1_bias: >
,assignvariableop_4_sequential_dense_2_kernel: 8
*assignvariableop_5_sequential_dense_2_bias:>
,assignvariableop_6_sequential_dense_3_kernel:8
*assignvariableop_7_sequential_dense_3_bias:/
%assignvariableop_8_training_adam_iter:	 1
'assignvariableop_9_training_adam_beta_1: 2
(assignvariableop_10_training_adam_beta_2: 1
'assignvariableop_11_training_adam_decay: 9
/assignvariableop_12_training_adam_learning_rate: M
;assignvariableop_13_training_adam_sequential_dense_kernel_m:G
9assignvariableop_14_training_adam_sequential_dense_bias_m:O
=assignvariableop_15_training_adam_sequential_dense_1_kernel_m: I
;assignvariableop_16_training_adam_sequential_dense_1_bias_m: O
=assignvariableop_17_training_adam_sequential_dense_2_kernel_m: I
;assignvariableop_18_training_adam_sequential_dense_2_bias_m:O
=assignvariableop_19_training_adam_sequential_dense_3_kernel_m:I
;assignvariableop_20_training_adam_sequential_dense_3_bias_m:M
;assignvariableop_21_training_adam_sequential_dense_kernel_v:G
9assignvariableop_22_training_adam_sequential_dense_bias_v:O
=assignvariableop_23_training_adam_sequential_dense_1_kernel_v: I
;assignvariableop_24_training_adam_sequential_dense_1_bias_v: O
=assignvariableop_25_training_adam_sequential_dense_2_kernel_v: I
;assignvariableop_26_training_adam_sequential_dense_2_bias_v:O
=assignvariableop_27_training_adam_sequential_dense_3_kernel_v:I
;assignvariableop_28_training_adam_sequential_dense_3_bias_v:
identity_30¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueöBóB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÂ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity§
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1­
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2±
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¯
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¯
AssignVariableOp_5AssignVariableOp*assignvariableop_5_sequential_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6±
AssignVariableOp_6AssignVariableOp,assignvariableop_6_sequential_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¯
AssignVariableOp_7AssignVariableOp*assignvariableop_7_sequential_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_training_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¬
AssignVariableOp_9AssignVariableOp'assignvariableop_9_training_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10°
AssignVariableOp_10AssignVariableOp(assignvariableop_10_training_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¯
AssignVariableOp_11AssignVariableOp'assignvariableop_11_training_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12·
AssignVariableOp_12AssignVariableOp/assignvariableop_12_training_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ã
AssignVariableOp_13AssignVariableOp;assignvariableop_13_training_adam_sequential_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_training_adam_sequential_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Å
AssignVariableOp_15AssignVariableOp=assignvariableop_15_training_adam_sequential_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ã
AssignVariableOp_16AssignVariableOp;assignvariableop_16_training_adam_sequential_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Å
AssignVariableOp_17AssignVariableOp=assignvariableop_17_training_adam_sequential_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ã
AssignVariableOp_18AssignVariableOp;assignvariableop_18_training_adam_sequential_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Å
AssignVariableOp_19AssignVariableOp=assignvariableop_19_training_adam_sequential_dense_3_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ã
AssignVariableOp_20AssignVariableOp;assignvariableop_20_training_adam_sequential_dense_3_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ã
AssignVariableOp_21AssignVariableOp;assignvariableop_21_training_adam_sequential_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Á
AssignVariableOp_22AssignVariableOp9assignvariableop_22_training_adam_sequential_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Å
AssignVariableOp_23AssignVariableOp=assignvariableop_23_training_adam_sequential_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ã
AssignVariableOp_24AssignVariableOp;assignvariableop_24_training_adam_sequential_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Å
AssignVariableOp_25AssignVariableOp=assignvariableop_25_training_adam_sequential_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ã
AssignVariableOp_26AssignVariableOp;assignvariableop_26_training_adam_sequential_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Å
AssignVariableOp_27AssignVariableOp=assignvariableop_27_training_adam_sequential_dense_3_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ã
AssignVariableOp_28AssignVariableOp;assignvariableop_28_training_adam_sequential_dense_3_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÜ
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29f
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_30Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
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
_user_specified_namefile_prefix"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ý[

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
Q_default_save_signature
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_sequential
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
ã
#iter

$beta_1

%beta_2
	&decay
'learning_ratemAmBmCmDmEmFmGmHvIvJvKvLvMvNvOvP"
	optimizer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
Ê
(layer_regularization_losses
)metrics
trainable_variables

*layers
regularization_losses
+non_trainable_variables
	variables
,layer_metrics
R__call__
Q_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
\serving_default"
signature_map
):'2sequential/dense/kernel
#:!2sequential/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
-layer_regularization_losses
.metrics
trainable_variables

/layers
regularization_losses
0non_trainable_variables
	variables
1layer_metrics
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
+:) 2sequential/dense_1/kernel
%:# 2sequential/dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
2layer_regularization_losses
3metrics
trainable_variables

4layers
regularization_losses
5non_trainable_variables
	variables
6layer_metrics
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
+:) 2sequential/dense_2/kernel
%:#2sequential/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
7layer_regularization_losses
8metrics
trainable_variables

9layers
regularization_losses
:non_trainable_variables
	variables
;layer_metrics
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
+:)2sequential/dense_3/kernel
%:#2sequential/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
<layer_regularization_losses
=metrics
trainable_variables

>layers
 regularization_losses
?non_trainable_variables
!	variables
@layer_metrics
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
7:52'training/Adam/sequential/dense/kernel/m
1:/2%training/Adam/sequential/dense/bias/m
9:7 2)training/Adam/sequential/dense_1/kernel/m
3:1 2'training/Adam/sequential/dense_1/bias/m
9:7 2)training/Adam/sequential/dense_2/kernel/m
3:12'training/Adam/sequential/dense_2/bias/m
9:72)training/Adam/sequential/dense_3/kernel/m
3:12'training/Adam/sequential/dense_3/bias/m
7:52'training/Adam/sequential/dense/kernel/v
1:/2%training/Adam/sequential/dense/bias/v
9:7 2)training/Adam/sequential/dense_1/kernel/v
3:1 2'training/Adam/sequential/dense_1/bias/v
9:7 2)training/Adam/sequential/dense_2/kernel/v
3:12'training/Adam/sequential/dense_2/bias/v
9:72)training/Adam/sequential/dense_3/kernel/v
3:12'training/Adam/sequential/dense_3/bias/v
ÉBÆ
__inference__wrapped_model_368input_1"
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
î2ë
(__inference_sequential_layer_call_fn_674
(__inference_sequential_layer_call_fn_687
(__inference_sequential_layer_call_fn_700
(__inference_sequential_layer_call_fn_713À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_sequential_layer_call_and_return_conditional_losses_745
C__inference_sequential_layer_call_and_return_conditional_losses_777
C__inference_sequential_layer_call_and_return_conditional_losses_809
C__inference_sequential_layer_call_and_return_conditional_losses_841À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Í2Ê
#__inference_dense_layer_call_fn_848¢
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
è2å
>__inference_dense_layer_call_and_return_conditional_losses_859¢
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
Ï2Ì
%__inference_dense_1_layer_call_fn_866¢
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
ê2ç
@__inference_dense_1_layer_call_and_return_conditional_losses_877¢
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
Ï2Ì
%__inference_dense_2_layer_call_fn_884¢
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
ê2ç
@__inference_dense_2_layer_call_and_return_conditional_losses_895¢
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
Ï2Ì
%__inference_dense_3_layer_call_fn_902¢
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
ê2ç
@__inference_dense_3_layer_call_and_return_conditional_losses_912¢
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
ÈBÅ
!__inference_signature_wrapper_661input_1"
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
 
__inference__wrapped_model_368q0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_1_layer_call_and_return_conditional_losses_877\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 x
%__inference_dense_1_layer_call_fn_866O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ  
@__inference_dense_2_layer_call_and_return_conditional_losses_895\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_2_layer_call_fn_884O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_3_layer_call_and_return_conditional_losses_912\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_3_layer_call_fn_902O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
>__inference_dense_layer_call_and_return_conditional_losses_859\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
#__inference_dense_layer_call_fn_848O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ±
C__inference_sequential_layer_call_and_return_conditional_losses_745j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
C__inference_sequential_layer_call_and_return_conditional_losses_777j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
C__inference_sequential_layer_call_and_return_conditional_losses_809k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
C__inference_sequential_layer_call_and_return_conditional_losses_841k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_sequential_layer_call_fn_674^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_sequential_layer_call_fn_687]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_sequential_layer_call_fn_700]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_sequential_layer_call_fn_713^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¡
!__inference_signature_wrapper_661|;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ