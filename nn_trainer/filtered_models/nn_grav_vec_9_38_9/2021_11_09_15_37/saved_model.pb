
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
 "serve*2.6.12v2.6.0-101-g3aa40c3ce9d8°

sequential_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_namesequential_1/dense_4/kernel

/sequential_1/dense_4/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_4/kernel*
_output_shapes

:	*
dtype0

sequential_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namesequential_1/dense_4/bias

-sequential_1/dense_4/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_4/bias*
_output_shapes
:	*
dtype0

sequential_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	&*,
shared_namesequential_1/dense_5/kernel

/sequential_1/dense_5/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_5/kernel*
_output_shapes

:	&*
dtype0

sequential_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namesequential_1/dense_5/bias

-sequential_1/dense_5/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_5/bias*
_output_shapes
:&*
dtype0

sequential_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&	*,
shared_namesequential_1/dense_6/kernel

/sequential_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_6/kernel*
_output_shapes

:&	*
dtype0

sequential_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namesequential_1/dense_6/bias

-sequential_1/dense_6/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_6/bias*
_output_shapes
:	*
dtype0

sequential_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_namesequential_1/dense_7/kernel

/sequential_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_7/kernel*
_output_shapes

:	*
dtype0

sequential_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_1/dense_7/bias

-sequential_1/dense_7/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_7/bias*
_output_shapes
:*
dtype0
|
training_2/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_2/Adam/iter
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
_output_shapes
: *
dtype0	

training_2/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_1
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
_output_shapes
: *
dtype0

training_2/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_2/Adam/beta_2
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_2/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_2/Adam/decay
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
_output_shapes
: *
dtype0

training_2/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_2/Adam/learning_rate

1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
_output_shapes
: *
dtype0
¶
-training_2/Adam/sequential_1/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*>
shared_name/-training_2/Adam/sequential_1/dense_4/kernel/m
¯
Atraining_2/Adam/sequential_1/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_4/kernel/m*
_output_shapes

:	*
dtype0
®
+training_2/Adam/sequential_1/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+training_2/Adam/sequential_1/dense_4/bias/m
§
?training_2/Adam/sequential_1/dense_4/bias/m/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_4/bias/m*
_output_shapes
:	*
dtype0
¶
-training_2/Adam/sequential_1/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	&*>
shared_name/-training_2/Adam/sequential_1/dense_5/kernel/m
¯
Atraining_2/Adam/sequential_1/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_5/kernel/m*
_output_shapes

:	&*
dtype0
®
+training_2/Adam/sequential_1/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*<
shared_name-+training_2/Adam/sequential_1/dense_5/bias/m
§
?training_2/Adam/sequential_1/dense_5/bias/m/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_5/bias/m*
_output_shapes
:&*
dtype0
¶
-training_2/Adam/sequential_1/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&	*>
shared_name/-training_2/Adam/sequential_1/dense_6/kernel/m
¯
Atraining_2/Adam/sequential_1/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_6/kernel/m*
_output_shapes

:&	*
dtype0
®
+training_2/Adam/sequential_1/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+training_2/Adam/sequential_1/dense_6/bias/m
§
?training_2/Adam/sequential_1/dense_6/bias/m/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_6/bias/m*
_output_shapes
:	*
dtype0
¶
-training_2/Adam/sequential_1/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*>
shared_name/-training_2/Adam/sequential_1/dense_7/kernel/m
¯
Atraining_2/Adam/sequential_1/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_7/kernel/m*
_output_shapes

:	*
dtype0
®
+training_2/Adam/sequential_1/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+training_2/Adam/sequential_1/dense_7/bias/m
§
?training_2/Adam/sequential_1/dense_7/bias/m/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_7/bias/m*
_output_shapes
:*
dtype0
¶
-training_2/Adam/sequential_1/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*>
shared_name/-training_2/Adam/sequential_1/dense_4/kernel/v
¯
Atraining_2/Adam/sequential_1/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_4/kernel/v*
_output_shapes

:	*
dtype0
®
+training_2/Adam/sequential_1/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+training_2/Adam/sequential_1/dense_4/bias/v
§
?training_2/Adam/sequential_1/dense_4/bias/v/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_4/bias/v*
_output_shapes
:	*
dtype0
¶
-training_2/Adam/sequential_1/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	&*>
shared_name/-training_2/Adam/sequential_1/dense_5/kernel/v
¯
Atraining_2/Adam/sequential_1/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_5/kernel/v*
_output_shapes

:	&*
dtype0
®
+training_2/Adam/sequential_1/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*<
shared_name-+training_2/Adam/sequential_1/dense_5/bias/v
§
?training_2/Adam/sequential_1/dense_5/bias/v/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_5/bias/v*
_output_shapes
:&*
dtype0
¶
-training_2/Adam/sequential_1/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&	*>
shared_name/-training_2/Adam/sequential_1/dense_6/kernel/v
¯
Atraining_2/Adam/sequential_1/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_6/kernel/v*
_output_shapes

:&	*
dtype0
®
+training_2/Adam/sequential_1/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*<
shared_name-+training_2/Adam/sequential_1/dense_6/bias/v
§
?training_2/Adam/sequential_1/dense_6/bias/v/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_6/bias/v*
_output_shapes
:	*
dtype0
¶
-training_2/Adam/sequential_1/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*>
shared_name/-training_2/Adam/sequential_1/dense_7/kernel/v
¯
Atraining_2/Adam/sequential_1/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp-training_2/Adam/sequential_1/dense_7/kernel/v*
_output_shapes

:	*
dtype0
®
+training_2/Adam/sequential_1/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+training_2/Adam/sequential_1/dense_7/bias/v
§
?training_2/Adam/sequential_1/dense_7/bias/v/Read/ReadVariableOpReadVariableOp+training_2/Adam/sequential_1/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ê-
valueÀ-B½- B¶-
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
ge
VARIABLE_VALUEsequential_1/dense_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential_1/dense_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEsequential_1/dense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential_1/dense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEsequential_1/dense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential_1/dense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ge
VARIABLE_VALUEsequential_1/dense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential_1/dense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
SQ
VARIABLE_VALUEtraining_2/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_2/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_2/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
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

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-training_2/Adam/sequential_1/dense_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+training_2/Adam/sequential_1/dense_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
£
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential_1/dense_4/kernelsequential_1/dense_4/biassequential_1/dense_5/kernelsequential_1/dense_5/biassequential_1/dense_6/kernelsequential_1/dense_6/biassequential_1/dense_7/kernelsequential_1/dense_7/bias*
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
GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_1792
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¹
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/sequential_1/dense_4/kernel/Read/ReadVariableOp-sequential_1/dense_4/bias/Read/ReadVariableOp/sequential_1/dense_5/kernel/Read/ReadVariableOp-sequential_1/dense_5/bias/Read/ReadVariableOp/sequential_1/dense_6/kernel/Read/ReadVariableOp-sequential_1/dense_6/bias/Read/ReadVariableOp/sequential_1/dense_7/kernel/Read/ReadVariableOp-sequential_1/dense_7/bias/Read/ReadVariableOp(training_2/Adam/iter/Read/ReadVariableOp*training_2/Adam/beta_1/Read/ReadVariableOp*training_2/Adam/beta_2/Read/ReadVariableOp)training_2/Adam/decay/Read/ReadVariableOp1training_2/Adam/learning_rate/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_4/kernel/m/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_4/bias/m/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_5/kernel/m/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_5/bias/m/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_6/kernel/m/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_6/bias/m/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_7/kernel/m/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_7/bias/m/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_4/kernel/v/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_4/bias/v/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_5/kernel/v/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_5/bias/v/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_6/kernel/v/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_6/bias/v/Read/ReadVariableOpAtraining_2/Adam/sequential_1/dense_7/kernel/v/Read/ReadVariableOp?training_2/Adam/sequential_1/dense_7/bias/v/Read/ReadVariableOpConst**
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
__inference__traced_save_2153
ð

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_1/dense_4/kernelsequential_1/dense_4/biassequential_1/dense_5/kernelsequential_1/dense_5/biassequential_1/dense_6/kernelsequential_1/dense_6/biassequential_1/dense_7/kernelsequential_1/dense_7/biastraining_2/Adam/itertraining_2/Adam/beta_1training_2/Adam/beta_2training_2/Adam/decaytraining_2/Adam/learning_rate-training_2/Adam/sequential_1/dense_4/kernel/m+training_2/Adam/sequential_1/dense_4/bias/m-training_2/Adam/sequential_1/dense_5/kernel/m+training_2/Adam/sequential_1/dense_5/bias/m-training_2/Adam/sequential_1/dense_6/kernel/m+training_2/Adam/sequential_1/dense_6/bias/m-training_2/Adam/sequential_1/dense_7/kernel/m+training_2/Adam/sequential_1/dense_7/bias/m-training_2/Adam/sequential_1/dense_4/kernel/v+training_2/Adam/sequential_1/dense_4/bias/v-training_2/Adam/sequential_1/dense_5/kernel/v+training_2/Adam/sequential_1/dense_5/bias/v-training_2/Adam/sequential_1/dense_6/kernel/v+training_2/Adam/sequential_1/dense_6/bias/v-training_2/Adam/sequential_1/dense_7/kernel/v+training_2/Adam/sequential_1/dense_7/bias/v*)
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
 __inference__traced_restore_2250³
K
©
__inference__traced_save_2153
file_prefix:
6savev2_sequential_1_dense_4_kernel_read_readvariableop8
4savev2_sequential_1_dense_4_bias_read_readvariableop:
6savev2_sequential_1_dense_5_kernel_read_readvariableop8
4savev2_sequential_1_dense_5_bias_read_readvariableop:
6savev2_sequential_1_dense_6_kernel_read_readvariableop8
4savev2_sequential_1_dense_6_bias_read_readvariableop:
6savev2_sequential_1_dense_7_kernel_read_readvariableop8
4savev2_sequential_1_dense_7_bias_read_readvariableop3
/savev2_training_2_adam_iter_read_readvariableop	5
1savev2_training_2_adam_beta_1_read_readvariableop5
1savev2_training_2_adam_beta_2_read_readvariableop4
0savev2_training_2_adam_decay_read_readvariableop<
8savev2_training_2_adam_learning_rate_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_4_kernel_m_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_4_bias_m_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_5_kernel_m_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_5_bias_m_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_6_kernel_m_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_6_bias_m_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_7_kernel_m_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_7_bias_m_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_4_kernel_v_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_4_bias_v_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_5_kernel_v_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_5_bias_v_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_6_kernel_v_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_6_bias_v_read_readvariableopL
Hsavev2_training_2_adam_sequential_1_dense_7_kernel_v_read_readvariableopJ
Fsavev2_training_2_adam_sequential_1_dense_7_bias_v_read_readvariableop
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
SaveV2/shape_and_slices¢
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_sequential_1_dense_4_kernel_read_readvariableop4savev2_sequential_1_dense_4_bias_read_readvariableop6savev2_sequential_1_dense_5_kernel_read_readvariableop4savev2_sequential_1_dense_5_bias_read_readvariableop6savev2_sequential_1_dense_6_kernel_read_readvariableop4savev2_sequential_1_dense_6_bias_read_readvariableop6savev2_sequential_1_dense_7_kernel_read_readvariableop4savev2_sequential_1_dense_7_bias_read_readvariableop/savev2_training_2_adam_iter_read_readvariableop1savev2_training_2_adam_beta_1_read_readvariableop1savev2_training_2_adam_beta_2_read_readvariableop0savev2_training_2_adam_decay_read_readvariableop8savev2_training_2_adam_learning_rate_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_4_kernel_m_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_4_bias_m_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_5_kernel_m_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_5_bias_m_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_6_kernel_m_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_6_bias_m_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_7_kernel_m_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_7_bias_m_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_4_kernel_v_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_4_bias_v_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_5_kernel_v_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_5_bias_v_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_6_kernel_v_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_6_bias_v_read_readvariableopHsavev2_training_2_adam_sequential_1_dense_7_kernel_v_read_readvariableopFsavev2_training_2_adam_sequential_1_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Î: :	:	:	&:&:&	:	:	:: : : : : :	:	:	&:&:&	:	:	::	:	:	&:&:&	:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	&: 

_output_shapes
:&:$ 

_output_shapes

:&	: 

_output_shapes
:	:$ 

_output_shapes

:	: 
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

:	: 

_output_shapes
:	:$ 

_output_shapes

:	&: 

_output_shapes
:&:$ 

_output_shapes

:&	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	&: 

_output_shapes
:&:$ 

_output_shapes

:&	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: 


A__inference_dense_4_layer_call_and_return_conditional_losses_1990

inputsC
1matmul_readvariableop_sequential_1_dense_4_kernel:	>
0biasadd_readvariableop_sequential_1_dense_4_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

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


A__inference_dense_6_layer_call_and_return_conditional_losses_2026

inputsC
1matmul_readvariableop_sequential_1_dense_6_kernel:&	>
0biasadd_readvariableop_sequential_1_dense_6_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ&: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs
Ç

A__inference_dense_4_layer_call_and_return_conditional_losses_1518

inputsC
1matmul_readvariableop_sequential_1_dense_4_kernel:	>
0biasadd_readvariableop_sequential_1_dense_4_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

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
Ü)
¾
F__inference_sequential_1_layer_call_and_return_conditional_losses_1972
input_1K
9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel:	F
8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias:	K
9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel:	&F
8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias:&K
9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel:&	F
8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias:	K
9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel:	F
8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpn
dense_4/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Cast¸
dense_4/MatMul/ReadVariableOpReadVariableOp9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_4/Cast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/MatMulµ
dense_4/BiasAdd/ReadVariableOpReadVariableOp8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/Relu¸
dense_5/MatMul/ReadVariableOpReadVariableOp9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/MatMulµ
dense_5/BiasAdd/ReadVariableOpReadVariableOp8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/Relu¸
dense_6/MatMul/ReadVariableOpReadVariableOp9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/MatMulµ
dense_6/BiasAdd/ReadVariableOpReadVariableOp8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/Relu¸
dense_7/MatMul/ReadVariableOpReadVariableOp9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulµ
dense_7/BiasAdd/ReadVariableOpReadVariableOp8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdds
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¶4
ç
__inference__wrapped_model_1499
input_1X
Fsequential_1_dense_4_matmul_readvariableop_sequential_1_dense_4_kernel:	S
Esequential_1_dense_4_biasadd_readvariableop_sequential_1_dense_4_bias:	X
Fsequential_1_dense_5_matmul_readvariableop_sequential_1_dense_5_kernel:	&S
Esequential_1_dense_5_biasadd_readvariableop_sequential_1_dense_5_bias:&X
Fsequential_1_dense_6_matmul_readvariableop_sequential_1_dense_6_kernel:&	S
Esequential_1_dense_6_biasadd_readvariableop_sequential_1_dense_6_bias:	X
Fsequential_1_dense_7_matmul_readvariableop_sequential_1_dense_7_kernel:	S
Esequential_1_dense_7_biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢+sequential_1/dense_4/BiasAdd/ReadVariableOp¢*sequential_1/dense_4/MatMul/ReadVariableOp¢+sequential_1/dense_5/BiasAdd/ReadVariableOp¢*sequential_1/dense_5/MatMul/ReadVariableOp¢+sequential_1/dense_6/BiasAdd/ReadVariableOp¢*sequential_1/dense_6/MatMul/ReadVariableOp¢+sequential_1/dense_7/BiasAdd/ReadVariableOp¢*sequential_1/dense_7/MatMul/ReadVariableOp
sequential_1/dense_4/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_4/Castß
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOpFsequential_1_dense_4_matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOpÉ
sequential_1/dense_4/MatMulMatMulsequential_1/dense_4/Cast:y:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_4/MatMulÜ
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOpEsequential_1_dense_4_biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOpÕ
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_4/BiasAdd
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_4/Reluß
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOpFsequential_1_dense_5_matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOpÓ
sequential_1/dense_5/MatMulMatMul'sequential_1/dense_4/Relu:activations:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
sequential_1/dense_5/MatMulÜ
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOpEsequential_1_dense_5_biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOpÕ
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
sequential_1/dense_5/BiasAdd
sequential_1/dense_5/ReluRelu%sequential_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
sequential_1/dense_5/Reluß
*sequential_1/dense_6/MatMul/ReadVariableOpReadVariableOpFsequential_1_dense_6_matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02,
*sequential_1/dense_6/MatMul/ReadVariableOpÓ
sequential_1/dense_6/MatMulMatMul'sequential_1/dense_5/Relu:activations:02sequential_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_6/MatMulÜ
+sequential_1/dense_6/BiasAdd/ReadVariableOpReadVariableOpEsequential_1_dense_6_biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02-
+sequential_1/dense_6/BiasAdd/ReadVariableOpÕ
sequential_1/dense_6/BiasAddBiasAdd%sequential_1/dense_6/MatMul:product:03sequential_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_6/BiasAdd
sequential_1/dense_6/ReluRelu%sequential_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
sequential_1/dense_6/Reluß
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOpFsequential_1_dense_7_matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02,
*sequential_1/dense_7/MatMul/ReadVariableOpÓ
sequential_1/dense_7/MatMulMatMul'sequential_1/dense_6/Relu:activations:02sequential_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_7/MatMulÜ
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpEsequential_1_dense_7_biasadd_readvariableop_sequential_1_dense_7_bias*
_output_shapes
:*
dtype02-
+sequential_1/dense_7/BiasAdd/ReadVariableOpÕ
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_1/dense_7/BiasAdd
IdentityIdentity%sequential_1/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityº
NoOpNoOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp,^sequential_1/dense_6/BiasAdd/ReadVariableOp+^sequential_1/dense_6/MatMul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp2Z
+sequential_1/dense_6/BiasAdd/ReadVariableOp+sequential_1/dense_6/BiasAdd/ReadVariableOp2X
*sequential_1/dense_6/MatMul/ReadVariableOp*sequential_1/dense_6/MatMul/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ù)
½
F__inference_sequential_1_layer_call_and_return_conditional_losses_1908

inputsK
9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel:	F
8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias:	K
9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel:	&F
8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias:&K
9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel:&	F
8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias:	K
9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel:	F
8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpm
dense_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Cast¸
dense_4/MatMul/ReadVariableOpReadVariableOp9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_4/Cast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/MatMulµ
dense_4/BiasAdd/ReadVariableOpReadVariableOp8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/Relu¸
dense_5/MatMul/ReadVariableOpReadVariableOp9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/MatMulµ
dense_5/BiasAdd/ReadVariableOpReadVariableOp8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/Relu¸
dense_6/MatMul/ReadVariableOpReadVariableOp9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/MatMulµ
dense_6/BiasAdd/ReadVariableOpReadVariableOp8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/Relu¸
dense_7/MatMul/ReadVariableOpReadVariableOp9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulµ
dense_7/BiasAdd/ReadVariableOpReadVariableOp8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdds
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
¼
"__inference_signature_wrapper_1792
input_1-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_dense_4_kernelsequential_1_dense_4_biassequential_1_dense_5_kernelsequential_1_dense_5_biassequential_1_dense_6_kernelsequential_1_dense_6_biassequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *(
f#R!
__inference__wrapped_model_14992
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
ë


A__inference_dense_7_layer_call_and_return_conditional_losses_1562

inputsC
1matmul_readvariableop_sequential_1_dense_7_kernel:	>
0biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_7_bias*
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
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs


F__inference_sequential_1_layer_call_and_return_conditional_losses_1567

inputs5
#dense_4_sequential_1_dense_4_kernel:	/
!dense_4_sequential_1_dense_4_bias:	5
#dense_5_sequential_1_dense_5_kernel:	&/
!dense_5_sequential_1_dense_5_bias:&5
#dense_6_sequential_1_dense_6_kernel:&	/
!dense_6_sequential_1_dense_6_bias:	5
#dense_7_sequential_1_dense_7_kernel:	/
!dense_7_sequential_1_dense_7_bias:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCallm
dense_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/CastÂ
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4/Cast:y:0#dense_4_sequential_1_dense_4_kernel!dense_4_sequential_1_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_15182!
dense_4/StatefulPartitionedCallÚ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0#dense_5_sequential_1_dense_5_kernel!dense_5_sequential_1_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_15332!
dense_5/StatefulPartitionedCallÚ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0#dense_6_sequential_1_dense_6_kernel!dense_6_sequential_1_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_15482!
dense_6/StatefulPartitionedCallÚ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#dense_7_sequential_1_dense_7_kernel!dense_7_sequential_1_dense_7_bias*
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
GPU2*0J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_15622!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÖ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á


A__inference_dense_7_layer_call_and_return_conditional_losses_2043

inputsC
1matmul_readvariableop_sequential_1_dense_7_kernel:	>
0biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_7_bias*
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
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Þ
Ä
+__inference_sequential_1_layer_call_fn_1818

inputs-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_4_kernelsequential_1_dense_4_biassequential_1_dense_5_kernelsequential_1_dense_5_biassequential_1_dense_6_kernelsequential_1_dense_6_biassequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15672
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

·
&__inference_dense_6_layer_call_fn_2015

inputs-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_6_kernelsequential_1_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_15482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ&: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs


F__inference_sequential_1_layer_call_and_return_conditional_losses_1685

inputs5
#dense_4_sequential_1_dense_4_kernel:	/
!dense_4_sequential_1_dense_4_bias:	5
#dense_5_sequential_1_dense_5_kernel:	&/
!dense_5_sequential_1_dense_5_bias:&5
#dense_6_sequential_1_dense_6_kernel:&	/
!dense_6_sequential_1_dense_6_bias:	5
#dense_7_sequential_1_dense_7_kernel:	/
!dense_7_sequential_1_dense_7_bias:
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCallm
dense_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/CastÂ
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4/Cast:y:0#dense_4_sequential_1_dense_4_kernel!dense_4_sequential_1_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_15182!
dense_4/StatefulPartitionedCallÚ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0#dense_5_sequential_1_dense_5_kernel!dense_5_sequential_1_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_15332!
dense_5/StatefulPartitionedCallÚ
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0#dense_6_sequential_1_dense_6_kernel!dense_6_sequential_1_dense_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_15482!
dense_6/StatefulPartitionedCallÚ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0#dense_7_sequential_1_dense_7_kernel!dense_7_sequential_1_dense_7_bias*
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
GPU2*0J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_15622!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÖ
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

·
&__inference_dense_4_layer_call_fn_1979

inputs-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_4_kernelsequential_1_dense_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_15182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

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
Ù)
½
F__inference_sequential_1_layer_call_and_return_conditional_losses_1876

inputsK
9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel:	F
8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias:	K
9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel:	&F
8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias:&K
9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel:&	F
8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias:	K
9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel:	F
8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpm
dense_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Cast¸
dense_4/MatMul/ReadVariableOpReadVariableOp9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_4/Cast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/MatMulµ
dense_4/BiasAdd/ReadVariableOpReadVariableOp8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/Relu¸
dense_5/MatMul/ReadVariableOpReadVariableOp9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/MatMulµ
dense_5/BiasAdd/ReadVariableOpReadVariableOp8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/Relu¸
dense_6/MatMul/ReadVariableOpReadVariableOp9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/MatMulµ
dense_6/BiasAdd/ReadVariableOpReadVariableOp8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/Relu¸
dense_7/MatMul/ReadVariableOpReadVariableOp9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulµ
dense_7/BiasAdd/ReadVariableOpReadVariableOp8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdds
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
Å
+__inference_sequential_1_layer_call_fn_1805
input_1-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_dense_4_kernelsequential_1_dense_4_biassequential_1_dense_5_kernelsequential_1_dense_5_biassequential_1_dense_6_kernelsequential_1_dense_6_biassequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15672
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
¶
ß
 __inference__traced_restore_2250
file_prefix>
,assignvariableop_sequential_1_dense_4_kernel:	:
,assignvariableop_1_sequential_1_dense_4_bias:	@
.assignvariableop_2_sequential_1_dense_5_kernel:	&:
,assignvariableop_3_sequential_1_dense_5_bias:&@
.assignvariableop_4_sequential_1_dense_6_kernel:&	:
,assignvariableop_5_sequential_1_dense_6_bias:	@
.assignvariableop_6_sequential_1_dense_7_kernel:	:
,assignvariableop_7_sequential_1_dense_7_bias:1
'assignvariableop_8_training_2_adam_iter:	 3
)assignvariableop_9_training_2_adam_beta_1: 4
*assignvariableop_10_training_2_adam_beta_2: 3
)assignvariableop_11_training_2_adam_decay: ;
1assignvariableop_12_training_2_adam_learning_rate: S
Aassignvariableop_13_training_2_adam_sequential_1_dense_4_kernel_m:	M
?assignvariableop_14_training_2_adam_sequential_1_dense_4_bias_m:	S
Aassignvariableop_15_training_2_adam_sequential_1_dense_5_kernel_m:	&M
?assignvariableop_16_training_2_adam_sequential_1_dense_5_bias_m:&S
Aassignvariableop_17_training_2_adam_sequential_1_dense_6_kernel_m:&	M
?assignvariableop_18_training_2_adam_sequential_1_dense_6_bias_m:	S
Aassignvariableop_19_training_2_adam_sequential_1_dense_7_kernel_m:	M
?assignvariableop_20_training_2_adam_sequential_1_dense_7_bias_m:S
Aassignvariableop_21_training_2_adam_sequential_1_dense_4_kernel_v:	M
?assignvariableop_22_training_2_adam_sequential_1_dense_4_bias_v:	S
Aassignvariableop_23_training_2_adam_sequential_1_dense_5_kernel_v:	&M
?assignvariableop_24_training_2_adam_sequential_1_dense_5_bias_v:&S
Aassignvariableop_25_training_2_adam_sequential_1_dense_6_kernel_v:&	M
?assignvariableop_26_training_2_adam_sequential_1_dense_6_bias_v:	S
Aassignvariableop_27_training_2_adam_sequential_1_dense_7_kernel_v:	M
?assignvariableop_28_training_2_adam_sequential_1_dense_7_bias_v:
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

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_sequential_1_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_sequential_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_sequential_1_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_sequential_1_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4³
AssignVariableOp_4AssignVariableOp.assignvariableop_4_sequential_1_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5±
AssignVariableOp_5AssignVariableOp,assignvariableop_5_sequential_1_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_sequential_1_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_sequential_1_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8¬
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_2_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp)assignvariableop_9_training_2_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10²
AssignVariableOp_10AssignVariableOp*assignvariableop_10_training_2_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11±
AssignVariableOp_11AssignVariableOp)assignvariableop_11_training_2_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¹
AssignVariableOp_12AssignVariableOp1assignvariableop_12_training_2_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13É
AssignVariableOp_13AssignVariableOpAassignvariableop_13_training_2_adam_sequential_1_dense_4_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ç
AssignVariableOp_14AssignVariableOp?assignvariableop_14_training_2_adam_sequential_1_dense_4_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15É
AssignVariableOp_15AssignVariableOpAassignvariableop_15_training_2_adam_sequential_1_dense_5_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ç
AssignVariableOp_16AssignVariableOp?assignvariableop_16_training_2_adam_sequential_1_dense_5_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17É
AssignVariableOp_17AssignVariableOpAassignvariableop_17_training_2_adam_sequential_1_dense_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ç
AssignVariableOp_18AssignVariableOp?assignvariableop_18_training_2_adam_sequential_1_dense_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19É
AssignVariableOp_19AssignVariableOpAassignvariableop_19_training_2_adam_sequential_1_dense_7_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ç
AssignVariableOp_20AssignVariableOp?assignvariableop_20_training_2_adam_sequential_1_dense_7_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21É
AssignVariableOp_21AssignVariableOpAassignvariableop_21_training_2_adam_sequential_1_dense_4_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ç
AssignVariableOp_22AssignVariableOp?assignvariableop_22_training_2_adam_sequential_1_dense_4_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23É
AssignVariableOp_23AssignVariableOpAassignvariableop_23_training_2_adam_sequential_1_dense_5_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ç
AssignVariableOp_24AssignVariableOp?assignvariableop_24_training_2_adam_sequential_1_dense_5_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25É
AssignVariableOp_25AssignVariableOpAassignvariableop_25_training_2_adam_sequential_1_dense_6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ç
AssignVariableOp_26AssignVariableOp?assignvariableop_26_training_2_adam_sequential_1_dense_6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27É
AssignVariableOp_27AssignVariableOpAassignvariableop_27_training_2_adam_sequential_1_dense_7_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ç
AssignVariableOp_28AssignVariableOp?assignvariableop_28_training_2_adam_sequential_1_dense_7_bias_vIdentity_28:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix


A__inference_dense_5_layer_call_and_return_conditional_losses_2008

inputsC
1matmul_readvariableop_sequential_1_dense_5_kernel:	&>
0biasadd_readvariableop_sequential_1_dense_5_bias:&
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Þ
Ä
+__inference_sequential_1_layer_call_fn_1831

inputs-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_4_kernelsequential_1_dense_4_biassequential_1_dense_5_kernelsequential_1_dense_5_biassequential_1_dense_6_kernelsequential_1_dense_6_biassequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_16852
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

·
&__inference_dense_5_layer_call_fn_1997

inputs-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_5_kernelsequential_1_dense_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_15332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

·
&__inference_dense_7_layer_call_fn_2033

inputs-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputssequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_15622
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
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ç

A__inference_dense_6_layer_call_and_return_conditional_losses_1548

inputsC
1matmul_readvariableop_sequential_1_dense_6_kernel:&	>
0biasadd_readvariableop_sequential_1_dense_6_bias:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ&: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs
á
Å
+__inference_sequential_1_layer_call_fn_1844
input_1-
sequential_1_dense_4_kernel:	'
sequential_1_dense_4_bias:	-
sequential_1_dense_5_kernel:	&'
sequential_1_dense_5_bias:&-
sequential_1_dense_6_kernel:&	'
sequential_1_dense_6_bias:	-
sequential_1_dense_7_kernel:	'
sequential_1_dense_7_bias:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_1_dense_4_kernelsequential_1_dense_4_biassequential_1_dense_5_kernelsequential_1_dense_5_biassequential_1_dense_6_kernelsequential_1_dense_6_biassequential_1_dense_7_kernelsequential_1_dense_7_bias*
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
GPU2*0J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_16852
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
Ç

A__inference_dense_5_layer_call_and_return_conditional_losses_1533

inputsC
1matmul_readvariableop_sequential_1_dense_5_kernel:	&>
0biasadd_readvariableop_sequential_1_dense_5_bias:&
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp 
MatMul/ReadVariableOpReadVariableOp1matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
MatMul
BiasAdd/ReadVariableOpReadVariableOp0biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ü)
¾
F__inference_sequential_1_layer_call_and_return_conditional_losses_1940
input_1K
9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel:	F
8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias:	K
9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel:	&F
8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias:&K
9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel:&	F
8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias:	K
9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel:	F
8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢dense_6/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpn
dense_4/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Cast¸
dense_4/MatMul/ReadVariableOpReadVariableOp9dense_4_matmul_readvariableop_sequential_1_dense_4_kernel*
_output_shapes

:	*
dtype02
dense_4/MatMul/ReadVariableOp
dense_4/MatMulMatMuldense_4/Cast:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/MatMulµ
dense_4/BiasAdd/ReadVariableOpReadVariableOp8dense_4_biasadd_readvariableop_sequential_1_dense_4_bias*
_output_shapes
:	*
dtype02 
dense_4/BiasAdd/ReadVariableOp¡
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_4/Relu¸
dense_5/MatMul/ReadVariableOpReadVariableOp9dense_5_matmul_readvariableop_sequential_1_dense_5_kernel*
_output_shapes

:	&*
dtype02
dense_5/MatMul/ReadVariableOp
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/MatMulµ
dense_5/BiasAdd/ReadVariableOpReadVariableOp8dense_5_biasadd_readvariableop_sequential_1_dense_5_bias*
_output_shapes
:&*
dtype02 
dense_5/BiasAdd/ReadVariableOp¡
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&2
dense_5/Relu¸
dense_6/MatMul/ReadVariableOpReadVariableOp9dense_6_matmul_readvariableop_sequential_1_dense_6_kernel*
_output_shapes

:&	*
dtype02
dense_6/MatMul/ReadVariableOp
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/MatMulµ
dense_6/BiasAdd/ReadVariableOpReadVariableOp8dense_6_biasadd_readvariableop_sequential_1_dense_6_bias*
_output_shapes
:	*
dtype02 
dense_6/BiasAdd/ReadVariableOp¡
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	2
dense_6/Relu¸
dense_7/MatMul/ReadVariableOpReadVariableOp9dense_7_matmul_readvariableop_sequential_1_dense_7_kernel*
_output_shapes

:	*
dtype02
dense_7/MatMul/ReadVariableOp
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMulµ
dense_7/BiasAdd/ReadVariableOpReadVariableOp8dense_7_biasadd_readvariableop_sequential_1_dense_7_bias*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¡
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdds
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"¨L
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:]
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
-:+	2sequential_1/dense_4/kernel
':%	2sequential_1/dense_4/bias
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
-:+	&2sequential_1/dense_5/kernel
':%&2sequential_1/dense_5/bias
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
-:+&	2sequential_1/dense_6/kernel
':%	2sequential_1/dense_6/bias
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
-:+	2sequential_1/dense_7/kernel
':%2sequential_1/dense_7/bias
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
:	 (2training_2/Adam/iter
 : (2training_2/Adam/beta_1
 : (2training_2/Adam/beta_2
: (2training_2/Adam/decay
':% (2training_2/Adam/learning_rate
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
=:;	2-training_2/Adam/sequential_1/dense_4/kernel/m
7:5	2+training_2/Adam/sequential_1/dense_4/bias/m
=:;	&2-training_2/Adam/sequential_1/dense_5/kernel/m
7:5&2+training_2/Adam/sequential_1/dense_5/bias/m
=:;&	2-training_2/Adam/sequential_1/dense_6/kernel/m
7:5	2+training_2/Adam/sequential_1/dense_6/bias/m
=:;	2-training_2/Adam/sequential_1/dense_7/kernel/m
7:52+training_2/Adam/sequential_1/dense_7/bias/m
=:;	2-training_2/Adam/sequential_1/dense_4/kernel/v
7:5	2+training_2/Adam/sequential_1/dense_4/bias/v
=:;	&2-training_2/Adam/sequential_1/dense_5/kernel/v
7:5&2+training_2/Adam/sequential_1/dense_5/bias/v
=:;&	2-training_2/Adam/sequential_1/dense_6/kernel/v
7:5	2+training_2/Adam/sequential_1/dense_6/bias/v
=:;	2-training_2/Adam/sequential_1/dense_7/kernel/v
7:52+training_2/Adam/sequential_1/dense_7/bias/v
ÊBÇ
__inference__wrapped_model_1499input_1"
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
ú2÷
+__inference_sequential_1_layer_call_fn_1805
+__inference_sequential_1_layer_call_fn_1818
+__inference_sequential_1_layer_call_fn_1831
+__inference_sequential_1_layer_call_fn_1844À
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
æ2ã
F__inference_sequential_1_layer_call_and_return_conditional_losses_1876
F__inference_sequential_1_layer_call_and_return_conditional_losses_1908
F__inference_sequential_1_layer_call_and_return_conditional_losses_1940
F__inference_sequential_1_layer_call_and_return_conditional_losses_1972À
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
Ð2Í
&__inference_dense_4_layer_call_fn_1979¢
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
ë2è
A__inference_dense_4_layer_call_and_return_conditional_losses_1990¢
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
Ð2Í
&__inference_dense_5_layer_call_fn_1997¢
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
ë2è
A__inference_dense_5_layer_call_and_return_conditional_losses_2008¢
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
Ð2Í
&__inference_dense_6_layer_call_fn_2015¢
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
ë2è
A__inference_dense_6_layer_call_and_return_conditional_losses_2026¢
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
Ð2Í
&__inference_dense_7_layer_call_fn_2033¢
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
ë2è
A__inference_dense_7_layer_call_and_return_conditional_losses_2043¢
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
ÉBÆ
"__inference_signature_wrapper_1792input_1"
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
 
__inference__wrapped_model_1499q0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ¡
A__inference_dense_4_layer_call_and_return_conditional_losses_1990\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ	
 y
&__inference_dense_4_layer_call_fn_1979O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ	¡
A__inference_dense_5_layer_call_and_return_conditional_losses_2008\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ&
 y
&__inference_dense_5_layer_call_fn_1997O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ&¡
A__inference_dense_6_layer_call_and_return_conditional_losses_2026\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ&
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ	
 y
&__inference_dense_6_layer_call_fn_2015O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ&
ª "ÿÿÿÿÿÿÿÿÿ	¡
A__inference_dense_7_layer_call_and_return_conditional_losses_2043\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_7_layer_call_fn_2033O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ´
F__inference_sequential_1_layer_call_and_return_conditional_losses_1876j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
F__inference_sequential_1_layer_call_and_return_conditional_losses_1908j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_sequential_1_layer_call_and_return_conditional_losses_1940k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
F__inference_sequential_1_layer_call_and_return_conditional_losses_1972k8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_1_layer_call_fn_1805^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_1_layer_call_fn_1818]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_1_layer_call_fn_1831]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_1_layer_call_fn_1844^8¢5
.¢+
!
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¢
"__inference_signature_wrapper_1792|;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ