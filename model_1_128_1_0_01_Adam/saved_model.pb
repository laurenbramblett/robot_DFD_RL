??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:"@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:@*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:@@*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:@*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:@@*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:@*
dtype0
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:@*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
|
training_4/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_4/Adam/iter
u
(training_4/Adam/iter/Read/ReadVariableOpReadVariableOptraining_4/Adam/iter*
_output_shapes
: *
dtype0	
?
training_4/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_1
y
*training_4/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_4/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_2
y
*training_4/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_4/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_4/Adam/decay
w
)training_4/Adam/decay/Read/ReadVariableOpReadVariableOptraining_4/Adam/decay*
_output_shapes
: *
dtype0
?
training_4/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_4/Adam/learning_rate
?
1training_4/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_4/Adam/learning_rate*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0
?
 training_4/Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*1
shared_name" training_4/Adam/dense_8/kernel/m
?
4training_4/Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_8/kernel/m*
_output_shapes

:"@*
dtype0
?
training_4/Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_4/Adam/dense_8/bias/m
?
2training_4/Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_8/bias/m*
_output_shapes
:@*
dtype0
?
 training_4/Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" training_4/Adam/dense_9/kernel/m
?
4training_4/Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_9/kernel/m*
_output_shapes

:@@*
dtype0
?
training_4/Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_4/Adam/dense_9/bias/m
?
2training_4/Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_9/bias/m*
_output_shapes
:@*
dtype0
?
!training_4/Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*2
shared_name#!training_4/Adam/dense_10/kernel/m
?
5training_4/Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOp!training_4/Adam/dense_10/kernel/m*
_output_shapes

:@@*
dtype0
?
training_4/Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_4/Adam/dense_10/bias/m
?
3training_4/Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_10/bias/m*
_output_shapes
:@*
dtype0
?
!training_4/Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!training_4/Adam/dense_11/kernel/m
?
5training_4/Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOp!training_4/Adam/dense_11/kernel/m*
_output_shapes

:@*
dtype0
?
training_4/Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_4/Adam/dense_11/bias/m
?
3training_4/Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_11/bias/m*
_output_shapes
:*
dtype0
?
 training_4/Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*1
shared_name" training_4/Adam/dense_8/kernel/v
?
4training_4/Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_8/kernel/v*
_output_shapes

:"@*
dtype0
?
training_4/Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_4/Adam/dense_8/bias/v
?
2training_4/Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_8/bias/v*
_output_shapes
:@*
dtype0
?
 training_4/Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" training_4/Adam/dense_9/kernel/v
?
4training_4/Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp training_4/Adam/dense_9/kernel/v*
_output_shapes

:@@*
dtype0
?
training_4/Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name training_4/Adam/dense_9/bias/v
?
2training_4/Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_9/bias/v*
_output_shapes
:@*
dtype0
?
!training_4/Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*2
shared_name#!training_4/Adam/dense_10/kernel/v
?
5training_4/Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOp!training_4/Adam/dense_10/kernel/v*
_output_shapes

:@@*
dtype0
?
training_4/Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!training_4/Adam/dense_10/bias/v
?
3training_4/Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_10/bias/v*
_output_shapes
:@*
dtype0
?
!training_4/Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!training_4/Adam/dense_11/kernel/v
?
5training_4/Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOp!training_4/Adam/dense_11/kernel/v*
_output_shapes

:@*
dtype0
?
training_4/Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!training_4/Adam/dense_11/bias/v
?
3training_4/Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/dense_11/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?6
value?6B?6 B?6
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratemSmTmUmVmWmX&mY'mZv[v\v]v^v_v`&va'vb*
<
0
1
2
3
4
5
&6
'7*
<
0
1
2
3
4
5
&6
'7*
* 
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

8serving_default* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
WQ
VARIABLE_VALUEtraining_4/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEtraining_4/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEtraining_4/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining_4/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEtraining_4/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

M0*
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
* 
H
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api*
VP
VARIABLE_VALUEtotal_144keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_144keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

N0
O1*

Q	variables*
??
VARIABLE_VALUE training_4/Adam/dense_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_4/Adam/dense_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!training_4/Adam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!training_4/Adam/dense_11/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_11/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_4/Adam/dense_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_4/Adam/dense_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!training_4/Adam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!training_4/Adam/dense_11/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEtraining_4/Adam/dense_11/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_dense_8_inputPlaceholder*'
_output_shapes
:?????????"*
dtype0*
shape:?????????"
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_8_inputdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6232
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp(training_4/Adam/iter/Read/ReadVariableOp*training_4/Adam/beta_1/Read/ReadVariableOp*training_4/Adam/beta_2/Read/ReadVariableOp)training_4/Adam/decay/Read/ReadVariableOp1training_4/Adam/learning_rate/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOp4training_4/Adam/dense_8/kernel/m/Read/ReadVariableOp2training_4/Adam/dense_8/bias/m/Read/ReadVariableOp4training_4/Adam/dense_9/kernel/m/Read/ReadVariableOp2training_4/Adam/dense_9/bias/m/Read/ReadVariableOp5training_4/Adam/dense_10/kernel/m/Read/ReadVariableOp3training_4/Adam/dense_10/bias/m/Read/ReadVariableOp5training_4/Adam/dense_11/kernel/m/Read/ReadVariableOp3training_4/Adam/dense_11/bias/m/Read/ReadVariableOp4training_4/Adam/dense_8/kernel/v/Read/ReadVariableOp2training_4/Adam/dense_8/bias/v/Read/ReadVariableOp4training_4/Adam/dense_9/kernel/v/Read/ReadVariableOp2training_4/Adam/dense_9/bias/v/Read/ReadVariableOp5training_4/Adam/dense_10/kernel/v/Read/ReadVariableOp3training_4/Adam/dense_10/bias/v/Read/ReadVariableOp5training_4/Adam/dense_11/kernel/v/Read/ReadVariableOp3training_4/Adam/dense_11/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_6419
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/biastraining_4/Adam/itertraining_4/Adam/beta_1training_4/Adam/beta_2training_4/Adam/decaytraining_4/Adam/learning_ratetotal_14count_14 training_4/Adam/dense_8/kernel/mtraining_4/Adam/dense_8/bias/m training_4/Adam/dense_9/kernel/mtraining_4/Adam/dense_9/bias/m!training_4/Adam/dense_10/kernel/mtraining_4/Adam/dense_10/bias/m!training_4/Adam/dense_11/kernel/mtraining_4/Adam/dense_11/bias/m training_4/Adam/dense_8/kernel/vtraining_4/Adam/dense_8/bias/v training_4/Adam/dense_9/kernel/vtraining_4/Adam/dense_9/bias/v!training_4/Adam/dense_10/kernel/vtraining_4/Adam/dense_10/bias/v!training_4/Adam/dense_11/kernel/vtraining_4/Adam/dense_11/bias/v*+
Tin$
"2 *
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_6522??
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_5924

inputs(
dense_8_dense_8_kernel:"@"
dense_8_dense_8_bias:@(
dense_9_dense_9_kernel:@@"
dense_9_dense_9_bias:@*
dense_10_dense_10_kernel:@@$
dense_10_dense_10_bias:@*
dense_11_dense_11_kernel:@$
dense_11_dense_11_bias:
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_dense_8_kerneldense_8_dense_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_5875?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_dense_9_kerneldense_9_dense_9_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_5890?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_dense_10_kerneldense_10_dense_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_5905?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_5919x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
&__inference_dense_8_layer_call_fn_6239

inputs 
dense_8_kernel:"@
dense_8_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_kerneldense_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_5875o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????": : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
'__inference_dense_10_layer_call_fn_6275

inputs!
dense_10_kernel:@@
dense_10_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_10_kerneldense_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_5905o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?F
?
__inference__traced_save_6419
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop3
/savev2_training_4_adam_iter_read_readvariableop	5
1savev2_training_4_adam_beta_1_read_readvariableop5
1savev2_training_4_adam_beta_2_read_readvariableop4
0savev2_training_4_adam_decay_read_readvariableop<
8savev2_training_4_adam_learning_rate_read_readvariableop'
#savev2_total_14_read_readvariableop'
#savev2_count_14_read_readvariableop?
;savev2_training_4_adam_dense_8_kernel_m_read_readvariableop=
9savev2_training_4_adam_dense_8_bias_m_read_readvariableop?
;savev2_training_4_adam_dense_9_kernel_m_read_readvariableop=
9savev2_training_4_adam_dense_9_bias_m_read_readvariableop@
<savev2_training_4_adam_dense_10_kernel_m_read_readvariableop>
:savev2_training_4_adam_dense_10_bias_m_read_readvariableop@
<savev2_training_4_adam_dense_11_kernel_m_read_readvariableop>
:savev2_training_4_adam_dense_11_bias_m_read_readvariableop?
;savev2_training_4_adam_dense_8_kernel_v_read_readvariableop=
9savev2_training_4_adam_dense_8_bias_v_read_readvariableop?
;savev2_training_4_adam_dense_9_kernel_v_read_readvariableop=
9savev2_training_4_adam_dense_9_bias_v_read_readvariableop@
<savev2_training_4_adam_dense_10_kernel_v_read_readvariableop>
:savev2_training_4_adam_dense_10_bias_v_read_readvariableop@
<savev2_training_4_adam_dense_11_kernel_v_read_readvariableop>
:savev2_training_4_adam_dense_11_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop/savev2_training_4_adam_iter_read_readvariableop1savev2_training_4_adam_beta_1_read_readvariableop1savev2_training_4_adam_beta_2_read_readvariableop0savev2_training_4_adam_decay_read_readvariableop8savev2_training_4_adam_learning_rate_read_readvariableop#savev2_total_14_read_readvariableop#savev2_count_14_read_readvariableop;savev2_training_4_adam_dense_8_kernel_m_read_readvariableop9savev2_training_4_adam_dense_8_bias_m_read_readvariableop;savev2_training_4_adam_dense_9_kernel_m_read_readvariableop9savev2_training_4_adam_dense_9_bias_m_read_readvariableop<savev2_training_4_adam_dense_10_kernel_m_read_readvariableop:savev2_training_4_adam_dense_10_bias_m_read_readvariableop<savev2_training_4_adam_dense_11_kernel_m_read_readvariableop:savev2_training_4_adam_dense_11_bias_m_read_readvariableop;savev2_training_4_adam_dense_8_kernel_v_read_readvariableop9savev2_training_4_adam_dense_8_bias_v_read_readvariableop;savev2_training_4_adam_dense_9_kernel_v_read_readvariableop9savev2_training_4_adam_dense_9_bias_v_read_readvariableop<savev2_training_4_adam_dense_10_kernel_v_read_readvariableop:savev2_training_4_adam_dense_10_bias_v_read_readvariableop<savev2_training_4_adam_dense_11_kernel_v_read_readvariableop:savev2_training_4_adam_dense_11_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :"@:@:@@:@:@@:@:@:: : : : : : : :"@:@:@@:@:@@:@:@::"@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:"@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	
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
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:"@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:"@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
: 
?

?
A__inference_dense_8_layer_call_and_return_conditional_losses_5875

inputs6
$matmul_readvariableop_dense_8_kernel:"@1
#biasadd_readvariableop_dense_8_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_8_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_8_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_6232
dense_8_input 
dense_8_kernel:"@
dense_8_bias:@ 
dense_9_kernel:@@
dense_9_bias:@!
dense_10_kernel:@@
dense_10_bias:@!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_kerneldense_8_biasdense_9_kerneldense_9_biasdense_10_kerneldense_10_biasdense_11_kerneldense_11_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_5857o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6041

inputs(
dense_8_dense_8_kernel:"@"
dense_8_dense_8_bias:@(
dense_9_dense_9_kernel:@@"
dense_9_dense_9_bias:@*
dense_10_dense_10_kernel:@@$
dense_10_dense_10_bias:@*
dense_11_dense_11_kernel:@$
dense_11_dense_11_bias:
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_dense_8_kerneldense_8_dense_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_5875?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_dense_9_kerneldense_9_dense_9_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_5890?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_dense_10_kerneldense_10_dense_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_5905?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_5919x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?

?
A__inference_dense_8_layer_call_and_return_conditional_losses_6250

inputs6
$matmul_readvariableop_dense_8_kernel:"@1
#biasadd_readvariableop_dense_8_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_8_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_8_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????": : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
B__inference_dense_11_layer_call_and_return_conditional_losses_6303

inputs7
%matmul_readvariableop_dense_11_kernel:@2
$biasadd_readvariableop_dense_11_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_11_kernel*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
+__inference_sequential_2_layer_call_fn_6155

inputs 
dense_8_kernel:"@
dense_8_bias:@ 
dense_9_kernel:@@
dense_9_bias:@!
dense_10_kernel:@@
dense_10_bias:@!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_kerneldense_8_biasdense_9_kerneldense_9_biasdense_10_kerneldense_10_biasdense_11_kerneldense_11_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_6041o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
+__inference_sequential_2_layer_call_fn_6097
dense_8_input 
dense_8_kernel:"@
dense_8_bias:@ 
dense_9_kernel:@@
dense_9_bias:@!
dense_10_kernel:@@
dense_10_bias:@!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_kerneldense_8_biasdense_9_kerneldense_9_biasdense_10_kerneldense_10_biasdense_11_kerneldense_11_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_6041o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
?#
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6186

inputs>
,dense_8_matmul_readvariableop_dense_8_kernel:"@9
+dense_8_biasadd_readvariableop_dense_8_bias:@>
,dense_9_matmul_readvariableop_dense_9_kernel:@@9
+dense_9_biasadd_readvariableop_dense_9_bias:@@
.dense_10_matmul_readvariableop_dense_10_kernel:@@;
-dense_10_biasadd_readvariableop_dense_10_bias:@@
.dense_11_matmul_readvariableop_dense_11_kernel:@;
-dense_11_biasadd_readvariableop_dense_11_bias:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp,dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:"@*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_8/BiasAdd/ReadVariableOpReadVariableOp+dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:@*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_9/MatMul/ReadVariableOpReadVariableOp,dense_9_matmul_readvariableop_dense_9_kernel*
_output_shapes

:@@*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_9/BiasAdd/ReadVariableOpReadVariableOp+dense_9_biasadd_readvariableop_dense_9_bias*
_output_shapes
:@*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_10/MatMul/ReadVariableOpReadVariableOp.dense_10_matmul_readvariableop_dense_10_kernel*
_output_shapes

:@@*
dtype0?
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_10/BiasAdd/ReadVariableOpReadVariableOp-dense_10_biasadd_readvariableop_dense_10_bias*
_output_shapes
:@*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_11/MatMul/ReadVariableOpReadVariableOp.dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes

:@*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp-dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?

?
B__inference_dense_10_layer_call_and_return_conditional_losses_6286

inputs7
%matmul_readvariableop_dense_10_kernel:@@2
$biasadd_readvariableop_dense_10_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_10_kernel*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_10_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
B__inference_dense_11_layer_call_and_return_conditional_losses_5919

inputs7
%matmul_readvariableop_dense_11_kernel:@2
$biasadd_readvariableop_dense_11_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_11_kernel*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6129
dense_8_input(
dense_8_dense_8_kernel:"@"
dense_8_dense_8_bias:@(
dense_9_dense_9_kernel:@@"
dense_9_dense_9_bias:@*
dense_10_dense_10_kernel:@@$
dense_10_dense_10_bias:@*
dense_11_dense_11_kernel:@$
dense_11_dense_11_bias:
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_dense_8_kerneldense_8_dense_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_5875?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_dense_9_kerneldense_9_dense_9_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_5890?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_dense_10_kerneldense_10_dense_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_5905?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_5919x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
?

?
A__inference_dense_9_layer_call_and_return_conditional_losses_6268

inputs6
$matmul_readvariableop_dense_9_kernel:@@1
#biasadd_readvariableop_dense_9_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_9_kernel*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_9_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?,
?
__inference__wrapped_model_5857
dense_8_inputK
9sequential_2_dense_8_matmul_readvariableop_dense_8_kernel:"@F
8sequential_2_dense_8_biasadd_readvariableop_dense_8_bias:@K
9sequential_2_dense_9_matmul_readvariableop_dense_9_kernel:@@F
8sequential_2_dense_9_biasadd_readvariableop_dense_9_bias:@M
;sequential_2_dense_10_matmul_readvariableop_dense_10_kernel:@@H
:sequential_2_dense_10_biasadd_readvariableop_dense_10_bias:@M
;sequential_2_dense_11_matmul_readvariableop_dense_11_kernel:@H
:sequential_2_dense_11_biasadd_readvariableop_dense_11_bias:
identity??,sequential_2/dense_10/BiasAdd/ReadVariableOp?+sequential_2/dense_10/MatMul/ReadVariableOp?,sequential_2/dense_11/BiasAdd/ReadVariableOp?+sequential_2/dense_11/MatMul/ReadVariableOp?+sequential_2/dense_8/BiasAdd/ReadVariableOp?*sequential_2/dense_8/MatMul/ReadVariableOp?+sequential_2/dense_9/BiasAdd/ReadVariableOp?*sequential_2/dense_9/MatMul/ReadVariableOp?
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp9sequential_2_dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:"@*
dtype0?
sequential_2/dense_8/MatMulMatMuldense_8_input2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp8sequential_2_dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:@*
dtype0?
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp9sequential_2_dense_9_matmul_readvariableop_dense_9_kernel*
_output_shapes

:@@*
dtype0?
sequential_2/dense_9/MatMulMatMul'sequential_2/dense_8/Relu:activations:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp8sequential_2_dense_9_biasadd_readvariableop_dense_9_bias*
_output_shapes
:@*
dtype0?
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@z
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp;sequential_2_dense_10_matmul_readvariableop_dense_10_kernel*
_output_shapes

:@@*
dtype0?
sequential_2/dense_10/MatMulMatMul'sequential_2/dense_9/Relu:activations:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_dense_10_biasadd_readvariableop_dense_10_bias*
_output_shapes
:@*
dtype0?
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@|
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp;sequential_2_dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes

:@*
dtype0?
sequential_2/dense_11/MatMulMatMul(sequential_2/dense_10/Relu:activations:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0?
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_2/dense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
?
?
'__inference_dense_11_layer_call_fn_6293

inputs!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_kerneldense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_5919o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
A__inference_dense_9_layer_call_and_return_conditional_losses_5890

inputs6
$matmul_readvariableop_dense_9_kernel:@@1
#biasadd_readvariableop_dense_9_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpz
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_9_kernel*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_9_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_dense_10_layer_call_and_return_conditional_losses_5905

inputs7
%matmul_readvariableop_dense_10_kernel:@@2
$biasadd_readvariableop_dense_10_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_10_kernel*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_10_bias*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6113
dense_8_input(
dense_8_dense_8_kernel:"@"
dense_8_dense_8_bias:@(
dense_9_dense_9_kernel:@@"
dense_9_dense_9_bias:@*
dense_10_dense_10_kernel:@@$
dense_10_dense_10_bias:@*
dense_11_dense_11_kernel:@$
dense_11_dense_11_bias:
identity?? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_dense_8_kerneldense_8_dense_8_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_5875?
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_dense_9_kerneldense_9_dense_9_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_5890?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_dense_10_kerneldense_10_dense_10_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_10_layer_call_and_return_conditional_losses_5905?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_5919x
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
??
?
 __inference__traced_restore_6522
file_prefix1
assignvariableop_dense_8_kernel:"@-
assignvariableop_1_dense_8_bias:@3
!assignvariableop_2_dense_9_kernel:@@-
assignvariableop_3_dense_9_bias:@4
"assignvariableop_4_dense_10_kernel:@@.
 assignvariableop_5_dense_10_bias:@4
"assignvariableop_6_dense_11_kernel:@.
 assignvariableop_7_dense_11_bias:1
'assignvariableop_8_training_4_adam_iter:	 3
)assignvariableop_9_training_4_adam_beta_1: 4
*assignvariableop_10_training_4_adam_beta_2: 3
)assignvariableop_11_training_4_adam_decay: ;
1assignvariableop_12_training_4_adam_learning_rate: &
assignvariableop_13_total_14: &
assignvariableop_14_count_14: F
4assignvariableop_15_training_4_adam_dense_8_kernel_m:"@@
2assignvariableop_16_training_4_adam_dense_8_bias_m:@F
4assignvariableop_17_training_4_adam_dense_9_kernel_m:@@@
2assignvariableop_18_training_4_adam_dense_9_bias_m:@G
5assignvariableop_19_training_4_adam_dense_10_kernel_m:@@A
3assignvariableop_20_training_4_adam_dense_10_bias_m:@G
5assignvariableop_21_training_4_adam_dense_11_kernel_m:@A
3assignvariableop_22_training_4_adam_dense_11_bias_m:F
4assignvariableop_23_training_4_adam_dense_8_kernel_v:"@@
2assignvariableop_24_training_4_adam_dense_8_bias_v:@F
4assignvariableop_25_training_4_adam_dense_9_kernel_v:@@@
2assignvariableop_26_training_4_adam_dense_9_bias_v:@G
5assignvariableop_27_training_4_adam_dense_10_kernel_v:@@A
3assignvariableop_28_training_4_adam_dense_10_bias_v:@G
5assignvariableop_29_training_4_adam_dense_11_kernel_v:@A
3assignvariableop_30_training_4_adam_dense_11_bias_v:
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_4_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_training_4_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp*assignvariableop_10_training_4_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp)assignvariableop_11_training_4_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp1assignvariableop_12_training_4_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_14Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_14Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp4assignvariableop_15_training_4_adam_dense_8_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_training_4_adam_dense_8_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp4assignvariableop_17_training_4_adam_dense_9_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_training_4_adam_dense_9_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp5assignvariableop_19_training_4_adam_dense_10_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp3assignvariableop_20_training_4_adam_dense_10_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_training_4_adam_dense_11_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_training_4_adam_dense_11_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp4assignvariableop_23_training_4_adam_dense_8_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp2assignvariableop_24_training_4_adam_dense_8_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_training_4_adam_dense_9_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_training_4_adam_dense_9_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp5assignvariableop_27_training_4_adam_dense_10_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_training_4_adam_dense_10_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp5assignvariableop_29_training_4_adam_dense_11_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_training_4_adam_dense_11_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
?
?
&__inference_dense_9_layer_call_fn_6257

inputs 
dense_9_kernel:@@
dense_9_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_9_kerneldense_9_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_9_layer_call_and_return_conditional_losses_5890o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
+__inference_sequential_2_layer_call_fn_6142

inputs 
dense_8_kernel:"@
dense_8_bias:@ 
dense_9_kernel:@@
dense_9_bias:@!
dense_10_kernel:@@
dense_10_bias:@!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_kerneldense_8_biasdense_9_kerneldense_9_biasdense_10_kerneldense_10_biasdense_11_kerneldense_11_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_5924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
+__inference_sequential_2_layer_call_fn_5935
dense_8_input 
dense_8_kernel:"@
dense_8_bias:@ 
dense_9_kernel:@@
dense_9_bias:@!
dense_10_kernel:@@
dense_10_bias:@!
dense_11_kernel:@
dense_11_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputdense_8_kerneldense_8_biasdense_9_kerneldense_9_biasdense_10_kerneldense_10_biasdense_11_kerneldense_11_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_5924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????"
'
_user_specified_namedense_8_input
?#
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6217

inputs>
,dense_8_matmul_readvariableop_dense_8_kernel:"@9
+dense_8_biasadd_readvariableop_dense_8_bias:@>
,dense_9_matmul_readvariableop_dense_9_kernel:@@9
+dense_9_biasadd_readvariableop_dense_9_bias:@@
.dense_10_matmul_readvariableop_dense_10_kernel:@@;
-dense_10_biasadd_readvariableop_dense_10_bias:@@
.dense_11_matmul_readvariableop_dense_11_kernel:@;
-dense_11_biasadd_readvariableop_dense_11_bias:
identity??dense_10/BiasAdd/ReadVariableOp?dense_10/MatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?dense_11/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp,dense_8_matmul_readvariableop_dense_8_kernel*
_output_shapes

:"@*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_8/BiasAdd/ReadVariableOpReadVariableOp+dense_8_biasadd_readvariableop_dense_8_bias*
_output_shapes
:@*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_9/MatMul/ReadVariableOpReadVariableOp,dense_9_matmul_readvariableop_dense_9_kernel*
_output_shapes

:@@*
dtype0?
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_9/BiasAdd/ReadVariableOpReadVariableOp+dense_9_biasadd_readvariableop_dense_9_bias*
_output_shapes
:@*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_10/MatMul/ReadVariableOpReadVariableOp.dense_10_matmul_readvariableop_dense_10_kernel*
_output_shapes

:@@*
dtype0?
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_10/BiasAdd/ReadVariableOpReadVariableOp-dense_10_biasadd_readvariableop_dense_10_bias*
_output_shapes
:@*
dtype0?
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_11/MatMul/ReadVariableOpReadVariableOp.dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes

:@*
dtype0?
dense_11/MatMulMatMuldense_10/Relu:activations:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_11/BiasAdd/ReadVariableOpReadVariableOp-dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0?
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
dense_8_input6
serving_default_dense_8_input:0?????????"<
dense_110
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?]
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratemSmTmUmVmWmX&mY'mZv[v\v]v^v_v`&va'vb"
	optimizer
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
X
0
1
2
3
4
5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_sequential_2_layer_call_fn_5935
+__inference_sequential_2_layer_call_fn_6142
+__inference_sequential_2_layer_call_fn_6155
+__inference_sequential_2_layer_call_fn_6097?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6186
F__inference_sequential_2_layer_call_and_return_conditional_losses_6217
F__inference_sequential_2_layer_call_and_return_conditional_losses_6113
F__inference_sequential_2_layer_call_and_return_conditional_losses_6129?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_5857dense_8_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
8serving_default"
signature_map
 :"@2dense_8/kernel
:@2dense_8/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_8_layer_call_fn_6239?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_8_layer_call_and_return_conditional_losses_6250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 :@@2dense_9/kernel
:@2dense_9/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_9_layer_call_fn_6257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_9_layer_call_and_return_conditional_losses_6268?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
!:@@2dense_10/kernel
:@2dense_10/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_10_layer_call_fn_6275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_10_layer_call_and_return_conditional_losses_6286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
!:@2dense_11/kernel
:2dense_11/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_11_layer_call_fn_6293?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_11_layer_call_and_return_conditional_losses_6303?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2training_4/Adam/iter
 : (2training_4/Adam/beta_1
 : (2training_4/Adam/beta_2
: (2training_4/Adam/decay
':% (2training_4/Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
"__inference_signature_wrapper_6232dense_8_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
^
	Ntotal
	Ocount
P
_fn_kwargs
Q	variables
R	keras_api"
_tf_keras_metric
:  (2total_14
:  (2count_14
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
0:."@2 training_4/Adam/dense_8/kernel/m
*:(@2training_4/Adam/dense_8/bias/m
0:.@@2 training_4/Adam/dense_9/kernel/m
*:(@2training_4/Adam/dense_9/bias/m
1:/@@2!training_4/Adam/dense_10/kernel/m
+:)@2training_4/Adam/dense_10/bias/m
1:/@2!training_4/Adam/dense_11/kernel/m
+:)2training_4/Adam/dense_11/bias/m
0:."@2 training_4/Adam/dense_8/kernel/v
*:(@2training_4/Adam/dense_8/bias/v
0:.@@2 training_4/Adam/dense_9/kernel/v
*:(@2training_4/Adam/dense_9/bias/v
1:/@@2!training_4/Adam/dense_10/kernel/v
+:)@2training_4/Adam/dense_10/bias/v
1:/@2!training_4/Adam/dense_11/kernel/v
+:)2training_4/Adam/dense_11/bias/v?
__inference__wrapped_model_5857w&'6?3
,?)
'?$
dense_8_input?????????"
? "3?0
.
dense_11"?
dense_11??????????
B__inference_dense_10_layer_call_and_return_conditional_losses_6286\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? z
'__inference_dense_10_layer_call_fn_6275O/?,
%?"
 ?
inputs?????????@
? "??????????@?
B__inference_dense_11_layer_call_and_return_conditional_losses_6303\&'/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_11_layer_call_fn_6293O&'/?,
%?"
 ?
inputs?????????@
? "???????????
A__inference_dense_8_layer_call_and_return_conditional_losses_6250\/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????@
? y
&__inference_dense_8_layer_call_fn_6239O/?,
%?"
 ?
inputs?????????"
? "??????????@?
A__inference_dense_9_layer_call_and_return_conditional_losses_6268\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? y
&__inference_dense_9_layer_call_fn_6257O/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6113q&'>?;
4?1
'?$
dense_8_input?????????"
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6129q&'>?;
4?1
'?$
dense_8_input?????????"
p

 
? "%?"
?
0?????????
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6186j&'7?4
-?*
 ?
inputs?????????"
p 

 
? "%?"
?
0?????????
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_6217j&'7?4
-?*
 ?
inputs?????????"
p

 
? "%?"
?
0?????????
? ?
+__inference_sequential_2_layer_call_fn_5935d&'>?;
4?1
'?$
dense_8_input?????????"
p 

 
? "???????????
+__inference_sequential_2_layer_call_fn_6097d&'>?;
4?1
'?$
dense_8_input?????????"
p

 
? "???????????
+__inference_sequential_2_layer_call_fn_6142]&'7?4
-?*
 ?
inputs?????????"
p 

 
? "???????????
+__inference_sequential_2_layer_call_fn_6155]&'7?4
-?*
 ?
inputs?????????"
p

 
? "???????????
"__inference_signature_wrapper_6232?&'G?D
? 
=?:
8
dense_8_input'?$
dense_8_input?????????""3?0
.
dense_11"?
dense_11?????????