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
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:"@*
dtype0
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:@*
dtype0
{
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?* 
shared_namedense_57/kernel
t
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel*
_output_shapes
:	@?*
dtype0
s
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_57/bias
l
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes	
:?*
dtype0
|
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_58/kernel
u
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel* 
_output_shapes
:
??*
dtype0
s
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_58/bias
l
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes	
:?*
dtype0
{
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_59/kernel
t
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes
:	?*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:*
dtype0
~
training_28/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *&
shared_nametraining_28/Adam/iter
w
)training_28/Adam/iter/Read/ReadVariableOpReadVariableOptraining_28/Adam/iter*
_output_shapes
: *
dtype0	
?
training_28/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_28/Adam/beta_1
{
+training_28/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_28/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_28/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_28/Adam/beta_2
{
+training_28/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_28/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_28/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_28/Adam/decay
y
*training_28/Adam/decay/Read/ReadVariableOpReadVariableOptraining_28/Adam/decay*
_output_shapes
: *
dtype0
?
training_28/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_28/Adam/learning_rate
?
2training_28/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_28/Adam/learning_rate*
_output_shapes
: *
dtype0
d
total_48VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_48
]
total_48/Read/ReadVariableOpReadVariableOptotal_48*
_output_shapes
: *
dtype0
d
count_48VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_48
]
count_48/Read/ReadVariableOpReadVariableOpcount_48*
_output_shapes
: *
dtype0
?
"training_28/Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*3
shared_name$"training_28/Adam/dense_56/kernel/m
?
6training_28/Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_56/kernel/m*
_output_shapes

:"@*
dtype0
?
 training_28/Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_28/Adam/dense_56/bias/m
?
4training_28/Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_56/bias/m*
_output_shapes
:@*
dtype0
?
"training_28/Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"training_28/Adam/dense_57/kernel/m
?
6training_28/Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_57/kernel/m*
_output_shapes
:	@?*
dtype0
?
 training_28/Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_28/Adam/dense_57/bias/m
?
4training_28/Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_57/bias/m*
_output_shapes	
:?*
dtype0
?
"training_28/Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"training_28/Adam/dense_58/kernel/m
?
6training_28/Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_58/kernel/m* 
_output_shapes
:
??*
dtype0
?
 training_28/Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_28/Adam/dense_58/bias/m
?
4training_28/Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_58/bias/m*
_output_shapes	
:?*
dtype0
?
"training_28/Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"training_28/Adam/dense_59/kernel/m
?
6training_28/Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_59/kernel/m*
_output_shapes
:	?*
dtype0
?
 training_28/Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_28/Adam/dense_59/bias/m
?
4training_28/Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_59/bias/m*
_output_shapes
:*
dtype0
?
"training_28/Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*3
shared_name$"training_28/Adam/dense_56/kernel/v
?
6training_28/Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_56/kernel/v*
_output_shapes

:"@*
dtype0
?
 training_28/Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_28/Adam/dense_56/bias/v
?
4training_28/Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_56/bias/v*
_output_shapes
:@*
dtype0
?
"training_28/Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"training_28/Adam/dense_57/kernel/v
?
6training_28/Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_57/kernel/v*
_output_shapes
:	@?*
dtype0
?
 training_28/Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_28/Adam/dense_57/bias/v
?
4training_28/Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_57/bias/v*
_output_shapes	
:?*
dtype0
?
"training_28/Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"training_28/Adam/dense_58/kernel/v
?
6training_28/Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_58/kernel/v* 
_output_shapes
:
??*
dtype0
?
 training_28/Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_28/Adam/dense_58/bias/v
?
4training_28/Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_58/bias/v*
_output_shapes	
:?*
dtype0
?
"training_28/Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"training_28/Adam/dense_59/kernel/v
?
6training_28/Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOp"training_28/Adam/dense_59/kernel/v*
_output_shapes
:	?*
dtype0
?
 training_28/Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_28/Adam/dense_59/bias/v
?
4training_28/Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOp training_28/Adam/dense_59/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?7
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
_Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_56/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_58/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_58/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_59/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_59/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
XR
VARIABLE_VALUEtraining_28/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_28/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_28/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEtraining_28/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEtraining_28/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEtotal_484keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_484keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

N0
O1*

Q	variables*
??
VARIABLE_VALUE"training_28/Adam/dense_56/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_56/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_57/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_57/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_58/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_58/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_59/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_59/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_56/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_56/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_57/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_57/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_58/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_58/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_28/Adam/dense_59/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_28/Adam/dense_59/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_dense_56_inputPlaceholder*'
_output_shapes
:?????????"*
dtype0*
shape:?????????"
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_56_inputdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias*
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
GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_27363
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOp)training_28/Adam/iter/Read/ReadVariableOp+training_28/Adam/beta_1/Read/ReadVariableOp+training_28/Adam/beta_2/Read/ReadVariableOp*training_28/Adam/decay/Read/ReadVariableOp2training_28/Adam/learning_rate/Read/ReadVariableOptotal_48/Read/ReadVariableOpcount_48/Read/ReadVariableOp6training_28/Adam/dense_56/kernel/m/Read/ReadVariableOp4training_28/Adam/dense_56/bias/m/Read/ReadVariableOp6training_28/Adam/dense_57/kernel/m/Read/ReadVariableOp4training_28/Adam/dense_57/bias/m/Read/ReadVariableOp6training_28/Adam/dense_58/kernel/m/Read/ReadVariableOp4training_28/Adam/dense_58/bias/m/Read/ReadVariableOp6training_28/Adam/dense_59/kernel/m/Read/ReadVariableOp4training_28/Adam/dense_59/bias/m/Read/ReadVariableOp6training_28/Adam/dense_56/kernel/v/Read/ReadVariableOp4training_28/Adam/dense_56/bias/v/Read/ReadVariableOp6training_28/Adam/dense_57/kernel/v/Read/ReadVariableOp4training_28/Adam/dense_57/bias/v/Read/ReadVariableOp6training_28/Adam/dense_58/kernel/v/Read/ReadVariableOp4training_28/Adam/dense_58/bias/v/Read/ReadVariableOp6training_28/Adam/dense_59/kernel/v/Read/ReadVariableOp4training_28/Adam/dense_59/bias/v/Read/ReadVariableOpConst*,
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
GPU 2J 8? *'
f"R 
__inference__traced_save_27550
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biastraining_28/Adam/itertraining_28/Adam/beta_1training_28/Adam/beta_2training_28/Adam/decaytraining_28/Adam/learning_ratetotal_48count_48"training_28/Adam/dense_56/kernel/m training_28/Adam/dense_56/bias/m"training_28/Adam/dense_57/kernel/m training_28/Adam/dense_57/bias/m"training_28/Adam/dense_58/kernel/m training_28/Adam/dense_58/bias/m"training_28/Adam/dense_59/kernel/m training_28/Adam/dense_59/bias/m"training_28/Adam/dense_56/kernel/v training_28/Adam/dense_56/bias/v"training_28/Adam/dense_57/kernel/v training_28/Adam/dense_57/bias/v"training_28/Adam/dense_58/kernel/v training_28/Adam/dense_58/bias/v"training_28/Adam/dense_59/kernel/v training_28/Adam/dense_59/bias/v*+
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_27653??
?

?
-__inference_sequential_14_layer_call_fn_27066
dense_56_input!
dense_56_kernel:"@
dense_56_bias:@"
dense_57_kernel:	@?
dense_57_bias:	?#
dense_58_kernel:
??
dense_58_bias:	?"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_kerneldense_56_biasdense_57_kerneldense_57_biasdense_58_kerneldense_58_biasdense_59_kerneldense_59_bias*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_27055o
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
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input
?
?
(__inference_dense_57_layer_call_fn_27388

inputs"
dense_57_kernel:	@?
dense_57_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_57_kerneldense_57_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_27021p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
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
?
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27055

inputs*
dense_56_dense_56_kernel:"@$
dense_56_dense_56_bias:@+
dense_57_dense_57_kernel:	@?%
dense_57_dense_57_bias:	?,
dense_58_dense_58_kernel:
??%
dense_58_dense_58_bias:	?+
dense_59_dense_59_kernel:	?$
dense_59_dense_59_bias:
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_dense_56_kerneldense_56_dense_56_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_27006?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_dense_57_kerneldense_57_dense_57_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_27021?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_dense_58_kerneldense_58_dense_58_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_27036?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_dense_59_kerneldense_59_dense_59_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_59_layer_call_and_return_conditional_losses_27050x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
(__inference_dense_59_layer_call_fn_27424

inputs"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_59_kerneldense_59_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_59_layer_call_and_return_conditional_losses_27050o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_56_layer_call_and_return_conditional_losses_27006

inputs7
%matmul_readvariableop_dense_56_kernel:"@2
$biasadd_readvariableop_dense_56_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_56_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_56_bias*
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
?
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27172

inputs*
dense_56_dense_56_kernel:"@$
dense_56_dense_56_bias:@+
dense_57_dense_57_kernel:	@?%
dense_57_dense_57_bias:	?,
dense_58_dense_58_kernel:
??%
dense_58_dense_58_bias:	?+
dense_59_dense_59_kernel:	?$
dense_59_dense_59_bias:
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_dense_56_kerneldense_56_dense_56_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_27006?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_dense_57_kerneldense_57_dense_57_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_27021?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_dense_58_kerneldense_58_dense_58_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_27036?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_dense_59_kerneldense_59_dense_59_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_59_layer_call_and_return_conditional_losses_27050x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?$
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27317

inputs@
.dense_56_matmul_readvariableop_dense_56_kernel:"@;
-dense_56_biasadd_readvariableop_dense_56_bias:@A
.dense_57_matmul_readvariableop_dense_57_kernel:	@?<
-dense_57_biasadd_readvariableop_dense_57_bias:	?B
.dense_58_matmul_readvariableop_dense_58_kernel:
??<
-dense_58_biasadd_readvariableop_dense_58_bias:	?A
.dense_59_matmul_readvariableop_dense_59_kernel:	?;
-dense_59_biasadd_readvariableop_dense_59_bias:
identity??dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?
dense_56/MatMul/ReadVariableOpReadVariableOp.dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:"@*
dtype0{
dense_56/MatMulMatMulinputs&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_56/BiasAdd/ReadVariableOpReadVariableOp-dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:@*
dtype0?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_57/MatMul/ReadVariableOpReadVariableOp.dense_57_matmul_readvariableop_dense_57_kernel*
_output_shapes
:	@?*
dtype0?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_57/BiasAdd/ReadVariableOpReadVariableOp-dense_57_biasadd_readvariableop_dense_57_bias*
_output_shapes	
:?*
dtype0?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_58/MatMul/ReadVariableOpReadVariableOp.dense_58_matmul_readvariableop_dense_58_kernel* 
_output_shapes
:
??*
dtype0?
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_58/BiasAdd/ReadVariableOpReadVariableOp-dense_58_biasadd_readvariableop_dense_58_bias*
_output_shapes	
:?*
dtype0?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_59/MatMul/ReadVariableOpReadVariableOp.dense_59_matmul_readvariableop_dense_59_kernel*
_output_shapes
:	?*
dtype0?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_59/BiasAdd/ReadVariableOpReadVariableOp-dense_59_biasadd_readvariableop_dense_59_bias*
_output_shapes
:*
dtype0?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
-__inference_sequential_14_layer_call_fn_27273

inputs!
dense_56_kernel:"@
dense_56_bias:@"
dense_57_kernel:	@?
dense_57_bias:	?#
dense_58_kernel:
??
dense_58_bias:	?"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_kerneldense_56_biasdense_57_kerneldense_57_biasdense_58_kerneldense_58_biasdense_59_kerneldense_59_bias*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_27055o
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
?
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27260
dense_56_input*
dense_56_dense_56_kernel:"@$
dense_56_dense_56_bias:@+
dense_57_dense_57_kernel:	@?%
dense_57_dense_57_bias:	?,
dense_58_dense_58_kernel:
??%
dense_58_dense_58_bias:	?+
dense_59_dense_59_kernel:	?$
dense_59_dense_59_bias:
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_dense_56_kerneldense_56_dense_56_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_27006?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_dense_57_kerneldense_57_dense_57_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_27021?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_dense_58_kerneldense_58_dense_58_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_27036?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_dense_59_kerneldense_59_dense_59_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_59_layer_call_and_return_conditional_losses_27050x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input
?
?
(__inference_dense_58_layer_call_fn_27406

inputs#
dense_58_kernel:
??
dense_58_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_58_kerneldense_58_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_27036p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_27653
file_prefix2
 assignvariableop_dense_56_kernel:"@.
 assignvariableop_1_dense_56_bias:@5
"assignvariableop_2_dense_57_kernel:	@?/
 assignvariableop_3_dense_57_bias:	?6
"assignvariableop_4_dense_58_kernel:
??/
 assignvariableop_5_dense_58_bias:	?5
"assignvariableop_6_dense_59_kernel:	?.
 assignvariableop_7_dense_59_bias:2
(assignvariableop_8_training_28_adam_iter:	 4
*assignvariableop_9_training_28_adam_beta_1: 5
+assignvariableop_10_training_28_adam_beta_2: 4
*assignvariableop_11_training_28_adam_decay: <
2assignvariableop_12_training_28_adam_learning_rate: &
assignvariableop_13_total_48: &
assignvariableop_14_count_48: H
6assignvariableop_15_training_28_adam_dense_56_kernel_m:"@B
4assignvariableop_16_training_28_adam_dense_56_bias_m:@I
6assignvariableop_17_training_28_adam_dense_57_kernel_m:	@?C
4assignvariableop_18_training_28_adam_dense_57_bias_m:	?J
6assignvariableop_19_training_28_adam_dense_58_kernel_m:
??C
4assignvariableop_20_training_28_adam_dense_58_bias_m:	?I
6assignvariableop_21_training_28_adam_dense_59_kernel_m:	?B
4assignvariableop_22_training_28_adam_dense_59_bias_m:H
6assignvariableop_23_training_28_adam_dense_56_kernel_v:"@B
4assignvariableop_24_training_28_adam_dense_56_bias_v:@I
6assignvariableop_25_training_28_adam_dense_57_kernel_v:	@?C
4assignvariableop_26_training_28_adam_dense_57_bias_v:	?J
6assignvariableop_27_training_28_adam_dense_58_kernel_v:
??C
4assignvariableop_28_training_28_adam_dense_58_bias_v:	?I
6assignvariableop_29_training_28_adam_dense_59_kernel_v:	?B
4assignvariableop_30_training_28_adam_dense_59_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_56_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_56_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_57_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_57_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_58_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_58_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_59_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_59_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_training_28_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_training_28_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_training_28_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_training_28_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp2assignvariableop_12_training_28_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_48Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_48Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp6assignvariableop_15_training_28_adam_dense_56_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_training_28_adam_dense_56_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_training_28_adam_dense_57_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_training_28_adam_dense_57_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_training_28_adam_dense_58_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_training_28_adam_dense_58_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_training_28_adam_dense_59_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_training_28_adam_dense_59_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_training_28_adam_dense_56_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_training_28_adam_dense_56_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_training_28_adam_dense_57_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_training_28_adam_dense_57_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_training_28_adam_dense_58_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_training_28_adam_dense_58_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_training_28_adam_dense_59_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_training_28_adam_dense_59_bias_vIdentity_30:output:0"/device:CPU:0*
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
?

?
-__inference_sequential_14_layer_call_fn_27228
dense_56_input!
dense_56_kernel:"@
dense_56_bias:@"
dense_57_kernel:	@?
dense_57_bias:	?#
dense_58_kernel:
??
dense_58_bias:	?"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_kerneldense_56_biasdense_57_kerneldense_57_biasdense_58_kerneldense_58_biasdense_59_kerneldense_59_bias*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_27172o
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
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input
?
?
(__inference_dense_56_layer_call_fn_27370

inputs!
dense_56_kernel:"@
dense_56_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_kerneldense_56_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_27006o
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
?

?
C__inference_dense_58_layer_call_and_return_conditional_losses_27417

inputs9
%matmul_readvariableop_dense_58_kernel:
??3
$biasadd_readvariableop_dense_58_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp}
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_58_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_58_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_14_layer_call_fn_27286

inputs!
dense_56_kernel:"@
dense_56_bias:@"
dense_57_kernel:	@?
dense_57_bias:	?#
dense_58_kernel:
??
dense_58_bias:	?"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_kerneldense_56_biasdense_57_kerneldense_57_biasdense_58_kerneldense_58_biasdense_59_kerneldense_59_bias*
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_27172o
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
?
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27244
dense_56_input*
dense_56_dense_56_kernel:"@$
dense_56_dense_56_bias:@+
dense_57_dense_57_kernel:	@?%
dense_57_dense_57_bias:	?,
dense_58_dense_58_kernel:
??%
dense_58_dense_58_bias:	?+
dense_59_dense_59_kernel:	?$
dense_59_dense_59_bias:
identity?? dense_56/StatefulPartitionedCall? dense_57/StatefulPartitionedCall? dense_58/StatefulPartitionedCall? dense_59/StatefulPartitionedCall?
 dense_56/StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_dense_56_kerneldense_56_dense_56_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_56_layer_call_and_return_conditional_losses_27006?
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_dense_57_kerneldense_57_dense_57_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_57_layer_call_and_return_conditional_losses_27021?
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_dense_58_kerneldense_58_dense_58_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_58_layer_call_and_return_conditional_losses_27036?
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_dense_59_kerneldense_59_dense_59_bias*
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
GPU 2J 8? *L
fGRE
C__inference_dense_59_layer_call_and_return_conditional_losses_27050x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall!^dense_59/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input
?

?
C__inference_dense_57_layer_call_and_return_conditional_losses_27021

inputs8
%matmul_readvariableop_dense_57_kernel:	@?3
$biasadd_readvariableop_dense_57_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_57_kernel*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_57_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
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

?
C__inference_dense_57_layer_call_and_return_conditional_losses_27399

inputs8
%matmul_readvariableop_dense_57_kernel:	@?3
$biasadd_readvariableop_dense_57_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_57_kernel*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_57_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
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
?G
?
__inference__traced_save_27550
file_prefix.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop4
0savev2_training_28_adam_iter_read_readvariableop	6
2savev2_training_28_adam_beta_1_read_readvariableop6
2savev2_training_28_adam_beta_2_read_readvariableop5
1savev2_training_28_adam_decay_read_readvariableop=
9savev2_training_28_adam_learning_rate_read_readvariableop'
#savev2_total_48_read_readvariableop'
#savev2_count_48_read_readvariableopA
=savev2_training_28_adam_dense_56_kernel_m_read_readvariableop?
;savev2_training_28_adam_dense_56_bias_m_read_readvariableopA
=savev2_training_28_adam_dense_57_kernel_m_read_readvariableop?
;savev2_training_28_adam_dense_57_bias_m_read_readvariableopA
=savev2_training_28_adam_dense_58_kernel_m_read_readvariableop?
;savev2_training_28_adam_dense_58_bias_m_read_readvariableopA
=savev2_training_28_adam_dense_59_kernel_m_read_readvariableop?
;savev2_training_28_adam_dense_59_bias_m_read_readvariableopA
=savev2_training_28_adam_dense_56_kernel_v_read_readvariableop?
;savev2_training_28_adam_dense_56_bias_v_read_readvariableopA
=savev2_training_28_adam_dense_57_kernel_v_read_readvariableop?
;savev2_training_28_adam_dense_57_bias_v_read_readvariableopA
=savev2_training_28_adam_dense_58_kernel_v_read_readvariableop?
;savev2_training_28_adam_dense_58_bias_v_read_readvariableopA
=savev2_training_28_adam_dense_59_kernel_v_read_readvariableop?
;savev2_training_28_adam_dense_59_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop0savev2_training_28_adam_iter_read_readvariableop2savev2_training_28_adam_beta_1_read_readvariableop2savev2_training_28_adam_beta_2_read_readvariableop1savev2_training_28_adam_decay_read_readvariableop9savev2_training_28_adam_learning_rate_read_readvariableop#savev2_total_48_read_readvariableop#savev2_count_48_read_readvariableop=savev2_training_28_adam_dense_56_kernel_m_read_readvariableop;savev2_training_28_adam_dense_56_bias_m_read_readvariableop=savev2_training_28_adam_dense_57_kernel_m_read_readvariableop;savev2_training_28_adam_dense_57_bias_m_read_readvariableop=savev2_training_28_adam_dense_58_kernel_m_read_readvariableop;savev2_training_28_adam_dense_58_bias_m_read_readvariableop=savev2_training_28_adam_dense_59_kernel_m_read_readvariableop;savev2_training_28_adam_dense_59_bias_m_read_readvariableop=savev2_training_28_adam_dense_56_kernel_v_read_readvariableop;savev2_training_28_adam_dense_56_bias_v_read_readvariableop=savev2_training_28_adam_dense_57_kernel_v_read_readvariableop;savev2_training_28_adam_dense_57_bias_v_read_readvariableop=savev2_training_28_adam_dense_58_kernel_v_read_readvariableop;savev2_training_28_adam_dense_58_bias_v_read_readvariableop=savev2_training_28_adam_dense_59_kernel_v_read_readvariableop;savev2_training_28_adam_dense_59_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :"@:@:	@?:?:
??:?:	?:: : : : : : : :"@:@:	@?:?:
??:?:	?::"@:@:	@?:?:
??:?:	?:: 2(
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
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 
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
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:"@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
: 
?	
?
C__inference_dense_59_layer_call_and_return_conditional_losses_27434

inputs8
%matmul_readvariableop_dense_59_kernel:	?2
$biasadd_readvariableop_dense_59_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_59_kernel*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_59_bias*
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
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_59_layer_call_and_return_conditional_losses_27050

inputs8
%matmul_readvariableop_dense_59_kernel:	?2
$biasadd_readvariableop_dense_59_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_59_kernel*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_59_bias*
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
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27348

inputs@
.dense_56_matmul_readvariableop_dense_56_kernel:"@;
-dense_56_biasadd_readvariableop_dense_56_bias:@A
.dense_57_matmul_readvariableop_dense_57_kernel:	@?<
-dense_57_biasadd_readvariableop_dense_57_bias:	?B
.dense_58_matmul_readvariableop_dense_58_kernel:
??<
-dense_58_biasadd_readvariableop_dense_58_bias:	?A
.dense_59_matmul_readvariableop_dense_59_kernel:	?;
-dense_59_biasadd_readvariableop_dense_59_bias:
identity??dense_56/BiasAdd/ReadVariableOp?dense_56/MatMul/ReadVariableOp?dense_57/BiasAdd/ReadVariableOp?dense_57/MatMul/ReadVariableOp?dense_58/BiasAdd/ReadVariableOp?dense_58/MatMul/ReadVariableOp?dense_59/BiasAdd/ReadVariableOp?dense_59/MatMul/ReadVariableOp?
dense_56/MatMul/ReadVariableOpReadVariableOp.dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:"@*
dtype0{
dense_56/MatMulMatMulinputs&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_56/BiasAdd/ReadVariableOpReadVariableOp-dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:@*
dtype0?
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_57/MatMul/ReadVariableOpReadVariableOp.dense_57_matmul_readvariableop_dense_57_kernel*
_output_shapes
:	@?*
dtype0?
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_57/BiasAdd/ReadVariableOpReadVariableOp-dense_57_biasadd_readvariableop_dense_57_bias*
_output_shapes	
:?*
dtype0?
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_58/MatMul/ReadVariableOpReadVariableOp.dense_58_matmul_readvariableop_dense_58_kernel* 
_output_shapes
:
??*
dtype0?
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_58/BiasAdd/ReadVariableOpReadVariableOp-dense_58_biasadd_readvariableop_dense_58_bias*
_output_shapes	
:?*
dtype0?
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_59/MatMul/ReadVariableOpReadVariableOp.dense_59_matmul_readvariableop_dense_59_kernel*
_output_shapes
:	?*
dtype0?
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_59/BiasAdd/ReadVariableOpReadVariableOp-dense_59_biasadd_readvariableop_dense_59_bias*
_output_shapes
:*
dtype0?
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?-
?
 __inference__wrapped_model_26988
dense_56_inputN
<sequential_14_dense_56_matmul_readvariableop_dense_56_kernel:"@I
;sequential_14_dense_56_biasadd_readvariableop_dense_56_bias:@O
<sequential_14_dense_57_matmul_readvariableop_dense_57_kernel:	@?J
;sequential_14_dense_57_biasadd_readvariableop_dense_57_bias:	?P
<sequential_14_dense_58_matmul_readvariableop_dense_58_kernel:
??J
;sequential_14_dense_58_biasadd_readvariableop_dense_58_bias:	?O
<sequential_14_dense_59_matmul_readvariableop_dense_59_kernel:	?I
;sequential_14_dense_59_biasadd_readvariableop_dense_59_bias:
identity??-sequential_14/dense_56/BiasAdd/ReadVariableOp?,sequential_14/dense_56/MatMul/ReadVariableOp?-sequential_14/dense_57/BiasAdd/ReadVariableOp?,sequential_14/dense_57/MatMul/ReadVariableOp?-sequential_14/dense_58/BiasAdd/ReadVariableOp?,sequential_14/dense_58/MatMul/ReadVariableOp?-sequential_14/dense_59/BiasAdd/ReadVariableOp?,sequential_14/dense_59/MatMul/ReadVariableOp?
,sequential_14/dense_56/MatMul/ReadVariableOpReadVariableOp<sequential_14_dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:"@*
dtype0?
sequential_14/dense_56/MatMulMatMuldense_56_input4sequential_14/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_14/dense_56/BiasAdd/ReadVariableOpReadVariableOp;sequential_14_dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:@*
dtype0?
sequential_14/dense_56/BiasAddBiasAdd'sequential_14/dense_56/MatMul:product:05sequential_14/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_14/dense_56/ReluRelu'sequential_14/dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_14/dense_57/MatMul/ReadVariableOpReadVariableOp<sequential_14_dense_57_matmul_readvariableop_dense_57_kernel*
_output_shapes
:	@?*
dtype0?
sequential_14/dense_57/MatMulMatMul)sequential_14/dense_56/Relu:activations:04sequential_14/dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_14/dense_57/BiasAdd/ReadVariableOpReadVariableOp;sequential_14_dense_57_biasadd_readvariableop_dense_57_bias*
_output_shapes	
:?*
dtype0?
sequential_14/dense_57/BiasAddBiasAdd'sequential_14/dense_57/MatMul:product:05sequential_14/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_14/dense_57/ReluRelu'sequential_14/dense_57/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_14/dense_58/MatMul/ReadVariableOpReadVariableOp<sequential_14_dense_58_matmul_readvariableop_dense_58_kernel* 
_output_shapes
:
??*
dtype0?
sequential_14/dense_58/MatMulMatMul)sequential_14/dense_57/Relu:activations:04sequential_14/dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_14/dense_58/BiasAdd/ReadVariableOpReadVariableOp;sequential_14_dense_58_biasadd_readvariableop_dense_58_bias*
_output_shapes	
:?*
dtype0?
sequential_14/dense_58/BiasAddBiasAdd'sequential_14/dense_58/MatMul:product:05sequential_14/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_14/dense_58/ReluRelu'sequential_14/dense_58/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_14/dense_59/MatMul/ReadVariableOpReadVariableOp<sequential_14_dense_59_matmul_readvariableop_dense_59_kernel*
_output_shapes
:	?*
dtype0?
sequential_14/dense_59/MatMulMatMul)sequential_14/dense_58/Relu:activations:04sequential_14/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_14/dense_59/BiasAdd/ReadVariableOpReadVariableOp;sequential_14_dense_59_biasadd_readvariableop_dense_59_bias*
_output_shapes
:*
dtype0?
sequential_14/dense_59/BiasAddBiasAdd'sequential_14/dense_59/MatMul:product:05sequential_14/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_14/dense_59/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^sequential_14/dense_56/BiasAdd/ReadVariableOp-^sequential_14/dense_56/MatMul/ReadVariableOp.^sequential_14/dense_57/BiasAdd/ReadVariableOp-^sequential_14/dense_57/MatMul/ReadVariableOp.^sequential_14/dense_58/BiasAdd/ReadVariableOp-^sequential_14/dense_58/MatMul/ReadVariableOp.^sequential_14/dense_59/BiasAdd/ReadVariableOp-^sequential_14/dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2^
-sequential_14/dense_56/BiasAdd/ReadVariableOp-sequential_14/dense_56/BiasAdd/ReadVariableOp2\
,sequential_14/dense_56/MatMul/ReadVariableOp,sequential_14/dense_56/MatMul/ReadVariableOp2^
-sequential_14/dense_57/BiasAdd/ReadVariableOp-sequential_14/dense_57/BiasAdd/ReadVariableOp2\
,sequential_14/dense_57/MatMul/ReadVariableOp,sequential_14/dense_57/MatMul/ReadVariableOp2^
-sequential_14/dense_58/BiasAdd/ReadVariableOp-sequential_14/dense_58/BiasAdd/ReadVariableOp2\
,sequential_14/dense_58/MatMul/ReadVariableOp,sequential_14/dense_58/MatMul/ReadVariableOp2^
-sequential_14/dense_59/BiasAdd/ReadVariableOp-sequential_14/dense_59/BiasAdd/ReadVariableOp2\
,sequential_14/dense_59/MatMul/ReadVariableOp,sequential_14/dense_59/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input
?

?
C__inference_dense_58_layer_call_and_return_conditional_losses_27036

inputs9
%matmul_readvariableop_dense_58_kernel:
??3
$biasadd_readvariableop_dense_58_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp}
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_58_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_58_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_56_layer_call_and_return_conditional_losses_27381

inputs7
%matmul_readvariableop_dense_56_kernel:"@2
$biasadd_readvariableop_dense_56_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_56_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_56_bias*
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
?
#__inference_signature_wrapper_27363
dense_56_input!
dense_56_kernel:"@
dense_56_bias:@"
dense_57_kernel:	@?
dense_57_bias:	?#
dense_58_kernel:
??
dense_58_bias:	?"
dense_59_kernel:	?
dense_59_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_56_inputdense_56_kerneldense_56_biasdense_57_kerneldense_57_biasdense_58_kerneldense_58_biasdense_59_kerneldense_59_bias*
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
GPU 2J 8? *)
f$R"
 __inference__wrapped_model_26988o
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
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_56_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_56_input7
 serving_default_dense_56_input:0?????????"<
dense_590
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?^
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
?2?
-__inference_sequential_14_layer_call_fn_27066
-__inference_sequential_14_layer_call_fn_27273
-__inference_sequential_14_layer_call_fn_27286
-__inference_sequential_14_layer_call_fn_27228?
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
H__inference_sequential_14_layer_call_and_return_conditional_losses_27317
H__inference_sequential_14_layer_call_and_return_conditional_losses_27348
H__inference_sequential_14_layer_call_and_return_conditional_losses_27244
H__inference_sequential_14_layer_call_and_return_conditional_losses_27260?
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
 __inference__wrapped_model_26988dense_56_input"?
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
!:"@2dense_56/kernel
:@2dense_56/bias
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
(__inference_dense_56_layer_call_fn_27370?
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
C__inference_dense_56_layer_call_and_return_conditional_losses_27381?
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
": 	@?2dense_57/kernel
:?2dense_57/bias
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
(__inference_dense_57_layer_call_fn_27388?
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
C__inference_dense_57_layer_call_and_return_conditional_losses_27399?
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
#:!
??2dense_58/kernel
:?2dense_58/bias
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
(__inference_dense_58_layer_call_fn_27406?
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
C__inference_dense_58_layer_call_and_return_conditional_losses_27417?
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
": 	?2dense_59/kernel
:2dense_59/bias
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
(__inference_dense_59_layer_call_fn_27424?
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
C__inference_dense_59_layer_call_and_return_conditional_losses_27434?
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
:	 (2training_28/Adam/iter
!: (2training_28/Adam/beta_1
!: (2training_28/Adam/beta_2
 : (2training_28/Adam/decay
(:& (2training_28/Adam/learning_rate
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
#__inference_signature_wrapper_27363dense_56_input"?
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
:  (2total_48
:  (2count_48
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
2:0"@2"training_28/Adam/dense_56/kernel/m
,:*@2 training_28/Adam/dense_56/bias/m
3:1	@?2"training_28/Adam/dense_57/kernel/m
-:+?2 training_28/Adam/dense_57/bias/m
4:2
??2"training_28/Adam/dense_58/kernel/m
-:+?2 training_28/Adam/dense_58/bias/m
3:1	?2"training_28/Adam/dense_59/kernel/m
,:*2 training_28/Adam/dense_59/bias/m
2:0"@2"training_28/Adam/dense_56/kernel/v
,:*@2 training_28/Adam/dense_56/bias/v
3:1	@?2"training_28/Adam/dense_57/kernel/v
-:+?2 training_28/Adam/dense_57/bias/v
4:2
??2"training_28/Adam/dense_58/kernel/v
-:+?2 training_28/Adam/dense_58/bias/v
3:1	?2"training_28/Adam/dense_59/kernel/v
,:*2 training_28/Adam/dense_59/bias/v?
 __inference__wrapped_model_26988x&'7?4
-?*
(?%
dense_56_input?????????"
? "3?0
.
dense_59"?
dense_59??????????
C__inference_dense_56_layer_call_and_return_conditional_losses_27381\/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????@
? {
(__inference_dense_56_layer_call_fn_27370O/?,
%?"
 ?
inputs?????????"
? "??????????@?
C__inference_dense_57_layer_call_and_return_conditional_losses_27399]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? |
(__inference_dense_57_layer_call_fn_27388P/?,
%?"
 ?
inputs?????????@
? "????????????
C__inference_dense_58_layer_call_and_return_conditional_losses_27417^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_58_layer_call_fn_27406Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_59_layer_call_and_return_conditional_losses_27434]&'0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_59_layer_call_fn_27424P&'0?-
&?#
!?
inputs??????????
? "???????????
H__inference_sequential_14_layer_call_and_return_conditional_losses_27244r&'??<
5?2
(?%
dense_56_input?????????"
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27260r&'??<
5?2
(?%
dense_56_input?????????"
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_14_layer_call_and_return_conditional_losses_27317j&'7?4
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
H__inference_sequential_14_layer_call_and_return_conditional_losses_27348j&'7?4
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
-__inference_sequential_14_layer_call_fn_27066e&'??<
5?2
(?%
dense_56_input?????????"
p 

 
? "???????????
-__inference_sequential_14_layer_call_fn_27228e&'??<
5?2
(?%
dense_56_input?????????"
p

 
? "???????????
-__inference_sequential_14_layer_call_fn_27273]&'7?4
-?*
 ?
inputs?????????"
p 

 
? "???????????
-__inference_sequential_14_layer_call_fn_27286]&'7?4
-?*
 ?
inputs?????????"
p

 
? "???????????
#__inference_signature_wrapper_27363?&'I?F
? 
??<
:
dense_56_input(?%
dense_56_input?????????""3?0
.
dense_59"?
dense_59?????????