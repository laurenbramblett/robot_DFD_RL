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
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@* 
shared_namedense_52/kernel
s
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes

:"@*
dtype0
r
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_52/bias
k
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes
:@*
dtype0
{
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?* 
shared_namedense_53/kernel
t
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
_output_shapes
:	@?*
dtype0
s
dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_53/bias
l
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes	
:?*
dtype0
|
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_54/kernel
u
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel* 
_output_shapes
:
??*
dtype0
s
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_54/bias
l
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes	
:?*
dtype0
{
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_55/kernel
t
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes
:	?*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:*
dtype0
~
training_26/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *&
shared_nametraining_26/Adam/iter
w
)training_26/Adam/iter/Read/ReadVariableOpReadVariableOptraining_26/Adam/iter*
_output_shapes
: *
dtype0	
?
training_26/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_26/Adam/beta_1
{
+training_26/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_26/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_26/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nametraining_26/Adam/beta_2
{
+training_26/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_26/Adam/beta_2*
_output_shapes
: *
dtype0
?
training_26/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_26/Adam/decay
y
*training_26/Adam/decay/Read/ReadVariableOpReadVariableOptraining_26/Adam/decay*
_output_shapes
: *
dtype0
?
training_26/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name training_26/Adam/learning_rate
?
2training_26/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_26/Adam/learning_rate*
_output_shapes
: *
dtype0
d
total_47VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_47
]
total_47/Read/ReadVariableOpReadVariableOptotal_47*
_output_shapes
: *
dtype0
d
count_47VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_47
]
count_47/Read/ReadVariableOpReadVariableOpcount_47*
_output_shapes
: *
dtype0
?
"training_26/Adam/dense_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*3
shared_name$"training_26/Adam/dense_52/kernel/m
?
6training_26/Adam/dense_52/kernel/m/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_52/kernel/m*
_output_shapes

:"@*
dtype0
?
 training_26/Adam/dense_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_26/Adam/dense_52/bias/m
?
4training_26/Adam/dense_52/bias/m/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_52/bias/m*
_output_shapes
:@*
dtype0
?
"training_26/Adam/dense_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"training_26/Adam/dense_53/kernel/m
?
6training_26/Adam/dense_53/kernel/m/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_53/kernel/m*
_output_shapes
:	@?*
dtype0
?
 training_26/Adam/dense_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_26/Adam/dense_53/bias/m
?
4training_26/Adam/dense_53/bias/m/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_53/bias/m*
_output_shapes	
:?*
dtype0
?
"training_26/Adam/dense_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"training_26/Adam/dense_54/kernel/m
?
6training_26/Adam/dense_54/kernel/m/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_54/kernel/m* 
_output_shapes
:
??*
dtype0
?
 training_26/Adam/dense_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_26/Adam/dense_54/bias/m
?
4training_26/Adam/dense_54/bias/m/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_54/bias/m*
_output_shapes	
:?*
dtype0
?
"training_26/Adam/dense_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"training_26/Adam/dense_55/kernel/m
?
6training_26/Adam/dense_55/kernel/m/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_55/kernel/m*
_output_shapes
:	?*
dtype0
?
 training_26/Adam/dense_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_26/Adam/dense_55/bias/m
?
4training_26/Adam/dense_55/bias/m/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_55/bias/m*
_output_shapes
:*
dtype0
?
"training_26/Adam/dense_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"@*3
shared_name$"training_26/Adam/dense_52/kernel/v
?
6training_26/Adam/dense_52/kernel/v/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_52/kernel/v*
_output_shapes

:"@*
dtype0
?
 training_26/Adam/dense_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" training_26/Adam/dense_52/bias/v
?
4training_26/Adam/dense_52/bias/v/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_52/bias/v*
_output_shapes
:@*
dtype0
?
"training_26/Adam/dense_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@?*3
shared_name$"training_26/Adam/dense_53/kernel/v
?
6training_26/Adam/dense_53/kernel/v/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_53/kernel/v*
_output_shapes
:	@?*
dtype0
?
 training_26/Adam/dense_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_26/Adam/dense_53/bias/v
?
4training_26/Adam/dense_53/bias/v/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_53/bias/v*
_output_shapes	
:?*
dtype0
?
"training_26/Adam/dense_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"training_26/Adam/dense_54/kernel/v
?
6training_26/Adam/dense_54/kernel/v/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_54/kernel/v* 
_output_shapes
:
??*
dtype0
?
 training_26/Adam/dense_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" training_26/Adam/dense_54/bias/v
?
4training_26/Adam/dense_54/bias/v/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_54/bias/v*
_output_shapes	
:?*
dtype0
?
"training_26/Adam/dense_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"training_26/Adam/dense_55/kernel/v
?
6training_26/Adam/dense_55/kernel/v/Read/ReadVariableOpReadVariableOp"training_26/Adam/dense_55/kernel/v*
_output_shapes
:	?*
dtype0
?
 training_26/Adam/dense_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training_26/Adam/dense_55/bias/v
?
4training_26/Adam/dense_55/bias/v/Read/ReadVariableOpReadVariableOp training_26/Adam/dense_55/bias/v*
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
VARIABLE_VALUEdense_52/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_52/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_53/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_53/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_54/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_54/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_55/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_55/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEtraining_26/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_26/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEtraining_26/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEtraining_26/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEtraining_26/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEtotal_474keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_474keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

N0
O1*

Q	variables*
??
VARIABLE_VALUE"training_26/Adam/dense_52/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_52/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_53/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_53/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_54/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_54/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_55/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_55/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_52/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_52/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_53/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_53/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_54/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_54/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"training_26/Adam/dense_55/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE training_26/Adam/dense_55/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_dense_52_inputPlaceholder*'
_output_shapes
:?????????"*
dtype0*
shape:?????????"
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_52_inputdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/bias*
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
#__inference_signature_wrapper_26266
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOp!dense_53/bias/Read/ReadVariableOp#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp)training_26/Adam/iter/Read/ReadVariableOp+training_26/Adam/beta_1/Read/ReadVariableOp+training_26/Adam/beta_2/Read/ReadVariableOp*training_26/Adam/decay/Read/ReadVariableOp2training_26/Adam/learning_rate/Read/ReadVariableOptotal_47/Read/ReadVariableOpcount_47/Read/ReadVariableOp6training_26/Adam/dense_52/kernel/m/Read/ReadVariableOp4training_26/Adam/dense_52/bias/m/Read/ReadVariableOp6training_26/Adam/dense_53/kernel/m/Read/ReadVariableOp4training_26/Adam/dense_53/bias/m/Read/ReadVariableOp6training_26/Adam/dense_54/kernel/m/Read/ReadVariableOp4training_26/Adam/dense_54/bias/m/Read/ReadVariableOp6training_26/Adam/dense_55/kernel/m/Read/ReadVariableOp4training_26/Adam/dense_55/bias/m/Read/ReadVariableOp6training_26/Adam/dense_52/kernel/v/Read/ReadVariableOp4training_26/Adam/dense_52/bias/v/Read/ReadVariableOp6training_26/Adam/dense_53/kernel/v/Read/ReadVariableOp4training_26/Adam/dense_53/bias/v/Read/ReadVariableOp6training_26/Adam/dense_54/kernel/v/Read/ReadVariableOp4training_26/Adam/dense_54/bias/v/Read/ReadVariableOp6training_26/Adam/dense_55/kernel/v/Read/ReadVariableOp4training_26/Adam/dense_55/bias/v/Read/ReadVariableOpConst*,
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
__inference__traced_save_26453
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/biastraining_26/Adam/itertraining_26/Adam/beta_1training_26/Adam/beta_2training_26/Adam/decaytraining_26/Adam/learning_ratetotal_47count_47"training_26/Adam/dense_52/kernel/m training_26/Adam/dense_52/bias/m"training_26/Adam/dense_53/kernel/m training_26/Adam/dense_53/bias/m"training_26/Adam/dense_54/kernel/m training_26/Adam/dense_54/bias/m"training_26/Adam/dense_55/kernel/m training_26/Adam/dense_55/bias/m"training_26/Adam/dense_52/kernel/v training_26/Adam/dense_52/bias/v"training_26/Adam/dense_53/kernel/v training_26/Adam/dense_53/bias/v"training_26/Adam/dense_54/kernel/v training_26/Adam/dense_54/bias/v"training_26/Adam/dense_55/kernel/v training_26/Adam/dense_55/bias/v*+
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
!__inference__traced_restore_26556??
?
?
(__inference_dense_54_layer_call_fn_26309

inputs#
dense_54_kernel:
??
dense_54_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_kerneldense_54_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_25939p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_55_layer_call_fn_26327

inputs"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_55_kerneldense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_25953o
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
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_54_layer_call_and_return_conditional_losses_26320

inputs9
%matmul_readvariableop_dense_54_kernel:
??3
$biasadd_readvariableop_dense_54_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp}
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_54_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_54_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_sequential_13_layer_call_fn_26189

inputs!
dense_52_kernel:"@
dense_52_bias:@"
dense_53_kernel:	@?
dense_53_bias:	?#
dense_54_kernel:
??
dense_54_bias:	?"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_kerneldense_52_biasdense_53_kerneldense_53_biasdense_54_kerneldense_54_biasdense_55_kerneldense_55_bias*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26075o
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
-__inference_sequential_13_layer_call_fn_25969
dense_52_input!
dense_52_kernel:"@
dense_52_bias:@"
dense_53_kernel:	@?
dense_53_bias:	?#
dense_54_kernel:
??
dense_54_bias:	?"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_kerneldense_52_biasdense_53_kerneldense_53_biasdense_54_kerneldense_54_biasdense_55_kerneldense_55_bias*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_25958o
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
_user_specified_namedense_52_input
??
?
!__inference__traced_restore_26556
file_prefix2
 assignvariableop_dense_52_kernel:"@.
 assignvariableop_1_dense_52_bias:@5
"assignvariableop_2_dense_53_kernel:	@?/
 assignvariableop_3_dense_53_bias:	?6
"assignvariableop_4_dense_54_kernel:
??/
 assignvariableop_5_dense_54_bias:	?5
"assignvariableop_6_dense_55_kernel:	?.
 assignvariableop_7_dense_55_bias:2
(assignvariableop_8_training_26_adam_iter:	 4
*assignvariableop_9_training_26_adam_beta_1: 5
+assignvariableop_10_training_26_adam_beta_2: 4
*assignvariableop_11_training_26_adam_decay: <
2assignvariableop_12_training_26_adam_learning_rate: &
assignvariableop_13_total_47: &
assignvariableop_14_count_47: H
6assignvariableop_15_training_26_adam_dense_52_kernel_m:"@B
4assignvariableop_16_training_26_adam_dense_52_bias_m:@I
6assignvariableop_17_training_26_adam_dense_53_kernel_m:	@?C
4assignvariableop_18_training_26_adam_dense_53_bias_m:	?J
6assignvariableop_19_training_26_adam_dense_54_kernel_m:
??C
4assignvariableop_20_training_26_adam_dense_54_bias_m:	?I
6assignvariableop_21_training_26_adam_dense_55_kernel_m:	?B
4assignvariableop_22_training_26_adam_dense_55_bias_m:H
6assignvariableop_23_training_26_adam_dense_52_kernel_v:"@B
4assignvariableop_24_training_26_adam_dense_52_bias_v:@I
6assignvariableop_25_training_26_adam_dense_53_kernel_v:	@?C
4assignvariableop_26_training_26_adam_dense_53_bias_v:	?J
6assignvariableop_27_training_26_adam_dense_54_kernel_v:
??C
4assignvariableop_28_training_26_adam_dense_54_bias_v:	?I
6assignvariableop_29_training_26_adam_dense_55_kernel_v:	?B
4assignvariableop_30_training_26_adam_dense_55_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_52_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_52_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_53_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_53_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_54_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_54_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_55_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_55_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_training_26_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_training_26_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp+assignvariableop_10_training_26_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_training_26_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp2assignvariableop_12_training_26_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_47Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_47Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp6assignvariableop_15_training_26_adam_dense_52_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_training_26_adam_dense_52_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_training_26_adam_dense_53_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_training_26_adam_dense_53_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_training_26_adam_dense_54_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_training_26_adam_dense_54_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_training_26_adam_dense_55_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp4assignvariableop_22_training_26_adam_dense_55_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_training_26_adam_dense_52_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_training_26_adam_dense_52_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_training_26_adam_dense_53_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_training_26_adam_dense_53_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_training_26_adam_dense_54_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_training_26_adam_dense_54_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_training_26_adam_dense_55_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp4assignvariableop_30_training_26_adam_dense_55_bias_vIdentity_30:output:0"/device:CPU:0*
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
C__inference_dense_53_layer_call_and_return_conditional_losses_26302

inputs8
%matmul_readvariableop_dense_53_kernel:	@?3
$biasadd_readvariableop_dense_53_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_53_kernel*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_53_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
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
-__inference_sequential_13_layer_call_fn_26176

inputs!
dense_52_kernel:"@
dense_52_bias:@"
dense_53_kernel:	@?
dense_53_bias:	?#
dense_54_kernel:
??
dense_54_bias:	?"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_kerneldense_52_biasdense_53_kerneldense_53_biasdense_54_kerneldense_54_biasdense_55_kerneldense_55_bias*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_25958o
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26163
dense_52_input*
dense_52_dense_52_kernel:"@$
dense_52_dense_52_bias:@+
dense_53_dense_53_kernel:	@?%
dense_53_dense_53_bias:	?,
dense_54_dense_54_kernel:
??%
dense_54_dense_54_bias:	?+
dense_55_dense_55_kernel:	?$
dense_55_dense_55_bias:
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_dense_52_kerneldense_52_dense_52_bias*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_25909?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_dense_53_kerneldense_53_dense_53_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_25924?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_dense_54_kerneldense_54_dense_54_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_25939?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_25953x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_52_input
?$
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_26251

inputs@
.dense_52_matmul_readvariableop_dense_52_kernel:"@;
-dense_52_biasadd_readvariableop_dense_52_bias:@A
.dense_53_matmul_readvariableop_dense_53_kernel:	@?<
-dense_53_biasadd_readvariableop_dense_53_bias:	?B
.dense_54_matmul_readvariableop_dense_54_kernel:
??<
-dense_54_biasadd_readvariableop_dense_54_bias:	?A
.dense_55_matmul_readvariableop_dense_55_kernel:	?;
-dense_55_biasadd_readvariableop_dense_55_bias:
identity??dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?
dense_52/MatMul/ReadVariableOpReadVariableOp.dense_52_matmul_readvariableop_dense_52_kernel*
_output_shapes

:"@*
dtype0{
dense_52/MatMulMatMulinputs&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_52/BiasAdd/ReadVariableOpReadVariableOp-dense_52_biasadd_readvariableop_dense_52_bias*
_output_shapes
:@*
dtype0?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_53/MatMul/ReadVariableOpReadVariableOp.dense_53_matmul_readvariableop_dense_53_kernel*
_output_shapes
:	@?*
dtype0?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp-dense_53_biasadd_readvariableop_dense_53_bias*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_54/MatMul/ReadVariableOpReadVariableOp.dense_54_matmul_readvariableop_dense_54_kernel* 
_output_shapes
:
??*
dtype0?
dense_54/MatMulMatMuldense_53/Relu:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_54/BiasAdd/ReadVariableOpReadVariableOp-dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes	
:?*
dtype0?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_55/MatMul/ReadVariableOpReadVariableOp.dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes
:	?*
dtype0?
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_55/BiasAdd/ReadVariableOpReadVariableOp-dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:*
dtype0?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_55/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?

?
C__inference_dense_52_layer_call_and_return_conditional_losses_26284

inputs7
%matmul_readvariableop_dense_52_kernel:"@2
$biasadd_readvariableop_dense_52_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_52_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_52_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_25939

inputs9
%matmul_readvariableop_dense_54_kernel:
??3
$biasadd_readvariableop_dense_54_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp}
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_54_kernel* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_54_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_26075

inputs*
dense_52_dense_52_kernel:"@$
dense_52_dense_52_bias:@+
dense_53_dense_53_kernel:	@?%
dense_53_dense_53_bias:	?,
dense_54_dense_54_kernel:
??%
dense_54_dense_54_bias:	?+
dense_55_dense_55_kernel:	?$
dense_55_dense_55_bias:
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_dense_52_kerneldense_52_dense_52_bias*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_25909?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_dense_53_kerneldense_53_dense_53_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_25924?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_dense_54_kerneldense_54_dense_54_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_25939?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_25953x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_26266
dense_52_input!
dense_52_kernel:"@
dense_52_bias:@"
dense_53_kernel:	@?
dense_53_bias:	?#
dense_54_kernel:
??
dense_54_bias:	?"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_kerneldense_52_biasdense_53_kerneldense_53_biasdense_54_kerneldense_54_biasdense_55_kerneldense_55_bias*
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
 __inference__wrapped_model_25891o
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
_user_specified_namedense_52_input
?
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_25958

inputs*
dense_52_dense_52_kernel:"@$
dense_52_dense_52_bias:@+
dense_53_dense_53_kernel:	@?%
dense_53_dense_53_bias:	?,
dense_54_dense_54_kernel:
??%
dense_54_dense_54_bias:	?+
dense_55_dense_55_kernel:	?$
dense_55_dense_55_bias:
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_dense_52_kerneldense_52_dense_52_bias*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_25909?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_dense_53_kerneldense_53_dense_53_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_25924?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_dense_54_kerneldense_54_dense_54_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_25939?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_25953x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?

?
-__inference_sequential_13_layer_call_fn_26131
dense_52_input!
dense_52_kernel:"@
dense_52_bias:@"
dense_53_kernel:	@?
dense_53_bias:	?#
dense_54_kernel:
??
dense_54_bias:	?"
dense_55_kernel:	?
dense_55_bias:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_kerneldense_52_biasdense_53_kerneldense_53_biasdense_54_kerneldense_54_biasdense_55_kerneldense_55_bias*
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26075o
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
_user_specified_namedense_52_input
?-
?
 __inference__wrapped_model_25891
dense_52_inputN
<sequential_13_dense_52_matmul_readvariableop_dense_52_kernel:"@I
;sequential_13_dense_52_biasadd_readvariableop_dense_52_bias:@O
<sequential_13_dense_53_matmul_readvariableop_dense_53_kernel:	@?J
;sequential_13_dense_53_biasadd_readvariableop_dense_53_bias:	?P
<sequential_13_dense_54_matmul_readvariableop_dense_54_kernel:
??J
;sequential_13_dense_54_biasadd_readvariableop_dense_54_bias:	?O
<sequential_13_dense_55_matmul_readvariableop_dense_55_kernel:	?I
;sequential_13_dense_55_biasadd_readvariableop_dense_55_bias:
identity??-sequential_13/dense_52/BiasAdd/ReadVariableOp?,sequential_13/dense_52/MatMul/ReadVariableOp?-sequential_13/dense_53/BiasAdd/ReadVariableOp?,sequential_13/dense_53/MatMul/ReadVariableOp?-sequential_13/dense_54/BiasAdd/ReadVariableOp?,sequential_13/dense_54/MatMul/ReadVariableOp?-sequential_13/dense_55/BiasAdd/ReadVariableOp?,sequential_13/dense_55/MatMul/ReadVariableOp?
,sequential_13/dense_52/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_52_matmul_readvariableop_dense_52_kernel*
_output_shapes

:"@*
dtype0?
sequential_13/dense_52/MatMulMatMuldense_52_input4sequential_13/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
-sequential_13/dense_52/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_52_biasadd_readvariableop_dense_52_bias*
_output_shapes
:@*
dtype0?
sequential_13/dense_52/BiasAddBiasAdd'sequential_13/dense_52/MatMul:product:05sequential_13/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@~
sequential_13/dense_52/ReluRelu'sequential_13/dense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
,sequential_13/dense_53/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_53_matmul_readvariableop_dense_53_kernel*
_output_shapes
:	@?*
dtype0?
sequential_13/dense_53/MatMulMatMul)sequential_13/dense_52/Relu:activations:04sequential_13/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_13/dense_53/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_53_biasadd_readvariableop_dense_53_bias*
_output_shapes	
:?*
dtype0?
sequential_13/dense_53/BiasAddBiasAdd'sequential_13/dense_53/MatMul:product:05sequential_13/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_13/dense_53/ReluRelu'sequential_13/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_13/dense_54/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_54_matmul_readvariableop_dense_54_kernel* 
_output_shapes
:
??*
dtype0?
sequential_13/dense_54/MatMulMatMul)sequential_13/dense_53/Relu:activations:04sequential_13/dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_13/dense_54/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes	
:?*
dtype0?
sequential_13/dense_54/BiasAddBiasAdd'sequential_13/dense_54/MatMul:product:05sequential_13/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_13/dense_54/ReluRelu'sequential_13/dense_54/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_13/dense_55/MatMul/ReadVariableOpReadVariableOp<sequential_13_dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes
:	?*
dtype0?
sequential_13/dense_55/MatMulMatMul)sequential_13/dense_54/Relu:activations:04sequential_13/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_13/dense_55/BiasAdd/ReadVariableOpReadVariableOp;sequential_13_dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:*
dtype0?
sequential_13/dense_55/BiasAddBiasAdd'sequential_13/dense_55/MatMul:product:05sequential_13/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
IdentityIdentity'sequential_13/dense_55/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^sequential_13/dense_52/BiasAdd/ReadVariableOp-^sequential_13/dense_52/MatMul/ReadVariableOp.^sequential_13/dense_53/BiasAdd/ReadVariableOp-^sequential_13/dense_53/MatMul/ReadVariableOp.^sequential_13/dense_54/BiasAdd/ReadVariableOp-^sequential_13/dense_54/MatMul/ReadVariableOp.^sequential_13/dense_55/BiasAdd/ReadVariableOp-^sequential_13/dense_55/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2^
-sequential_13/dense_52/BiasAdd/ReadVariableOp-sequential_13/dense_52/BiasAdd/ReadVariableOp2\
,sequential_13/dense_52/MatMul/ReadVariableOp,sequential_13/dense_52/MatMul/ReadVariableOp2^
-sequential_13/dense_53/BiasAdd/ReadVariableOp-sequential_13/dense_53/BiasAdd/ReadVariableOp2\
,sequential_13/dense_53/MatMul/ReadVariableOp,sequential_13/dense_53/MatMul/ReadVariableOp2^
-sequential_13/dense_54/BiasAdd/ReadVariableOp-sequential_13/dense_54/BiasAdd/ReadVariableOp2\
,sequential_13/dense_54/MatMul/ReadVariableOp,sequential_13/dense_54/MatMul/ReadVariableOp2^
-sequential_13/dense_55/BiasAdd/ReadVariableOp-sequential_13/dense_55/BiasAdd/ReadVariableOp2\
,sequential_13/dense_55/MatMul/ReadVariableOp,sequential_13/dense_55/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_52_input
?	
?
C__inference_dense_55_layer_call_and_return_conditional_losses_25953

inputs8
%matmul_readvariableop_dense_55_kernel:	?2
$biasadd_readvariableop_dense_55_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_55_kernel*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_55_bias*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_dense_52_layer_call_and_return_conditional_losses_25909

inputs7
%matmul_readvariableop_dense_52_kernel:"@2
$biasadd_readvariableop_dense_52_bias:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_52_kernel*
_output_shapes

:"@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_52_bias*
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
?G
?
__inference__traced_save_26453
file_prefix.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop,
(savev2_dense_53_bias_read_readvariableop.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop4
0savev2_training_26_adam_iter_read_readvariableop	6
2savev2_training_26_adam_beta_1_read_readvariableop6
2savev2_training_26_adam_beta_2_read_readvariableop5
1savev2_training_26_adam_decay_read_readvariableop=
9savev2_training_26_adam_learning_rate_read_readvariableop'
#savev2_total_47_read_readvariableop'
#savev2_count_47_read_readvariableopA
=savev2_training_26_adam_dense_52_kernel_m_read_readvariableop?
;savev2_training_26_adam_dense_52_bias_m_read_readvariableopA
=savev2_training_26_adam_dense_53_kernel_m_read_readvariableop?
;savev2_training_26_adam_dense_53_bias_m_read_readvariableopA
=savev2_training_26_adam_dense_54_kernel_m_read_readvariableop?
;savev2_training_26_adam_dense_54_bias_m_read_readvariableopA
=savev2_training_26_adam_dense_55_kernel_m_read_readvariableop?
;savev2_training_26_adam_dense_55_bias_m_read_readvariableopA
=savev2_training_26_adam_dense_52_kernel_v_read_readvariableop?
;savev2_training_26_adam_dense_52_bias_v_read_readvariableopA
=savev2_training_26_adam_dense_53_kernel_v_read_readvariableop?
;savev2_training_26_adam_dense_53_bias_v_read_readvariableopA
=savev2_training_26_adam_dense_54_kernel_v_read_readvariableop?
;savev2_training_26_adam_dense_54_bias_v_read_readvariableopA
=savev2_training_26_adam_dense_55_kernel_v_read_readvariableop?
;savev2_training_26_adam_dense_55_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableop*savev2_dense_53_kernel_read_readvariableop(savev2_dense_53_bias_read_readvariableop*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop0savev2_training_26_adam_iter_read_readvariableop2savev2_training_26_adam_beta_1_read_readvariableop2savev2_training_26_adam_beta_2_read_readvariableop1savev2_training_26_adam_decay_read_readvariableop9savev2_training_26_adam_learning_rate_read_readvariableop#savev2_total_47_read_readvariableop#savev2_count_47_read_readvariableop=savev2_training_26_adam_dense_52_kernel_m_read_readvariableop;savev2_training_26_adam_dense_52_bias_m_read_readvariableop=savev2_training_26_adam_dense_53_kernel_m_read_readvariableop;savev2_training_26_adam_dense_53_bias_m_read_readvariableop=savev2_training_26_adam_dense_54_kernel_m_read_readvariableop;savev2_training_26_adam_dense_54_bias_m_read_readvariableop=savev2_training_26_adam_dense_55_kernel_m_read_readvariableop;savev2_training_26_adam_dense_55_bias_m_read_readvariableop=savev2_training_26_adam_dense_52_kernel_v_read_readvariableop;savev2_training_26_adam_dense_52_bias_v_read_readvariableop=savev2_training_26_adam_dense_53_kernel_v_read_readvariableop;savev2_training_26_adam_dense_53_bias_v_read_readvariableop=savev2_training_26_adam_dense_54_kernel_v_read_readvariableop;savev2_training_26_adam_dense_54_bias_v_read_readvariableop=savev2_training_26_adam_dense_55_kernel_v_read_readvariableop;savev2_training_26_adam_dense_55_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :"@:@:	@?:?:
??:?:	?:: : : : : : : :"@:@:	@?:?:
??:?:	?::"@:@:	@?:?:
??:?:	?:: 2(
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
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 
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
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:"@: 

_output_shapes
:@:%!

_output_shapes
:	@?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
:: 

_output_shapes
: 
?$
?
H__inference_sequential_13_layer_call_and_return_conditional_losses_26220

inputs@
.dense_52_matmul_readvariableop_dense_52_kernel:"@;
-dense_52_biasadd_readvariableop_dense_52_bias:@A
.dense_53_matmul_readvariableop_dense_53_kernel:	@?<
-dense_53_biasadd_readvariableop_dense_53_bias:	?B
.dense_54_matmul_readvariableop_dense_54_kernel:
??<
-dense_54_biasadd_readvariableop_dense_54_bias:	?A
.dense_55_matmul_readvariableop_dense_55_kernel:	?;
-dense_55_biasadd_readvariableop_dense_55_bias:
identity??dense_52/BiasAdd/ReadVariableOp?dense_52/MatMul/ReadVariableOp?dense_53/BiasAdd/ReadVariableOp?dense_53/MatMul/ReadVariableOp?dense_54/BiasAdd/ReadVariableOp?dense_54/MatMul/ReadVariableOp?dense_55/BiasAdd/ReadVariableOp?dense_55/MatMul/ReadVariableOp?
dense_52/MatMul/ReadVariableOpReadVariableOp.dense_52_matmul_readvariableop_dense_52_kernel*
_output_shapes

:"@*
dtype0{
dense_52/MatMulMatMulinputs&dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_52/BiasAdd/ReadVariableOpReadVariableOp-dense_52_biasadd_readvariableop_dense_52_bias*
_output_shapes
:@*
dtype0?
dense_52/BiasAddBiasAdddense_52/MatMul:product:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_52/ReluReludense_52/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_53/MatMul/ReadVariableOpReadVariableOp.dense_53_matmul_readvariableop_dense_53_kernel*
_output_shapes
:	@?*
dtype0?
dense_53/MatMulMatMuldense_52/Relu:activations:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_53/BiasAdd/ReadVariableOpReadVariableOp-dense_53_biasadd_readvariableop_dense_53_bias*
_output_shapes	
:?*
dtype0?
dense_53/BiasAddBiasAdddense_53/MatMul:product:0'dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_53/ReluReludense_53/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_54/MatMul/ReadVariableOpReadVariableOp.dense_54_matmul_readvariableop_dense_54_kernel* 
_output_shapes
:
??*
dtype0?
dense_54/MatMulMatMuldense_53/Relu:activations:0&dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_54/BiasAdd/ReadVariableOpReadVariableOp-dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes	
:?*
dtype0?
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_55/MatMul/ReadVariableOpReadVariableOp.dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes
:	?*
dtype0?
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_55/BiasAdd/ReadVariableOpReadVariableOp-dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:*
dtype0?
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_55/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_52/BiasAdd/ReadVariableOp^dense_52/MatMul/ReadVariableOp ^dense_53/BiasAdd/ReadVariableOp^dense_53/MatMul/ReadVariableOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2B
dense_52/BiasAdd/ReadVariableOpdense_52/BiasAdd/ReadVariableOp2@
dense_52/MatMul/ReadVariableOpdense_52/MatMul/ReadVariableOp2B
dense_53/BiasAdd/ReadVariableOpdense_53/BiasAdd/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
(__inference_dense_53_layer_call_fn_26291

inputs"
dense_53_kernel:	@?
dense_53_bias:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_53_kerneldense_53_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_25924p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26147
dense_52_input*
dense_52_dense_52_kernel:"@$
dense_52_dense_52_bias:@+
dense_53_dense_53_kernel:	@?%
dense_53_dense_53_bias:	?,
dense_54_dense_54_kernel:
??%
dense_54_dense_54_bias:	?+
dense_55_dense_55_kernel:	?$
dense_55_dense_55_bias:
identity?? dense_52/StatefulPartitionedCall? dense_53/StatefulPartitionedCall? dense_54/StatefulPartitionedCall? dense_55/StatefulPartitionedCall?
 dense_52/StatefulPartitionedCallStatefulPartitionedCalldense_52_inputdense_52_dense_52_kerneldense_52_dense_52_bias*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_25909?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_dense_53_kerneldense_53_dense_53_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_53_layer_call_and_return_conditional_losses_25924?
 dense_54/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0dense_54_dense_54_kerneldense_54_dense_54_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_54_layer_call_and_return_conditional_losses_25939?
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_25953x
IdentityIdentity)dense_55/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????": : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall:W S
'
_output_shapes
:?????????"
(
_user_specified_namedense_52_input
?	
?
C__inference_dense_55_layer_call_and_return_conditional_losses_26337

inputs8
%matmul_readvariableop_dense_55_kernel:	?2
$biasadd_readvariableop_dense_55_bias:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_55_kernel*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_55_bias*
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
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_52_layer_call_fn_26273

inputs!
dense_52_kernel:"@
dense_52_bias:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_52_kerneldense_52_bias*
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
C__inference_dense_52_layer_call_and_return_conditional_losses_25909o
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
C__inference_dense_53_layer_call_and_return_conditional_losses_25924

inputs8
%matmul_readvariableop_dense_53_kernel:	@?3
$biasadd_readvariableop_dense_53_bias:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_53_kernel*
_output_shapes
:	@?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????x
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_53_bias*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
dense_52_input7
 serving_default_dense_52_input:0?????????"<
dense_550
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
-__inference_sequential_13_layer_call_fn_25969
-__inference_sequential_13_layer_call_fn_26176
-__inference_sequential_13_layer_call_fn_26189
-__inference_sequential_13_layer_call_fn_26131?
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26220
H__inference_sequential_13_layer_call_and_return_conditional_losses_26251
H__inference_sequential_13_layer_call_and_return_conditional_losses_26147
H__inference_sequential_13_layer_call_and_return_conditional_losses_26163?
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
 __inference__wrapped_model_25891dense_52_input"?
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
!:"@2dense_52/kernel
:@2dense_52/bias
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
(__inference_dense_52_layer_call_fn_26273?
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
C__inference_dense_52_layer_call_and_return_conditional_losses_26284?
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
": 	@?2dense_53/kernel
:?2dense_53/bias
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
(__inference_dense_53_layer_call_fn_26291?
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
C__inference_dense_53_layer_call_and_return_conditional_losses_26302?
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
??2dense_54/kernel
:?2dense_54/bias
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
(__inference_dense_54_layer_call_fn_26309?
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
C__inference_dense_54_layer_call_and_return_conditional_losses_26320?
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
": 	?2dense_55/kernel
:2dense_55/bias
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
(__inference_dense_55_layer_call_fn_26327?
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
C__inference_dense_55_layer_call_and_return_conditional_losses_26337?
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
:	 (2training_26/Adam/iter
!: (2training_26/Adam/beta_1
!: (2training_26/Adam/beta_2
 : (2training_26/Adam/decay
(:& (2training_26/Adam/learning_rate
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
#__inference_signature_wrapper_26266dense_52_input"?
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
:  (2total_47
:  (2count_47
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
2:0"@2"training_26/Adam/dense_52/kernel/m
,:*@2 training_26/Adam/dense_52/bias/m
3:1	@?2"training_26/Adam/dense_53/kernel/m
-:+?2 training_26/Adam/dense_53/bias/m
4:2
??2"training_26/Adam/dense_54/kernel/m
-:+?2 training_26/Adam/dense_54/bias/m
3:1	?2"training_26/Adam/dense_55/kernel/m
,:*2 training_26/Adam/dense_55/bias/m
2:0"@2"training_26/Adam/dense_52/kernel/v
,:*@2 training_26/Adam/dense_52/bias/v
3:1	@?2"training_26/Adam/dense_53/kernel/v
-:+?2 training_26/Adam/dense_53/bias/v
4:2
??2"training_26/Adam/dense_54/kernel/v
-:+?2 training_26/Adam/dense_54/bias/v
3:1	?2"training_26/Adam/dense_55/kernel/v
,:*2 training_26/Adam/dense_55/bias/v?
 __inference__wrapped_model_25891x&'7?4
-?*
(?%
dense_52_input?????????"
? "3?0
.
dense_55"?
dense_55??????????
C__inference_dense_52_layer_call_and_return_conditional_losses_26284\/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????@
? {
(__inference_dense_52_layer_call_fn_26273O/?,
%?"
 ?
inputs?????????"
? "??????????@?
C__inference_dense_53_layer_call_and_return_conditional_losses_26302]/?,
%?"
 ?
inputs?????????@
? "&?#
?
0??????????
? |
(__inference_dense_53_layer_call_fn_26291P/?,
%?"
 ?
inputs?????????@
? "????????????
C__inference_dense_54_layer_call_and_return_conditional_losses_26320^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_54_layer_call_fn_26309Q0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_55_layer_call_and_return_conditional_losses_26337]&'0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_55_layer_call_fn_26327P&'0?-
&?#
!?
inputs??????????
? "???????????
H__inference_sequential_13_layer_call_and_return_conditional_losses_26147r&'??<
5?2
(?%
dense_52_input?????????"
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_26163r&'??<
5?2
(?%
dense_52_input?????????"
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_13_layer_call_and_return_conditional_losses_26220j&'7?4
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
H__inference_sequential_13_layer_call_and_return_conditional_losses_26251j&'7?4
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
-__inference_sequential_13_layer_call_fn_25969e&'??<
5?2
(?%
dense_52_input?????????"
p 

 
? "???????????
-__inference_sequential_13_layer_call_fn_26131e&'??<
5?2
(?%
dense_52_input?????????"
p

 
? "???????????
-__inference_sequential_13_layer_call_fn_26176]&'7?4
-?*
 ?
inputs?????????"
p 

 
? "???????????
-__inference_sequential_13_layer_call_fn_26189]&'7?4
-?*
 ?
inputs?????????"
p

 
? "???????????
#__inference_signature_wrapper_26266?&'I?F
? 
??<
:
dense_52_input(?%
dense_52_input?????????""3?0
.
dense_55"?
dense_55?????????