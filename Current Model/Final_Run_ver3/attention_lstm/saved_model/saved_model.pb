��

�!� 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
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
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�
�
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
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��

�
lstm_22/lstm_cell/biasVarHandleOp*
_output_shapes
: *'

debug_namelstm_22/lstm_cell/bias/*
dtype0*
shape:�*'
shared_namelstm_22/lstm_cell/bias
~
*lstm_22/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell/bias*
_output_shapes	
:�*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOplstm_22/lstm_cell/bias*
_class
loc:@Variable*
_output_shapes	
:�*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:�*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
b
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes	
:�*
dtype0
�
"lstm_22/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *3

debug_name%#lstm_22/lstm_cell/recurrent_kernel/*
dtype0*
shape:	@�*3
shared_name$"lstm_22/lstm_cell/recurrent_kernel
�
6lstm_22/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_22/lstm_cell/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOp"lstm_22/lstm_cell/recurrent_kernel*
_class
loc:@Variable_1*
_output_shapes
:	@�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	@�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	@�*
dtype0
�
lstm_22/lstm_cell/kernelVarHandleOp*
_output_shapes
: *)

debug_namelstm_22/lstm_cell/kernel/*
dtype0*
shape:
��*)
shared_namelstm_22/lstm_cell/kernel
�
,lstm_22/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell/kernel* 
_output_shapes
:
��*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOplstm_22/lstm_cell/kernel*
_class
loc:@Variable_2* 
_output_shapes
:
��*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:
��*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
k
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2* 
_output_shapes
:
��*
dtype0
�
adam/dense_27_bias_velocityVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_27_bias_velocity/*
dtype0*
shape:*,
shared_nameadam/dense_27_bias_velocity
�
/adam/dense_27_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_27_bias_velocity*
_output_shapes
:*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpadam/dense_27_bias_velocity*
_class
loc:@Variable_3*
_output_shapes
:*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:*
dtype0
�
adam/dense_27_bias_momentumVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_27_bias_momentum/*
dtype0*
shape:*,
shared_nameadam/dense_27_bias_momentum
�
/adam/dense_27_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_27_bias_momentum*
_output_shapes
:*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpadam/dense_27_bias_momentum*
_class
loc:@Variable_4*
_output_shapes
:*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:*
dtype0
�
adam/dense_27_kernel_velocityVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_27_kernel_velocity/*
dtype0*
shape
:@*.
shared_nameadam/dense_27_kernel_velocity
�
1adam/dense_27_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_27_kernel_velocity*
_output_shapes

:@*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOpadam/dense_27_kernel_velocity*
_class
loc:@Variable_5*
_output_shapes

:@*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape
:@*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
i
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:@*
dtype0
�
adam/dense_27_kernel_momentumVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_27_kernel_momentum/*
dtype0*
shape
:@*.
shared_nameadam/dense_27_kernel_momentum
�
1adam/dense_27_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_27_kernel_momentum*
_output_shapes

:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpadam/dense_27_kernel_momentum*
_class
loc:@Variable_6*
_output_shapes

:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape
:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
i
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes

:@*
dtype0
�
adam/dense_26_bias_velocityVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_26_bias_velocity/*
dtype0*
shape:@*,
shared_nameadam/dense_26_bias_velocity
�
/adam/dense_26_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_26_bias_velocity*
_output_shapes
:@*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpadam/dense_26_bias_velocity*
_class
loc:@Variable_7*
_output_shapes
:@*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:@*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:@*
dtype0
�
adam/dense_26_bias_momentumVarHandleOp*
_output_shapes
: *,

debug_nameadam/dense_26_bias_momentum/*
dtype0*
shape:@*,
shared_nameadam/dense_26_bias_momentum
�
/adam/dense_26_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_26_bias_momentum*
_output_shapes
:@*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpadam/dense_26_bias_momentum*
_class
loc:@Variable_8*
_output_shapes
:@*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:@*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:@*
dtype0
�
adam/dense_26_kernel_velocityVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_26_kernel_velocity/*
dtype0*
shape
:@@*.
shared_nameadam/dense_26_kernel_velocity
�
1adam/dense_26_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_26_kernel_velocity*
_output_shapes

:@@*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpadam/dense_26_kernel_velocity*
_class
loc:@Variable_9*
_output_shapes

:@@*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape
:@@*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
i
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes

:@@*
dtype0
�
adam/dense_26_kernel_momentumVarHandleOp*
_output_shapes
: *.

debug_name adam/dense_26_kernel_momentum/*
dtype0*
shape
:@@*.
shared_nameadam/dense_26_kernel_momentum
�
1adam/dense_26_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_26_kernel_momentum*
_output_shapes

:@@*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/dense_26_kernel_momentum*
_class
loc:@Variable_10*
_output_shapes

:@@*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape
:@@*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
k
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes

:@@*
dtype0
�
/adam/masked_attention_layer_att_weight_velocityVarHandleOp*
_output_shapes
: *@

debug_name20adam/masked_attention_layer_att_weight_velocity/*
dtype0*
shape
:@*@
shared_name1/adam/masked_attention_layer_att_weight_velocity
�
Cadam/masked_attention_layer_att_weight_velocity/Read/ReadVariableOpReadVariableOp/adam/masked_attention_layer_att_weight_velocity*
_output_shapes

:@*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp/adam/masked_attention_layer_att_weight_velocity*
_class
loc:@Variable_11*
_output_shapes

:@*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape
:@*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
k
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes

:@*
dtype0
�
/adam/masked_attention_layer_att_weight_momentumVarHandleOp*
_output_shapes
: *@

debug_name20adam/masked_attention_layer_att_weight_momentum/*
dtype0*
shape
:@*@
shared_name1/adam/masked_attention_layer_att_weight_momentum
�
Cadam/masked_attention_layer_att_weight_momentum/Read/ReadVariableOpReadVariableOp/adam/masked_attention_layer_att_weight_momentum*
_output_shapes

:@*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp/adam/masked_attention_layer_att_weight_momentum*
_class
loc:@Variable_12*
_output_shapes

:@*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape
:@*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
k
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes

:@*
dtype0
�
$adam/lstm_22_lstm_cell_bias_velocityVarHandleOp*
_output_shapes
: *5

debug_name'%adam/lstm_22_lstm_cell_bias_velocity/*
dtype0*
shape:�*5
shared_name&$adam/lstm_22_lstm_cell_bias_velocity
�
8adam/lstm_22_lstm_cell_bias_velocity/Read/ReadVariableOpReadVariableOp$adam/lstm_22_lstm_cell_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp$adam/lstm_22_lstm_cell_bias_velocity*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
$adam/lstm_22_lstm_cell_bias_momentumVarHandleOp*
_output_shapes
: *5

debug_name'%adam/lstm_22_lstm_cell_bias_momentum/*
dtype0*
shape:�*5
shared_name&$adam/lstm_22_lstm_cell_bias_momentum
�
8adam/lstm_22_lstm_cell_bias_momentum/Read/ReadVariableOpReadVariableOp$adam/lstm_22_lstm_cell_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOp$adam/lstm_22_lstm_cell_bias_momentum*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
0adam/lstm_22_lstm_cell_recurrent_kernel_velocityVarHandleOp*
_output_shapes
: *A

debug_name31adam/lstm_22_lstm_cell_recurrent_kernel_velocity/*
dtype0*
shape:	@�*A
shared_name20adam/lstm_22_lstm_cell_recurrent_kernel_velocity
�
Dadam/lstm_22_lstm_cell_recurrent_kernel_velocity/Read/ReadVariableOpReadVariableOp0adam/lstm_22_lstm_cell_recurrent_kernel_velocity*
_output_shapes
:	@�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOp0adam/lstm_22_lstm_cell_recurrent_kernel_velocity*
_class
loc:@Variable_15*
_output_shapes
:	@�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:	@�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
l
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes
:	@�*
dtype0
�
0adam/lstm_22_lstm_cell_recurrent_kernel_momentumVarHandleOp*
_output_shapes
: *A

debug_name31adam/lstm_22_lstm_cell_recurrent_kernel_momentum/*
dtype0*
shape:	@�*A
shared_name20adam/lstm_22_lstm_cell_recurrent_kernel_momentum
�
Dadam/lstm_22_lstm_cell_recurrent_kernel_momentum/Read/ReadVariableOpReadVariableOp0adam/lstm_22_lstm_cell_recurrent_kernel_momentum*
_output_shapes
:	@�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOp0adam/lstm_22_lstm_cell_recurrent_kernel_momentum*
_class
loc:@Variable_16*
_output_shapes
:	@�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:	@�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
l
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes
:	@�*
dtype0
�
&adam/lstm_22_lstm_cell_kernel_velocityVarHandleOp*
_output_shapes
: *7

debug_name)'adam/lstm_22_lstm_cell_kernel_velocity/*
dtype0*
shape:
��*7
shared_name(&adam/lstm_22_lstm_cell_kernel_velocity
�
:adam/lstm_22_lstm_cell_kernel_velocity/Read/ReadVariableOpReadVariableOp&adam/lstm_22_lstm_cell_kernel_velocity* 
_output_shapes
:
��*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp&adam/lstm_22_lstm_cell_kernel_velocity*
_class
loc:@Variable_17* 
_output_shapes
:
��*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:
��*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
m
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17* 
_output_shapes
:
��*
dtype0
�
&adam/lstm_22_lstm_cell_kernel_momentumVarHandleOp*
_output_shapes
: *7

debug_name)'adam/lstm_22_lstm_cell_kernel_momentum/*
dtype0*
shape:
��*7
shared_name(&adam/lstm_22_lstm_cell_kernel_momentum
�
:adam/lstm_22_lstm_cell_kernel_momentum/Read/ReadVariableOpReadVariableOp&adam/lstm_22_lstm_cell_kernel_momentum* 
_output_shapes
:
��*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp&adam/lstm_22_lstm_cell_kernel_momentum*
_class
loc:@Variable_18* 
_output_shapes
:
��*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:
��*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
m
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18* 
_output_shapes
:
��*
dtype0
�
dense_27/biasVarHandleOp*
_output_shapes
: *

debug_namedense_27/bias/*
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpdense_27/bias*
_class
loc:@Variable_19*
_output_shapes
:*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
g
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes
:*
dtype0
�
dense_27/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_27/kernel/*
dtype0*
shape
:@* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:@*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOpdense_27/kernel*
_class
loc:@Variable_20*
_output_shapes

:@*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape
:@*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
k
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes

:@*
dtype0
�
dense_26/biasVarHandleOp*
_output_shapes
: *

debug_namedense_26/bias/*
dtype0*
shape:@*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:@*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpdense_26/bias*
_class
loc:@Variable_21*
_output_shapes
:@*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape:@*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
g
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes
:@*
dtype0
�
dense_26/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_26/kernel/*
dtype0*
shape
:@@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@@*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpdense_26/kernel*
_class
loc:@Variable_22*
_output_shapes

:@@*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape
:@@*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
k
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes

:@@*
dtype0
�
!masked_attention_layer/att_weightVarHandleOp*
_output_shapes
: *2

debug_name$"masked_attention_layer/att_weight/*
dtype0*
shape
:@*2
shared_name#!masked_attention_layer/att_weight
�
5masked_attention_layer/att_weight/Read/ReadVariableOpReadVariableOp!masked_attention_layer/att_weight*
_output_shapes

:@*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp!masked_attention_layer/att_weight*
_class
loc:@Variable_23*
_output_shapes

:@*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape
:@*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
k
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*
_output_shapes

:@*
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_24*
_output_shapes
: *
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape: *
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
c
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_25/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_25*
_output_shapes
: *
dtype0	
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0	*
shape: *
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0	
c
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes
: *
dtype0	
n
serving_default_xPlaceholder*$
_output_shapes
:��*
dtype0*
shape:��
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_xlstm_22/lstm_cell/kernel"lstm_22/lstm_cell/recurrent_kernellstm_22/lstm_cell/bias!masked_attention_layer/att_weightdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *-
f(R&
$__inference_signature_wrapper_677215

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value�!B�! B�!
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures*
* 
* 
* 
* 
* 
5
0
2
4
5
6
7
8*
5
0
1
2
3
4
5
6*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities*

trace_0* 

serving_default* 
]
_inbound_nodes
_outbound_nodes
_losses
 	_loss_ids
!_losses_override* 
]
"_inbound_nodes
#_outbound_nodes
$_losses
%	_loss_ids
&_losses_override* 
�
'cell
(_inbound_nodes
)_outbound_nodes
*_losses
+	_loss_ids
,_losses_override
-
state_size
._build_shapes_dict*
~
/W
0_inbound_nodes
1_outbound_nodes
2_losses
3	_loss_ids
4_losses_override
5_build_shapes_dict*
�
6_kernel
7bias
8_inbound_nodes
9_outbound_nodes
:_losses
;	_loss_ids
<_losses_override
=_build_shapes_dict*
]
>_inbound_nodes
?_outbound_nodes
@_losses
A	_loss_ids
B_losses_override* 
�
C_kernel
Dbias
E_inbound_nodes
F_outbound_nodes
G_losses
H	_loss_ids
I_losses_override
J_build_shapes_dict*
�
0
1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17*
<
[0
\1
]2
/3
64
75
C6
D7*
* 
UO
VARIABLE_VALUEVariable_250optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_243optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
�

[kernel
\recurrent_kernel
]bias
^_inbound_nodes
__outbound_nodes
`_losses
a	_loss_ids
b_losses_override
c
state_size
d_build_shapes_dict*
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUEVariable_23*_operations/5/W/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_220_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_21-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
UO
VARIABLE_VALUEVariable_200_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_19-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_181optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_171optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_161optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_151optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_141optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_131optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_121optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_111optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_102optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_92optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_82optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_72optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_62optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_52optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_42optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_32optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_2;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_1;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_save_677764
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *+
f&R$
"__inference__traced_restore_677851��
�
�
$__inference_signature_wrapper_677215
x
unknown:
��
	unknown_0:	@�
	unknown_1:	�
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_XlaMustCompile(*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU 2J 8� �J *&
f!R
__inference_serving_fn_677193f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :��: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name677211:&"
 
_user_specified_name677209:&"
 
_user_specified_name677207:&"
 
_user_specified_name677205:&"
 
_user_specified_name677203:&"
 
_user_specified_name677201:&"
 
_user_specified_name677199:&"
 
_user_specified_name677197:G C
$
_output_shapes
:��

_user_specified_namex
�
�
+functional_13_1_lstm_22_1_while_cond_677015P
Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counterA
=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max/
+functional_13_1_lstm_22_1_while_placeholder1
-functional_13_1_lstm_22_1_while_placeholder_11
-functional_13_1_lstm_22_1_while_placeholder_21
-functional_13_1_lstm_22_1_while_placeholder_31
-functional_13_1_lstm_22_1_while_placeholder_4h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677015___redundant_placeholder0h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677015___redundant_placeholder1h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677015___redundant_placeholder2h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677015___redundant_placeholder3h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677015___redundant_placeholder4,
(functional_13_1_lstm_22_1_while_identity
i
&functional_13_1/lstm_22_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value
B :��
$functional_13_1/lstm_22_1/while/LessLess+functional_13_1_lstm_22_1_while_placeholder/functional_13_1/lstm_22_1/while/Less/y:output:0*
T0*
_output_shapes
: �
&functional_13_1/lstm_22_1/while/Less_1LessLfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counter=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/LogicalAnd
LogicalAnd*functional_13_1/lstm_22_1/while/Less_1:z:0(functional_13_1/lstm_22_1/while/Less:z:0*
_output_shapes
: �
(functional_13_1/lstm_22_1/while/IdentityIdentity.functional_13_1/lstm_22_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "]
(functional_13_1_lstm_22_1_while_identity1functional_13_1/lstm_22_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :@:@:@::::::

_output_shapes
::

_output_shapes
::$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namefunctional_13_1/lstm_22_1/Max:d `

_output_shapes
: 
F
_user_specified_name.,functional_13_1/lstm_22_1/while/loop_counter
��
�
__inference__traced_save_677764
file_prefix,
"read_disablecopyonread_variable_25:	 .
$read_1_disablecopyonread_variable_24: 6
$read_2_disablecopyonread_variable_23:@6
$read_3_disablecopyonread_variable_22:@@2
$read_4_disablecopyonread_variable_21:@6
$read_5_disablecopyonread_variable_20:@2
$read_6_disablecopyonread_variable_19:8
$read_7_disablecopyonread_variable_18:
��8
$read_8_disablecopyonread_variable_17:
��7
$read_9_disablecopyonread_variable_16:	@�8
%read_10_disablecopyonread_variable_15:	@�4
%read_11_disablecopyonread_variable_14:	�4
%read_12_disablecopyonread_variable_13:	�7
%read_13_disablecopyonread_variable_12:@7
%read_14_disablecopyonread_variable_11:@7
%read_15_disablecopyonread_variable_10:@@6
$read_16_disablecopyonread_variable_9:@@2
$read_17_disablecopyonread_variable_8:@2
$read_18_disablecopyonread_variable_7:@6
$read_19_disablecopyonread_variable_6:@6
$read_20_disablecopyonread_variable_5:@2
$read_21_disablecopyonread_variable_4:2
$read_22_disablecopyonread_variable_3:8
$read_23_disablecopyonread_variable_2:
��7
$read_24_disablecopyonread_variable_1:	@�1
"read_25_disablecopyonread_variable:	�
savev2_const
identity_53��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_25*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_25^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_24*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_24^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_23*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_23^Read_2/DisableCopyOnRead*
_output_shapes

:@*
dtype0^

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_22*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_22^Read_3/DisableCopyOnRead*
_output_shapes

:@@*
dtype0^

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes

:@@c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:@@i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_21*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_21^Read_4/DisableCopyOnRead*
_output_shapes
:@*
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_20*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_20^Read_5/DisableCopyOnRead*
_output_shapes

:@*
dtype0_
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:@i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_19*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_19^Read_6/DisableCopyOnRead*
_output_shapes
:*
dtype0[
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_18*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_18^Read_7/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_17*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_17^Read_8/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0a
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_16*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_16^Read_9/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0`
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_15*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_15^Read_10/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_14*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_14^Read_11/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_13*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_13^Read_12/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_12*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_12^Read_13/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_11*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_11^Read_14/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_10*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_10^Read_15/DisableCopyOnRead*
_output_shapes

:@@*
dtype0`
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes

:@@e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:@@j
Read_16/DisableCopyOnReadDisableCopyOnRead$read_16_disablecopyonread_variable_9*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp$read_16_disablecopyonread_variable_9^Read_16/DisableCopyOnRead*
_output_shapes

:@@*
dtype0`
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes

:@@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@@j
Read_17/DisableCopyOnReadDisableCopyOnRead$read_17_disablecopyonread_variable_8*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp$read_17_disablecopyonread_variable_8^Read_17/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_18/DisableCopyOnReadDisableCopyOnRead$read_18_disablecopyonread_variable_7*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp$read_18_disablecopyonread_variable_7^Read_18/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_variable_6*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_variable_6^Read_19/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_20/DisableCopyOnReadDisableCopyOnRead$read_20_disablecopyonread_variable_5*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp$read_20_disablecopyonread_variable_5^Read_20/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_4*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_4^Read_21/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_3*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_3^Read_22/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:j
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_2*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_2^Read_23/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_1*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_1^Read_24/DisableCopyOnRead*
_output_shapes
:	@�*
dtype0a
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:	@�f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	@�h
Read_25/DisableCopyOnReadDisableCopyOnRead"read_25_disablecopyonread_variable*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp"read_25_disablecopyonread_variable^Read_25/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB*_operations/5/W/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_52Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_53IdentityIdentity_52:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+
'
%
_user_specified_nameVariable_16:+	'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�q
�
+functional_13_1_lstm_22_1_while_body_677303P
Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counterA
=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max/
+functional_13_1_lstm_22_1_while_placeholder1
-functional_13_1_lstm_22_1_while_placeholder_11
-functional_13_1_lstm_22_1_while_placeholder_21
-functional_13_1_lstm_22_1_while_placeholder_31
-functional_13_1_lstm_22_1_while_placeholder_4�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0^
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��_
Lfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:	@�Z
Kfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�,
(functional_13_1_lstm_22_1_while_identity.
*functional_13_1_lstm_22_1_while_identity_1.
*functional_13_1_lstm_22_1_while_identity_2.
*functional_13_1_lstm_22_1_while_identity_3.
*functional_13_1_lstm_22_1_while_identity_4.
*functional_13_1_lstm_22_1_while_identity_5.
*functional_13_1_lstm_22_1_while_identity_6�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor\
Hfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource:
��]
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource:	@�X
Ifunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource:	���?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp�Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp�@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp�
Qfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Cfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0+functional_13_1_lstm_22_1_while_placeholderZfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
Sfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Efunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0+functional_13_1_lstm_22_1_while_placeholder\functional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpJfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
2functional_13_1/lstm_22_1/while/lstm_cell_1/MatMulMatMulJfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem:item:0Gfunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpLfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
4functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul_1MatMul-functional_13_1_lstm_22_1_while_placeholder_3Ifunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/functional_13_1/lstm_22_1/while/lstm_cell_1/addAddV2<functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul:product:0>functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpKfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
1functional_13_1/lstm_22_1/while/lstm_cell_1/add_1AddV23functional_13_1/lstm_22_1/while/lstm_cell_1/add:z:0Hfunctional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������}
;functional_13_1/lstm_22_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
1functional_13_1/lstm_22_1/while/lstm_cell_1/splitSplitDfunctional_13_1/lstm_22_1/while/lstm_cell_1/split/split_dim:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
3functional_13_1/lstm_22_1/while/lstm_cell_1/SigmoidSigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
5functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_1Sigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
/functional_13_1/lstm_22_1/while/lstm_cell_1/mulMul9functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_1:y:0-functional_13_1_lstm_22_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
0functional_13_1/lstm_22_1/while/lstm_cell_1/TanhTanh:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/mul_1Mul7functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid:y:04functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/add_2AddV23functional_13_1/lstm_22_1/while/lstm_cell_1/mul:z:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
5functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_2Sigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
2functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh_1Tanh5functional_13_1/lstm_22_1/while/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2Mul9functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_2:y:06functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@
.functional_13_1/lstm_22_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
$functional_13_1/lstm_22_1/while/TileTileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:07functional_13_1/lstm_22_1/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
(functional_13_1/lstm_22_1/while/SelectV2SelectV2-functional_13_1/lstm_22_1/while/Tile:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2:z:0-functional_13_1_lstm_22_1_while_placeholder_2*
T0*'
_output_shapes
:���������@�
0functional_13_1/lstm_22_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
&functional_13_1/lstm_22_1/while/Tile_1TileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:09functional_13_1/lstm_22_1/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
0functional_13_1/lstm_22_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
&functional_13_1/lstm_22_1/while/Tile_2TileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:09functional_13_1/lstm_22_1/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:����������
*functional_13_1/lstm_22_1/while/SelectV2_1SelectV2/functional_13_1/lstm_22_1/while/Tile_1:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2:z:0-functional_13_1_lstm_22_1_while_placeholder_3*
T0*'
_output_shapes
:���������@�
*functional_13_1/lstm_22_1/while/SelectV2_2SelectV2/functional_13_1/lstm_22_1/while/Tile_2:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/add_2:z:0-functional_13_1_lstm_22_1_while_placeholder_4*
T0*'
_output_shapes
:���������@�
Dfunctional_13_1/lstm_22_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-functional_13_1_lstm_22_1_while_placeholder_1+functional_13_1_lstm_22_1_while_placeholder1functional_13_1/lstm_22_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���g
%functional_13_1/lstm_22_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
#functional_13_1/lstm_22_1/while/addAddV2+functional_13_1_lstm_22_1_while_placeholder.functional_13_1/lstm_22_1/while/add/y:output:0*
T0*
_output_shapes
: i
'functional_13_1/lstm_22_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
%functional_13_1/lstm_22_1/while/add_1AddV2Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counter0functional_13_1/lstm_22_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
(functional_13_1/lstm_22_1/while/IdentityIdentity)functional_13_1/lstm_22_1/while/add_1:z:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_1Identity=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_2Identity'functional_13_1/lstm_22_1/while/add:z:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_3IdentityTfunctional_13_1/lstm_22_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_4Identity1functional_13_1/lstm_22_1/while/SelectV2:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*'
_output_shapes
:���������@�
*functional_13_1/lstm_22_1/while/Identity_5Identity3functional_13_1/lstm_22_1/while/SelectV2_1:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*'
_output_shapes
:���������@�
*functional_13_1/lstm_22_1/while/Identity_6Identity3functional_13_1/lstm_22_1/while/SelectV2_2:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*'
_output_shapes
:���������@�
$functional_13_1/lstm_22_1/while/NoOpNoOp@^functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOpB^functional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpA^functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "a
*functional_13_1_lstm_22_1_while_identity_13functional_13_1/lstm_22_1/while/Identity_1:output:0"a
*functional_13_1_lstm_22_1_while_identity_23functional_13_1/lstm_22_1/while/Identity_2:output:0"a
*functional_13_1_lstm_22_1_while_identity_33functional_13_1/lstm_22_1/while/Identity_3:output:0"a
*functional_13_1_lstm_22_1_while_identity_43functional_13_1/lstm_22_1/while/Identity_4:output:0"a
*functional_13_1_lstm_22_1_while_identity_53functional_13_1/lstm_22_1/while/Identity_5:output:0"a
*functional_13_1_lstm_22_1_while_identity_63functional_13_1/lstm_22_1/while/Identity_6:output:0"]
(functional_13_1_lstm_22_1_while_identity1functional_13_1/lstm_22_1/while/Identity:output:0"�
Ifunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resourceKfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resourceLfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
Hfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resourceJfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K: : : : :���������@:���������@:���������@: : : : : 2�
?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpAfunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:{w

_output_shapes
: 
]
_user_specified_nameECfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor:yu

_output_shapes
: 
[
_user_specified_nameCAfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namefunctional_13_1/lstm_22_1/Max:d `

_output_shapes
: 
F
_user_specified_name.,functional_13_1/lstm_22_1/while/loop_counter
�v
�
"__inference__traced_restore_677851
file_prefix&
assignvariableop_variable_25:	 (
assignvariableop_1_variable_24: 0
assignvariableop_2_variable_23:@0
assignvariableop_3_variable_22:@@,
assignvariableop_4_variable_21:@0
assignvariableop_5_variable_20:@,
assignvariableop_6_variable_19:2
assignvariableop_7_variable_18:
��2
assignvariableop_8_variable_17:
��1
assignvariableop_9_variable_16:	@�2
assignvariableop_10_variable_15:	@�.
assignvariableop_11_variable_14:	�.
assignvariableop_12_variable_13:	�1
assignvariableop_13_variable_12:@1
assignvariableop_14_variable_11:@1
assignvariableop_15_variable_10:@@0
assignvariableop_16_variable_9:@@,
assignvariableop_17_variable_8:@,
assignvariableop_18_variable_7:@0
assignvariableop_19_variable_6:@0
assignvariableop_20_variable_5:@,
assignvariableop_21_variable_4:,
assignvariableop_22_variable_3:2
assignvariableop_23_variable_2:
��1
assignvariableop_24_variable_1:	@�+
assignvariableop_25_variable:	�
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB*_operations/5/W/.ATTRIBUTES/VARIABLE_VALUEB0_operations/6/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/bias/.ATTRIBUTES/VARIABLE_VALUEB0_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_25Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_24Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_23Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_22Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_21Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_20Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_19Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_18Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_17Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_16Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_15Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_14Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_13Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_12Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_11Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_10Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_9Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_8Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_7Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_6Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_5Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_4Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_3Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variableIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_27Identity_27:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+
'
%
_user_specified_nameVariable_16:+	'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_25:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�	
"__inference_serving_default_677482

inputsV
Bfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resource:
��W
Dfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resource:	@�R
Cfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource:	�Z
Hfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource:@I
7functional_13_1_dense_26_1_cast_readvariableop_resource:@@D
6functional_13_1_dense_26_1_add_readvariableop_resource:@I
7functional_13_1_dense_27_1_cast_readvariableop_resource:@D
6functional_13_1_dense_27_1_add_readvariableop_resource:
identity��-functional_13_1/dense_26_1/Add/ReadVariableOp�.functional_13_1/dense_26_1/Cast/ReadVariableOp�-functional_13_1/dense_27_1/Add/ReadVariableOp�.functional_13_1/dense_27_1/Cast/ReadVariableOp�9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp�;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp�:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp�functional_13_1/lstm_22_1/while�Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpZ
functional_13_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
functional_13_1/NotEqualNotEqualinputsfunctional_13_1/Const:output:0*
T0*-
_output_shapes
:�����������x
%functional_13_1/Any/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
functional_13_1/AnyAnyfunctional_13_1/NotEqual:z:0.functional_13_1/Any/reduction_indices:output:0*(
_output_shapes
:����������g
"functional_13_1/masking_13_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%functional_13_1/masking_13_1/NotEqualNotEqualinputs+functional_13_1/masking_13_1/Const:output:0*
T0*-
_output_shapes
:�����������}
2functional_13_1/masking_13_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 functional_13_1/masking_13_1/AnyAny)functional_13_1/masking_13_1/NotEqual:z:0;functional_13_1/masking_13_1/Any/reduction_indices:output:0*,
_output_shapes
:����������*
	keep_dims(�
!functional_13_1/masking_13_1/CastCast)functional_13_1/masking_13_1/Any:output:0*

DstT0*

SrcT0
*,
_output_shapes
:�����������
 functional_13_1/masking_13_1/mulMulinputs%functional_13_1/masking_13_1/Cast:y:0*
T0*-
_output_shapes
:������������
$functional_13_1/masking_13_1/SqueezeSqueeze)functional_13_1/masking_13_1/Any:output:0*
T0
*(
_output_shapes
:����������*
squeeze_dims
�
functional_13_1/lstm_22_1/ShapeShape$functional_13_1/masking_13_1/mul:z:0*
T0*
_output_shapes
::��w
-functional_13_1/lstm_22_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/functional_13_1/lstm_22_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/functional_13_1/lstm_22_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'functional_13_1/lstm_22_1/strided_sliceStridedSlice(functional_13_1/lstm_22_1/Shape:output:06functional_13_1/lstm_22_1/strided_slice/stack:output:08functional_13_1/lstm_22_1/strided_slice/stack_1:output:08functional_13_1/lstm_22_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(functional_13_1/lstm_22_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
&functional_13_1/lstm_22_1/zeros/packedPack0functional_13_1/lstm_22_1/strided_slice:output:01functional_13_1/lstm_22_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%functional_13_1/lstm_22_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
functional_13_1/lstm_22_1/zerosFill/functional_13_1/lstm_22_1/zeros/packed:output:0.functional_13_1/lstm_22_1/zeros/Const:output:0*
T0*'
_output_shapes
:���������@l
*functional_13_1/lstm_22_1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
(functional_13_1/lstm_22_1/zeros_1/packedPack0functional_13_1/lstm_22_1/strided_slice:output:03functional_13_1/lstm_22_1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:l
'functional_13_1/lstm_22_1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
!functional_13_1/lstm_22_1/zeros_1Fill1functional_13_1/lstm_22_1/zeros_1/packed:output:00functional_13_1/lstm_22_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@�
/functional_13_1/lstm_22_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
1functional_13_1/lstm_22_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
1functional_13_1/lstm_22_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
)functional_13_1/lstm_22_1/strided_slice_1StridedSlice$functional_13_1/masking_13_1/mul:z:08functional_13_1/lstm_22_1/strided_slice_1/stack:output:0:functional_13_1/lstm_22_1/strided_slice_1/stack_1:output:0:functional_13_1/lstm_22_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask}
(functional_13_1/lstm_22_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
#functional_13_1/lstm_22_1/transpose	Transpose$functional_13_1/masking_13_1/mul:z:01functional_13_1/lstm_22_1/transpose/perm:output:0*
T0*-
_output_shapes
:�����������s
(functional_13_1/lstm_22_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
$functional_13_1/lstm_22_1/ExpandDims
ExpandDimsfunctional_13_1/Any:output:01functional_13_1/lstm_22_1/ExpandDims/dim:output:0*
T0
*,
_output_shapes
:����������
*functional_13_1/lstm_22_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%functional_13_1/lstm_22_1/transpose_1	Transpose-functional_13_1/lstm_22_1/ExpandDims:output:03functional_13_1/lstm_22_1/transpose_1/perm:output:0*
T0
*,
_output_shapes
:�����������
5functional_13_1/lstm_22_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������w
4functional_13_1/lstm_22_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
'functional_13_1/lstm_22_1/TensorArrayV2TensorListReserve>functional_13_1/lstm_22_1/TensorArrayV2/element_shape:output:0=functional_13_1/lstm_22_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ofunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Afunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'functional_13_1/lstm_22_1/transpose:y:0Xfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���y
/functional_13_1/lstm_22_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1functional_13_1/lstm_22_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1functional_13_1/lstm_22_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)functional_13_1/lstm_22_1/strided_slice_2StridedSlice'functional_13_1/lstm_22_1/transpose:y:08functional_13_1/lstm_22_1/strided_slice_2/stack:output:0:functional_13_1/lstm_22_1/strided_slice_2/stack_1:output:0:functional_13_1/lstm_22_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpBfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
,functional_13_1/lstm_22_1/lstm_cell_1/MatMulMatMul2functional_13_1/lstm_22_1/strided_slice_2:output:0Afunctional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpDfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
.functional_13_1/lstm_22_1/lstm_cell_1/MatMul_1MatMul(functional_13_1/lstm_22_1/zeros:output:0Cfunctional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
)functional_13_1/lstm_22_1/lstm_cell_1/addAddV26functional_13_1/lstm_22_1/lstm_cell_1/MatMul:product:08functional_13_1/lstm_22_1/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:�����������
:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpCfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+functional_13_1/lstm_22_1/lstm_cell_1/add_1AddV2-functional_13_1/lstm_22_1/lstm_cell_1/add:z:0Bfunctional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������w
5functional_13_1/lstm_22_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
+functional_13_1/lstm_22_1/lstm_cell_1/splitSplit>functional_13_1/lstm_22_1/lstm_cell_1/split/split_dim:output:0/functional_13_1/lstm_22_1/lstm_cell_1/add_1:z:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split�
-functional_13_1/lstm_22_1/lstm_cell_1/SigmoidSigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:0*
T0*'
_output_shapes
:���������@�
/functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_1Sigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:1*
T0*'
_output_shapes
:���������@�
)functional_13_1/lstm_22_1/lstm_cell_1/mulMul3functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_1:y:0*functional_13_1/lstm_22_1/zeros_1:output:0*
T0*'
_output_shapes
:���������@�
*functional_13_1/lstm_22_1/lstm_cell_1/TanhTanh4functional_13_1/lstm_22_1/lstm_cell_1/split:output:2*
T0*'
_output_shapes
:���������@�
+functional_13_1/lstm_22_1/lstm_cell_1/mul_1Mul1functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid:y:0.functional_13_1/lstm_22_1/lstm_cell_1/Tanh:y:0*
T0*'
_output_shapes
:���������@�
+functional_13_1/lstm_22_1/lstm_cell_1/add_2AddV2-functional_13_1/lstm_22_1/lstm_cell_1/mul:z:0/functional_13_1/lstm_22_1/lstm_cell_1/mul_1:z:0*
T0*'
_output_shapes
:���������@�
/functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_2Sigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:3*
T0*'
_output_shapes
:���������@�
,functional_13_1/lstm_22_1/lstm_cell_1/Tanh_1Tanh/functional_13_1/lstm_22_1/lstm_cell_1/add_2:z:0*
T0*'
_output_shapes
:���������@�
+functional_13_1/lstm_22_1/lstm_cell_1/mul_2Mul3functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_2:y:00functional_13_1/lstm_22_1/lstm_cell_1/Tanh_1:y:0*
T0*'
_output_shapes
:���������@�
7functional_13_1/lstm_22_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   y
6functional_13_1/lstm_22_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
)functional_13_1/lstm_22_1/TensorArrayV2_1TensorListReserve@functional_13_1/lstm_22_1/TensorArrayV2_1/element_shape:output:0?functional_13_1/lstm_22_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���`
functional_13_1/lstm_22_1/timeConst*
_output_shapes
: *
dtype0*
value	B : g
$functional_13_1/lstm_22_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value
B :�`
functional_13_1/lstm_22_1/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%functional_13_1/lstm_22_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : g
%functional_13_1/lstm_22_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
functional_13_1/lstm_22_1/rangeRange.functional_13_1/lstm_22_1/range/start:output:0'functional_13_1/lstm_22_1/Rank:output:0.functional_13_1/lstm_22_1/range/delta:output:0*
_output_shapes
: f
#functional_13_1/lstm_22_1/Max/inputConst*
_output_shapes
: *
dtype0*
value
B :��
functional_13_1/lstm_22_1/MaxMax,functional_13_1/lstm_22_1/Max/input:output:0(functional_13_1/lstm_22_1/range:output:0*
T0*
_output_shapes
: �
7functional_13_1/lstm_22_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������y
6functional_13_1/lstm_22_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
)functional_13_1/lstm_22_1/TensorArrayV2_2TensorListReserve@functional_13_1/lstm_22_1/TensorArrayV2_2/element_shape:output:0?functional_13_1/lstm_22_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Qfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Cfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor)functional_13_1/lstm_22_1/transpose_1:y:0Zfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
$functional_13_1/lstm_22_1/zeros_like	ZerosLike/functional_13_1/lstm_22_1/lstm_cell_1/mul_2:z:0*
T0*'
_output_shapes
:���������@n
,functional_13_1/lstm_22_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �	
functional_13_1/lstm_22_1/whileWhile5functional_13_1/lstm_22_1/while/loop_counter:output:0&functional_13_1/lstm_22_1/Max:output:0'functional_13_1/lstm_22_1/time:output:02functional_13_1/lstm_22_1/TensorArrayV2_1:handle:0(functional_13_1/lstm_22_1/zeros_like:y:0(functional_13_1/lstm_22_1/zeros:output:0*functional_13_1/lstm_22_1/zeros_1:output:0Qfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Sfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Bfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resourceDfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resourceCfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*_
_output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*7
body/R-
+functional_13_1_lstm_22_1_while_body_677303*7
cond/R-
+functional_13_1_lstm_22_1_while_cond_677302*^
output_shapesM
K: : : : :���������@:���������@:���������@: : : : : *
parallel_iterations �
Jfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
<functional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStackTensorListStack(functional_13_1/lstm_22_1/while:output:3Sfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������@*
element_dtype0*
num_elements��
/functional_13_1/lstm_22_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1functional_13_1/lstm_22_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1functional_13_1/lstm_22_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)functional_13_1/lstm_22_1/strided_slice_3StridedSliceEfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack:tensor:08functional_13_1/lstm_22_1/strided_slice_3/stack:output:0:functional_13_1/lstm_22_1/strided_slice_3/stack_1:output:0:functional_13_1/lstm_22_1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask
*functional_13_1/lstm_22_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%functional_13_1/lstm_22_1/transpose_2	TransposeEfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack:tensor:03functional_13_1/lstm_22_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:����������@�
.functional_13_1/masked_attention_layer_1/ShapeShape)functional_13_1/lstm_22_1/transpose_2:y:0*
T0*
_output_shapes
::���
0functional_13_1/masked_attention_layer_1/unstackUnpack7functional_13_1/masked_attention_layer_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num�
?functional_13_1/masked_attention_layer_1/Shape_1/ReadVariableOpReadVariableOpHfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
0functional_13_1/masked_attention_layer_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@      �
2functional_13_1/masked_attention_layer_1/unstack_1Unpack9functional_13_1/masked_attention_layer_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
num�
6functional_13_1/masked_attention_layer_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
0functional_13_1/masked_attention_layer_1/ReshapeReshape)functional_13_1/lstm_22_1/transpose_2:y:0?functional_13_1/masked_attention_layer_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������@�
Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpReadVariableOpHfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
7functional_13_1/masked_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
2functional_13_1/masked_attention_layer_1/transpose	TransposeIfunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOp:value:0@functional_13_1/masked_attention_layer_1/transpose/perm:output:0*
T0*
_output_shapes

:@�
8functional_13_1/masked_attention_layer_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   �����
2functional_13_1/masked_attention_layer_1/Reshape_1Reshape6functional_13_1/masked_attention_layer_1/transpose:y:0Afunctional_13_1/masked_attention_layer_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@�
/functional_13_1/masked_attention_layer_1/MatMulMatMul9functional_13_1/masked_attention_layer_1/Reshape:output:0;functional_13_1/masked_attention_layer_1/Reshape_1:output:0*
T0*'
_output_shapes
:���������}
:functional_13_1/masked_attention_layer_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�|
:functional_13_1/masked_attention_layer_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
8functional_13_1/masked_attention_layer_1/Reshape_2/shapePack9functional_13_1/masked_attention_layer_1/unstack:output:0Cfunctional_13_1/masked_attention_layer_1/Reshape_2/shape/1:output:0Cfunctional_13_1/masked_attention_layer_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
2functional_13_1/masked_attention_layer_1/Reshape_2Reshape9functional_13_1/masked_attention_layer_1/MatMul:product:0Afunctional_13_1/masked_attention_layer_1/Reshape_2/shape:output:0*
T0*,
_output_shapes
:�����������
-functional_13_1/masked_attention_layer_1/TanhTanh;functional_13_1/masked_attention_layer_1/Reshape_2:output:0*
T0*,
_output_shapes
:�����������
-functional_13_1/masked_attention_layer_1/CastCastfunctional_13_1/Any:output:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
7functional_13_1/masked_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
3functional_13_1/masked_attention_layer_1/ExpandDims
ExpandDims1functional_13_1/masked_attention_layer_1/Cast:y:0@functional_13_1/masked_attention_layer_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:�����������
,functional_13_1/masked_attention_layer_1/mulMul1functional_13_1/masked_attention_layer_1/Tanh:y:0<functional_13_1/masked_attention_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:����������s
.functional_13_1/masked_attention_layer_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,functional_13_1/masked_attention_layer_1/subSub7functional_13_1/masked_attention_layer_1/sub/x:output:0<functional_13_1/masked_attention_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:����������u
0functional_13_1/masked_attention_layer_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�P�
.functional_13_1/masked_attention_layer_1/mul_1Mul9functional_13_1/masked_attention_layer_1/mul_1/x:output:00functional_13_1/masked_attention_layer_1/sub:z:0*
T0*,
_output_shapes
:�����������
.functional_13_1/masked_attention_layer_1/sub_1Sub0functional_13_1/masked_attention_layer_1/mul:z:02functional_13_1/masked_attention_layer_1/mul_1:z:0*
T0*,
_output_shapes
:����������o
-functional_13_1/masked_attention_layer_1/RankConst*
_output_shapes
: *
dtype0*
value	B :q
/functional_13_1/masked_attention_layer_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :r
0functional_13_1/masked_attention_layer_1/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/Sub_2Sub8functional_13_1/masked_attention_layer_1/Rank_1:output:09functional_13_1/masked_attention_layer_1/Sub_2/y:output:0*
T0*
_output_shapes
: v
4functional_13_1/masked_attention_layer_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : v
4functional_13_1/masked_attention_layer_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :v
4functional_13_1/masked_attention_layer_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/rangeRange=functional_13_1/masked_attention_layer_1/range/start:output:0=functional_13_1/masked_attention_layer_1/range/limit:output:0=functional_13_1/masked_attention_layer_1/range/delta:output:0*
_output_shapes
:x
6functional_13_1/masked_attention_layer_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_1Range?functional_13_1/masked_attention_layer_1/range_1/start:output:02functional_13_1/masked_attention_layer_1/Sub_2:z:0?functional_13_1/masked_attention_layer_1/range_1/delta:output:0*
_output_shapes
: �
8functional_13_1/masked_attention_layer_1/concat/values_1Pack2functional_13_1/masked_attention_layer_1/Sub_2:z:0*
N*
T0*
_output_shapes
:�
8functional_13_1/masked_attention_layer_1/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:v
4functional_13_1/masked_attention_layer_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/functional_13_1/masked_attention_layer_1/concatConcatV27functional_13_1/masked_attention_layer_1/range:output:0Afunctional_13_1/masked_attention_layer_1/concat/values_1:output:09functional_13_1/masked_attention_layer_1/range_1:output:0Afunctional_13_1/masked_attention_layer_1/concat/values_3:output:0=functional_13_1/masked_attention_layer_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4functional_13_1/masked_attention_layer_1/transpose_1	Transpose2functional_13_1/masked_attention_layer_1/sub_1:z:08functional_13_1/masked_attention_layer_1/concat:output:0*
T0*,
_output_shapes
:�����������
0functional_13_1/masked_attention_layer_1/SoftmaxSoftmax8functional_13_1/masked_attention_layer_1/transpose_1:y:0*
T0*,
_output_shapes
:����������r
0functional_13_1/masked_attention_layer_1/Sub_3/yConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/Sub_3Sub8functional_13_1/masked_attention_layer_1/Rank_1:output:09functional_13_1/masked_attention_layer_1/Sub_3/y:output:0*
T0*
_output_shapes
: x
6functional_13_1/masked_attention_layer_1/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : x
6functional_13_1/masked_attention_layer_1/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_2Range?functional_13_1/masked_attention_layer_1/range_2/start:output:0?functional_13_1/masked_attention_layer_1/range_2/limit:output:0?functional_13_1/masked_attention_layer_1/range_2/delta:output:0*
_output_shapes
:x
6functional_13_1/masked_attention_layer_1/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_3Range?functional_13_1/masked_attention_layer_1/range_3/start:output:02functional_13_1/masked_attention_layer_1/Sub_3:z:0?functional_13_1/masked_attention_layer_1/range_3/delta:output:0*
_output_shapes
: �
:functional_13_1/masked_attention_layer_1/concat_1/values_1Pack2functional_13_1/masked_attention_layer_1/Sub_3:z:0*
N*
T0*
_output_shapes
:�
:functional_13_1/masked_attention_layer_1/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:x
6functional_13_1/masked_attention_layer_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1functional_13_1/masked_attention_layer_1/concat_1ConcatV29functional_13_1/masked_attention_layer_1/range_2:output:0Cfunctional_13_1/masked_attention_layer_1/concat_1/values_1:output:09functional_13_1/masked_attention_layer_1/range_3:output:0Cfunctional_13_1/masked_attention_layer_1/concat_1/values_3:output:0?functional_13_1/masked_attention_layer_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4functional_13_1/masked_attention_layer_1/transpose_2	Transpose:functional_13_1/masked_attention_layer_1/Softmax:softmax:0:functional_13_1/masked_attention_layer_1/concat_1:output:0*
T0*,
_output_shapes
:�����������
.functional_13_1/masked_attention_layer_1/mul_2Mul)functional_13_1/lstm_22_1/transpose_2:y:08functional_13_1/masked_attention_layer_1/transpose_2:y:0*
T0*,
_output_shapes
:����������@�
>functional_13_1/masked_attention_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,functional_13_1/masked_attention_layer_1/SumSum2functional_13_1/masked_attention_layer_1/mul_2:z:0Gfunctional_13_1/masked_attention_layer_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������@�
.functional_13_1/dense_26_1/Cast/ReadVariableOpReadVariableOp7functional_13_1_dense_26_1_cast_readvariableop_resource*
_output_shapes

:@@*
dtype0�
!functional_13_1/dense_26_1/MatMulMatMul5functional_13_1/masked_attention_layer_1/Sum:output:06functional_13_1/dense_26_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
-functional_13_1/dense_26_1/Add/ReadVariableOpReadVariableOp6functional_13_1_dense_26_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_13_1/dense_26_1/AddAddV2+functional_13_1/dense_26_1/MatMul:product:05functional_13_1/dense_26_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@}
functional_13_1/dense_26_1/TanhTanh"functional_13_1/dense_26_1/Add:z:0*
T0*'
_output_shapes
:���������@�
.functional_13_1/dense_27_1/Cast/ReadVariableOpReadVariableOp7functional_13_1_dense_27_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
!functional_13_1/dense_27_1/MatMulMatMul#functional_13_1/dense_26_1/Tanh:y:06functional_13_1/dense_27_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-functional_13_1/dense_27_1/Add/ReadVariableOpReadVariableOp6functional_13_1_dense_27_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_13_1/dense_27_1/AddAddV2+functional_13_1/dense_27_1/MatMul:product:05functional_13_1/dense_27_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"functional_13_1/dense_27_1/SoftmaxSoftmax"functional_13_1/dense_27_1/Add:z:0*
T0*'
_output_shapes
:���������{
IdentityIdentity,functional_13_1/dense_27_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^functional_13_1/dense_26_1/Add/ReadVariableOp/^functional_13_1/dense_26_1/Cast/ReadVariableOp.^functional_13_1/dense_27_1/Add/ReadVariableOp/^functional_13_1/dense_27_1/Cast/ReadVariableOp:^functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp<^functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp;^functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp ^functional_13_1/lstm_22_1/whileB^functional_13_1/masked_attention_layer_1/transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : : : 2^
-functional_13_1/dense_26_1/Add/ReadVariableOp-functional_13_1/dense_26_1/Add/ReadVariableOp2`
.functional_13_1/dense_26_1/Cast/ReadVariableOp.functional_13_1/dense_26_1/Cast/ReadVariableOp2^
-functional_13_1/dense_27_1/Add/ReadVariableOp-functional_13_1/dense_27_1/Add/ReadVariableOp2`
.functional_13_1/dense_27_1/Cast/ReadVariableOp.functional_13_1/dense_27_1/Cast/ReadVariableOp2v
9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp2z
;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp2x
:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp2B
functional_13_1/lstm_22_1/whilefunctional_13_1/lstm_22_1/while2�
Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpAfunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+functional_13_1_lstm_22_1_while_cond_677302P
Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counterA
=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max/
+functional_13_1_lstm_22_1_while_placeholder1
-functional_13_1_lstm_22_1_while_placeholder_11
-functional_13_1_lstm_22_1_while_placeholder_21
-functional_13_1_lstm_22_1_while_placeholder_31
-functional_13_1_lstm_22_1_while_placeholder_4h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677302___redundant_placeholder0h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677302___redundant_placeholder1h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677302___redundant_placeholder2h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677302___redundant_placeholder3h
dfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_cond_677302___redundant_placeholder4,
(functional_13_1_lstm_22_1_while_identity
i
&functional_13_1/lstm_22_1/while/Less/yConst*
_output_shapes
: *
dtype0*
value
B :��
$functional_13_1/lstm_22_1/while/LessLess+functional_13_1_lstm_22_1_while_placeholder/functional_13_1/lstm_22_1/while/Less/y:output:0*
T0*
_output_shapes
: �
&functional_13_1/lstm_22_1/while/Less_1LessLfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counter=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/LogicalAnd
LogicalAnd*functional_13_1/lstm_22_1/while/Less_1:z:0(functional_13_1/lstm_22_1/while/Less:z:0*
_output_shapes
: �
(functional_13_1/lstm_22_1/while/IdentityIdentity.functional_13_1/lstm_22_1/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "]
(functional_13_1_lstm_22_1_while_identity1functional_13_1/lstm_22_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U: : : : :���������@:���������@:���������@::::::

_output_shapes
::

_output_shapes
::-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namefunctional_13_1/lstm_22_1/Max:d `

_output_shapes
: 
F
_user_specified_name.,functional_13_1/lstm_22_1/while/loop_counter
��
�	
__inference_serving_fn_677193
xV
Bfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resource:
��W
Dfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resource:	@�R
Cfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource:	�Z
Hfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource:@I
7functional_13_1_dense_26_1_cast_readvariableop_resource:@@D
6functional_13_1_dense_26_1_add_readvariableop_resource:@I
7functional_13_1_dense_27_1_cast_readvariableop_resource:@D
6functional_13_1_dense_27_1_add_readvariableop_resource:
identity��-functional_13_1/dense_26_1/Add/ReadVariableOp�.functional_13_1/dense_26_1/Cast/ReadVariableOp�-functional_13_1/dense_27_1/Add/ReadVariableOp�.functional_13_1/dense_27_1/Cast/ReadVariableOp�9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp�;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp�:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp�functional_13_1/lstm_22_1/while�Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpZ
functional_13_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    v
functional_13_1/NotEqualNotEqualxfunctional_13_1/Const:output:0*
T0*$
_output_shapes
:��x
%functional_13_1/Any/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
functional_13_1/AnyAnyfunctional_13_1/NotEqual:z:0.functional_13_1/Any/reduction_indices:output:0*
_output_shapes
:	�g
"functional_13_1/masking_13_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
%functional_13_1/masking_13_1/NotEqualNotEqualx+functional_13_1/masking_13_1/Const:output:0*
T0*$
_output_shapes
:��}
2functional_13_1/masking_13_1/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 functional_13_1/masking_13_1/AnyAny)functional_13_1/masking_13_1/NotEqual:z:0;functional_13_1/masking_13_1/Any/reduction_indices:output:0*#
_output_shapes
:�*
	keep_dims(�
!functional_13_1/masking_13_1/CastCast)functional_13_1/masking_13_1/Any:output:0*

DstT0*

SrcT0
*#
_output_shapes
:��
 functional_13_1/masking_13_1/mulMulx%functional_13_1/masking_13_1/Cast:y:0*
T0*$
_output_shapes
:���
$functional_13_1/masking_13_1/SqueezeSqueeze)functional_13_1/masking_13_1/Any:output:0*
T0
*
_output_shapes
:	�*
squeeze_dims
t
functional_13_1/lstm_22_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   �   �   t
functional_13_1/lstm_22_1/zerosConst*
_output_shapes

:@*
dtype0*
valueB@*    v
!functional_13_1/lstm_22_1/zeros_1Const*
_output_shapes

:@*
dtype0*
valueB@*    �
-functional_13_1/lstm_22_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            �
/functional_13_1/lstm_22_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           �
/functional_13_1/lstm_22_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
'functional_13_1/lstm_22_1/strided_sliceStridedSlice$functional_13_1/masking_13_1/mul:z:06functional_13_1/lstm_22_1/strided_slice/stack:output:08functional_13_1/lstm_22_1/strided_slice/stack_1:output:08functional_13_1/lstm_22_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask*
shrink_axis_mask}
(functional_13_1/lstm_22_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
#functional_13_1/lstm_22_1/transpose	Transpose$functional_13_1/masking_13_1/mul:z:01functional_13_1/lstm_22_1/transpose/perm:output:0*
T0*$
_output_shapes
:��s
(functional_13_1/lstm_22_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
$functional_13_1/lstm_22_1/ExpandDims
ExpandDimsfunctional_13_1/Any:output:01functional_13_1/lstm_22_1/ExpandDims/dim:output:0*
T0
*#
_output_shapes
:�
*functional_13_1/lstm_22_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%functional_13_1/lstm_22_1/transpose_1	Transpose-functional_13_1/lstm_22_1/ExpandDims:output:03functional_13_1/lstm_22_1/transpose_1/perm:output:0*
T0
*#
_output_shapes
:��
5functional_13_1/lstm_22_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������w
4functional_13_1/lstm_22_1/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
'functional_13_1/lstm_22_1/TensorArrayV2TensorListReserve>functional_13_1/lstm_22_1/TensorArrayV2/element_shape:output:0=functional_13_1/lstm_22_1/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ofunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   �   �
Afunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'functional_13_1/lstm_22_1/transpose:y:0Xfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���y
/functional_13_1/lstm_22_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1functional_13_1/lstm_22_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1functional_13_1/lstm_22_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)functional_13_1/lstm_22_1/strided_slice_1StridedSlice'functional_13_1/lstm_22_1/transpose:y:08functional_13_1/lstm_22_1/strided_slice_1/stack:output:0:functional_13_1/lstm_22_1/strided_slice_1/stack_1:output:0:functional_13_1/lstm_22_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*
shrink_axis_mask�
9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOpReadVariableOpBfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
,functional_13_1/lstm_22_1/lstm_cell_1/MatMulMatMul2functional_13_1/lstm_22_1/strided_slice_1:output:0Afunctional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpDfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
.functional_13_1/lstm_22_1/lstm_cell_1/MatMul_1MatMul(functional_13_1/lstm_22_1/zeros:output:0Cfunctional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
)functional_13_1/lstm_22_1/lstm_cell_1/addAddV26functional_13_1/lstm_22_1/lstm_cell_1/MatMul:product:08functional_13_1/lstm_22_1/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes
:	��
:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOpReadVariableOpCfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+functional_13_1/lstm_22_1/lstm_cell_1/add_1AddV2-functional_13_1/lstm_22_1/lstm_cell_1/add:z:0Bfunctional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�w
5functional_13_1/lstm_22_1/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
+functional_13_1/lstm_22_1/lstm_cell_1/splitSplit>functional_13_1/lstm_22_1/lstm_cell_1/split/split_dim:output:0/functional_13_1/lstm_22_1/lstm_cell_1/add_1:z:0*
T0*<
_output_shapes*
(:@:@:@:@*
	num_split�
-functional_13_1/lstm_22_1/lstm_cell_1/SigmoidSigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:0*
T0*
_output_shapes

:@�
/functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_1Sigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:1*
T0*
_output_shapes

:@�
)functional_13_1/lstm_22_1/lstm_cell_1/mulMul3functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_1:y:0*functional_13_1/lstm_22_1/zeros_1:output:0*
T0*
_output_shapes

:@�
*functional_13_1/lstm_22_1/lstm_cell_1/TanhTanh4functional_13_1/lstm_22_1/lstm_cell_1/split:output:2*
T0*
_output_shapes

:@�
+functional_13_1/lstm_22_1/lstm_cell_1/mul_1Mul1functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid:y:0.functional_13_1/lstm_22_1/lstm_cell_1/Tanh:y:0*
T0*
_output_shapes

:@�
+functional_13_1/lstm_22_1/lstm_cell_1/add_2AddV2-functional_13_1/lstm_22_1/lstm_cell_1/mul:z:0/functional_13_1/lstm_22_1/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:@�
/functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_2Sigmoid4functional_13_1/lstm_22_1/lstm_cell_1/split:output:3*
T0*
_output_shapes

:@�
,functional_13_1/lstm_22_1/lstm_cell_1/Tanh_1Tanh/functional_13_1/lstm_22_1/lstm_cell_1/add_2:z:0*
T0*
_output_shapes

:@�
+functional_13_1/lstm_22_1/lstm_cell_1/mul_2Mul3functional_13_1/lstm_22_1/lstm_cell_1/Sigmoid_2:y:00functional_13_1/lstm_22_1/lstm_cell_1/Tanh_1:y:0*
T0*
_output_shapes

:@�
7functional_13_1/lstm_22_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   y
6functional_13_1/lstm_22_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
)functional_13_1/lstm_22_1/TensorArrayV2_1TensorListReserve@functional_13_1/lstm_22_1/TensorArrayV2_1/element_shape:output:0?functional_13_1/lstm_22_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���`
functional_13_1/lstm_22_1/timeConst*
_output_shapes
: *
dtype0*
value	B : g
$functional_13_1/lstm_22_1/Rank/ConstConst*
_output_shapes
: *
dtype0*
value
B :�`
functional_13_1/lstm_22_1/RankConst*
_output_shapes
: *
dtype0*
value	B : g
%functional_13_1/lstm_22_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : g
%functional_13_1/lstm_22_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
functional_13_1/lstm_22_1/rangeRange.functional_13_1/lstm_22_1/range/start:output:0'functional_13_1/lstm_22_1/Rank:output:0.functional_13_1/lstm_22_1/range/delta:output:0*
_output_shapes
: f
#functional_13_1/lstm_22_1/Max/inputConst*
_output_shapes
: *
dtype0*
value
B :��
functional_13_1/lstm_22_1/MaxMax,functional_13_1/lstm_22_1/Max/input:output:0(functional_13_1/lstm_22_1/range:output:0*
T0*
_output_shapes
: �
7functional_13_1/lstm_22_1/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������y
6functional_13_1/lstm_22_1/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value
B :��
)functional_13_1/lstm_22_1/TensorArrayV2_2TensorListReserve@functional_13_1/lstm_22_1/TensorArrayV2_2/element_shape:output:0?functional_13_1/lstm_22_1/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Qfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Cfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor)functional_13_1/lstm_22_1/transpose_1:y:0Zfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:���y
$functional_13_1/lstm_22_1/zeros_likeConst*
_output_shapes

:@*
dtype0*
valueB@*    n
,functional_13_1/lstm_22_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
functional_13_1/lstm_22_1/whileWhile5functional_13_1/lstm_22_1/while/loop_counter:output:0&functional_13_1/lstm_22_1/Max:output:0'functional_13_1/lstm_22_1/time:output:02functional_13_1/lstm_22_1/TensorArrayV2_1:handle:0-functional_13_1/lstm_22_1/zeros_like:output:0(functional_13_1/lstm_22_1/zeros:output:0*functional_13_1/lstm_22_1/zeros_1:output:0Qfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Sfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Bfunctional_13_1_lstm_22_1_lstm_cell_1_cast_readvariableop_resourceDfunctional_13_1_lstm_22_1_lstm_cell_1_cast_1_readvariableop_resourceCfunctional_13_1_lstm_22_1_lstm_cell_1_add_1_readvariableop_resource*
T
2*
_num_original_outputs*D
_output_shapes2
0: : : : :@:@:@: : : : : *%
_read_only_resource_inputs
	
*(
"_xla_propagate_compile_time_consts(*7
body/R-
+functional_13_1_lstm_22_1_while_body_677016*7
cond/R-
+functional_13_1_lstm_22_1_while_cond_677015*C
output_shapes2
0: : : : :@:@:@: : : : : *
parallel_iterations �
Jfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
<functional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStackTensorListStack(functional_13_1/lstm_22_1/while:output:3Sfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:�@*
element_dtype0*
num_elements��
/functional_13_1/lstm_22_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1functional_13_1/lstm_22_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1functional_13_1/lstm_22_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)functional_13_1/lstm_22_1/strided_slice_2StridedSliceEfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack:tensor:08functional_13_1/lstm_22_1/strided_slice_2/stack:output:0:functional_13_1/lstm_22_1/strided_slice_2/stack_1:output:0:functional_13_1/lstm_22_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@*
shrink_axis_mask
*functional_13_1/lstm_22_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%functional_13_1/lstm_22_1/transpose_2	TransposeEfunctional_13_1/lstm_22_1/TensorArrayV2Stack/TensorListStack:tensor:03functional_13_1/lstm_22_1/transpose_2/perm:output:0*
T0*#
_output_shapes
:�@�
.functional_13_1/masked_attention_layer_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   �   @   �
0functional_13_1/masked_attention_layer_1/unstackUnpack7functional_13_1/masked_attention_layer_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num�
?functional_13_1/masked_attention_layer_1/Shape_1/ReadVariableOpReadVariableOpHfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
0functional_13_1/masked_attention_layer_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@      �
2functional_13_1/masked_attention_layer_1/unstack_1Unpack9functional_13_1/masked_attention_layer_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
num�
6functional_13_1/masked_attention_layer_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
0functional_13_1/masked_attention_layer_1/ReshapeReshape)functional_13_1/lstm_22_1/transpose_2:y:0?functional_13_1/masked_attention_layer_1/Reshape/shape:output:0*
T0*
_output_shapes
:	�@�
Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpReadVariableOpHfunctional_13_1_masked_attention_layer_1_shape_1_readvariableop_resource*
_output_shapes

:@*
dtype0�
7functional_13_1/masked_attention_layer_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
2functional_13_1/masked_attention_layer_1/transpose	TransposeIfunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOp:value:0@functional_13_1/masked_attention_layer_1/transpose/perm:output:0*
T0*
_output_shapes

:@�
8functional_13_1/masked_attention_layer_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   �����
2functional_13_1/masked_attention_layer_1/Reshape_1Reshape6functional_13_1/masked_attention_layer_1/transpose:y:0Afunctional_13_1/masked_attention_layer_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:@�
/functional_13_1/masked_attention_layer_1/MatMulMatMul9functional_13_1/masked_attention_layer_1/Reshape:output:0;functional_13_1/masked_attention_layer_1/Reshape_1:output:0*
T0*
_output_shapes
:	��
8functional_13_1/masked_attention_layer_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   �      �
2functional_13_1/masked_attention_layer_1/Reshape_2Reshape9functional_13_1/masked_attention_layer_1/MatMul:product:0Afunctional_13_1/masked_attention_layer_1/Reshape_2/shape:output:0*
T0*#
_output_shapes
:��
-functional_13_1/masked_attention_layer_1/TanhTanh;functional_13_1/masked_attention_layer_1/Reshape_2:output:0*
T0*#
_output_shapes
:��
-functional_13_1/masked_attention_layer_1/CastCastfunctional_13_1/Any:output:0*

DstT0*

SrcT0
*
_output_shapes
:	��
7functional_13_1/masked_attention_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
3functional_13_1/masked_attention_layer_1/ExpandDims
ExpandDims1functional_13_1/masked_attention_layer_1/Cast:y:0@functional_13_1/masked_attention_layer_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:��
,functional_13_1/masked_attention_layer_1/mulMul1functional_13_1/masked_attention_layer_1/Tanh:y:0<functional_13_1/masked_attention_layer_1/ExpandDims:output:0*
T0*#
_output_shapes
:�s
.functional_13_1/masked_attention_layer_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
,functional_13_1/masked_attention_layer_1/subSub7functional_13_1/masked_attention_layer_1/sub/x:output:0<functional_13_1/masked_attention_layer_1/ExpandDims:output:0*
T0*#
_output_shapes
:�u
0functional_13_1/masked_attention_layer_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�P�
.functional_13_1/masked_attention_layer_1/mul_1Mul9functional_13_1/masked_attention_layer_1/mul_1/x:output:00functional_13_1/masked_attention_layer_1/sub:z:0*
T0*#
_output_shapes
:��
.functional_13_1/masked_attention_layer_1/sub_1Sub0functional_13_1/masked_attention_layer_1/mul:z:02functional_13_1/masked_attention_layer_1/mul_1:z:0*
T0*#
_output_shapes
:�o
-functional_13_1/masked_attention_layer_1/RankConst*
_output_shapes
: *
dtype0*
value	B :q
/functional_13_1/masked_attention_layer_1/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :r
0functional_13_1/masked_attention_layer_1/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/Sub_2Sub8functional_13_1/masked_attention_layer_1/Rank_1:output:09functional_13_1/masked_attention_layer_1/Sub_2/y:output:0*
T0*
_output_shapes
: v
4functional_13_1/masked_attention_layer_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : v
4functional_13_1/masked_attention_layer_1/range/limitConst*
_output_shapes
: *
dtype0*
value	B :v
4functional_13_1/masked_attention_layer_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/rangeRange=functional_13_1/masked_attention_layer_1/range/start:output:0=functional_13_1/masked_attention_layer_1/range/limit:output:0=functional_13_1/masked_attention_layer_1/range/delta:output:0*
_output_shapes
:x
6functional_13_1/masked_attention_layer_1/range_1/startConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_1Range?functional_13_1/masked_attention_layer_1/range_1/start:output:02functional_13_1/masked_attention_layer_1/Sub_2:z:0?functional_13_1/masked_attention_layer_1/range_1/delta:output:0*
_output_shapes
: �
8functional_13_1/masked_attention_layer_1/concat/values_1Pack2functional_13_1/masked_attention_layer_1/Sub_2:z:0*
N*
T0*
_output_shapes
:�
8functional_13_1/masked_attention_layer_1/concat/values_3Const*
_output_shapes
:*
dtype0*
valueB:v
4functional_13_1/masked_attention_layer_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/functional_13_1/masked_attention_layer_1/concatConcatV27functional_13_1/masked_attention_layer_1/range:output:0Afunctional_13_1/masked_attention_layer_1/concat/values_1:output:09functional_13_1/masked_attention_layer_1/range_1:output:0Afunctional_13_1/masked_attention_layer_1/concat/values_3:output:0=functional_13_1/masked_attention_layer_1/concat/axis:output:0*
N*
T0*
_output_shapes
:�
4functional_13_1/masked_attention_layer_1/transpose_1	Transpose2functional_13_1/masked_attention_layer_1/sub_1:z:08functional_13_1/masked_attention_layer_1/concat:output:0*
T0*#
_output_shapes
:��
0functional_13_1/masked_attention_layer_1/SoftmaxSoftmax8functional_13_1/masked_attention_layer_1/transpose_1:y:0*
T0*#
_output_shapes
:�r
0functional_13_1/masked_attention_layer_1/Sub_3/yConst*
_output_shapes
: *
dtype0*
value	B :�
.functional_13_1/masked_attention_layer_1/Sub_3Sub8functional_13_1/masked_attention_layer_1/Rank_1:output:09functional_13_1/masked_attention_layer_1/Sub_3/y:output:0*
T0*
_output_shapes
: x
6functional_13_1/masked_attention_layer_1/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : x
6functional_13_1/masked_attention_layer_1/range_2/limitConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_2Range?functional_13_1/masked_attention_layer_1/range_2/start:output:0?functional_13_1/masked_attention_layer_1/range_2/limit:output:0?functional_13_1/masked_attention_layer_1/range_2/delta:output:0*
_output_shapes
:x
6functional_13_1/masked_attention_layer_1/range_3/startConst*
_output_shapes
: *
dtype0*
value	B :x
6functional_13_1/masked_attention_layer_1/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
0functional_13_1/masked_attention_layer_1/range_3Range?functional_13_1/masked_attention_layer_1/range_3/start:output:02functional_13_1/masked_attention_layer_1/Sub_3:z:0?functional_13_1/masked_attention_layer_1/range_3/delta:output:0*
_output_shapes
: �
:functional_13_1/masked_attention_layer_1/concat_1/values_1Pack2functional_13_1/masked_attention_layer_1/Sub_3:z:0*
N*
T0*
_output_shapes
:�
:functional_13_1/masked_attention_layer_1/concat_1/values_3Const*
_output_shapes
:*
dtype0*
valueB:x
6functional_13_1/masked_attention_layer_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
1functional_13_1/masked_attention_layer_1/concat_1ConcatV29functional_13_1/masked_attention_layer_1/range_2:output:0Cfunctional_13_1/masked_attention_layer_1/concat_1/values_1:output:09functional_13_1/masked_attention_layer_1/range_3:output:0Cfunctional_13_1/masked_attention_layer_1/concat_1/values_3:output:0?functional_13_1/masked_attention_layer_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
4functional_13_1/masked_attention_layer_1/transpose_2	Transpose:functional_13_1/masked_attention_layer_1/Softmax:softmax:0:functional_13_1/masked_attention_layer_1/concat_1:output:0*
T0*#
_output_shapes
:��
.functional_13_1/masked_attention_layer_1/mul_2Mul)functional_13_1/lstm_22_1/transpose_2:y:08functional_13_1/masked_attention_layer_1/transpose_2:y:0*
T0*#
_output_shapes
:�@�
>functional_13_1/masked_attention_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
,functional_13_1/masked_attention_layer_1/SumSum2functional_13_1/masked_attention_layer_1/mul_2:z:0Gfunctional_13_1/masked_attention_layer_1/Sum/reduction_indices:output:0*
T0*
_output_shapes

:@�
.functional_13_1/dense_26_1/Cast/ReadVariableOpReadVariableOp7functional_13_1_dense_26_1_cast_readvariableop_resource*
_output_shapes

:@@*
dtype0�
!functional_13_1/dense_26_1/MatMulMatMul5functional_13_1/masked_attention_layer_1/Sum:output:06functional_13_1/dense_26_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
-functional_13_1/dense_26_1/Add/ReadVariableOpReadVariableOp6functional_13_1_dense_26_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_13_1/dense_26_1/AddAddV2+functional_13_1/dense_26_1/MatMul:product:05functional_13_1/dense_26_1/Add/ReadVariableOp:value:0*
T0*
_output_shapes

:@t
functional_13_1/dense_26_1/TanhTanh"functional_13_1/dense_26_1/Add:z:0*
T0*
_output_shapes

:@�
.functional_13_1/dense_27_1/Cast/ReadVariableOpReadVariableOp7functional_13_1_dense_27_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
!functional_13_1/dense_27_1/MatMulMatMul#functional_13_1/dense_26_1/Tanh:y:06functional_13_1/dense_27_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes

:�
-functional_13_1/dense_27_1/Add/ReadVariableOpReadVariableOp6functional_13_1_dense_27_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_13_1/dense_27_1/AddAddV2+functional_13_1/dense_27_1/MatMul:product:05functional_13_1/dense_27_1/Add/ReadVariableOp:value:0*
T0*
_output_shapes

:z
"functional_13_1/dense_27_1/SoftmaxSoftmax"functional_13_1/dense_27_1/Add:z:0*
T0*
_output_shapes

:r
IdentityIdentity,functional_13_1/dense_27_1/Softmax:softmax:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp.^functional_13_1/dense_26_1/Add/ReadVariableOp/^functional_13_1/dense_26_1/Cast/ReadVariableOp.^functional_13_1/dense_27_1/Add/ReadVariableOp/^functional_13_1/dense_27_1/Cast/ReadVariableOp:^functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp<^functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp;^functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp ^functional_13_1/lstm_22_1/whileB^functional_13_1/masked_attention_layer_1/transpose/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :��: : : : : : : : *
	_noinline(2^
-functional_13_1/dense_26_1/Add/ReadVariableOp-functional_13_1/dense_26_1/Add/ReadVariableOp2`
.functional_13_1/dense_26_1/Cast/ReadVariableOp.functional_13_1/dense_26_1/Cast/ReadVariableOp2^
-functional_13_1/dense_27_1/Add/ReadVariableOp-functional_13_1/dense_27_1/Add/ReadVariableOp2`
.functional_13_1/dense_27_1/Cast/ReadVariableOp.functional_13_1/dense_27_1/Cast/ReadVariableOp2v
9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp9functional_13_1/lstm_22_1/lstm_cell_1/Cast/ReadVariableOp2z
;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp;functional_13_1/lstm_22_1/lstm_cell_1/Cast_1/ReadVariableOp2x
:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp:functional_13_1/lstm_22_1/lstm_cell_1/add_1/ReadVariableOp2B
functional_13_1/lstm_22_1/whilefunctional_13_1/lstm_22_1/while2�
Afunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOpAfunctional_13_1/masked_attention_layer_1/transpose/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C
$
_output_shapes
:��

_user_specified_namex
�n
�
+functional_13_1_lstm_22_1_while_body_677016P
Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counterA
=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max/
+functional_13_1_lstm_22_1_while_placeholder1
-functional_13_1_lstm_22_1_while_placeholder_11
-functional_13_1_lstm_22_1_while_placeholder_21
-functional_13_1_lstm_22_1_while_placeholder_31
-functional_13_1_lstm_22_1_while_placeholder_4�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0^
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0:
��_
Lfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0:	@�Z
Kfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0:	�,
(functional_13_1_lstm_22_1_while_identity.
*functional_13_1_lstm_22_1_while_identity_1.
*functional_13_1_lstm_22_1_while_identity_2.
*functional_13_1_lstm_22_1_while_identity_3.
*functional_13_1_lstm_22_1_while_identity_4.
*functional_13_1_lstm_22_1_while_identity_5.
*functional_13_1_lstm_22_1_while_identity_6�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor\
Hfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource:
��]
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource:	@�X
Ifunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource:	���?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp�Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp�@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp�
Qfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"   �   �
Cfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0+functional_13_1_lstm_22_1_while_placeholderZfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes
:	�*
element_dtype0�
Sfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Efunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0+functional_13_1_lstm_22_1_while_placeholder\functional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0
�
?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOpReadVariableOpJfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0* 
_output_shapes
:
��*
dtype0�
2functional_13_1/lstm_22_1/while/lstm_cell_1/MatMulMatMulJfunctional_13_1/lstm_22_1/while/TensorArrayV2Read/TensorListGetItem:item:0Gfunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpReadVariableOpLfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
4functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul_1MatMul-functional_13_1_lstm_22_1_while_placeholder_3Ifunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
/functional_13_1/lstm_22_1/while/lstm_cell_1/addAddV2<functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul:product:0>functional_13_1/lstm_22_1/while/lstm_cell_1/MatMul_1:product:0*
T0*
_output_shapes
:	��
@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOpReadVariableOpKfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0*
_output_shapes	
:�*
dtype0�
1functional_13_1/lstm_22_1/while/lstm_cell_1/add_1AddV23functional_13_1/lstm_22_1/while/lstm_cell_1/add:z:0Hfunctional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�}
;functional_13_1/lstm_22_1/while/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
1functional_13_1/lstm_22_1/while/lstm_cell_1/splitSplitDfunctional_13_1/lstm_22_1/while/lstm_cell_1/split/split_dim:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/add_1:z:0*
T0*<
_output_shapes*
(:@:@:@:@*
	num_split�
3functional_13_1/lstm_22_1/while/lstm_cell_1/SigmoidSigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:0*
T0*
_output_shapes

:@�
5functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_1Sigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:1*
T0*
_output_shapes

:@�
/functional_13_1/lstm_22_1/while/lstm_cell_1/mulMul9functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_1:y:0-functional_13_1_lstm_22_1_while_placeholder_4*
T0*
_output_shapes

:@�
0functional_13_1/lstm_22_1/while/lstm_cell_1/TanhTanh:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:2*
T0*
_output_shapes

:@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/mul_1Mul7functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid:y:04functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh:y:0*
T0*
_output_shapes

:@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/add_2AddV23functional_13_1/lstm_22_1/while/lstm_cell_1/mul:z:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_1:z:0*
T0*
_output_shapes

:@�
5functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_2Sigmoid:functional_13_1/lstm_22_1/while/lstm_cell_1/split:output:3*
T0*
_output_shapes

:@�
2functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh_1Tanh5functional_13_1/lstm_22_1/while/lstm_cell_1/add_2:z:0*
T0*
_output_shapes

:@�
1functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2Mul9functional_13_1/lstm_22_1/while/lstm_cell_1/Sigmoid_2:y:06functional_13_1/lstm_22_1/while/lstm_cell_1/Tanh_1:y:0*
T0*
_output_shapes

:@
.functional_13_1/lstm_22_1/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
$functional_13_1/lstm_22_1/while/TileTileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:07functional_13_1/lstm_22_1/while/Tile/multiples:output:0*
T0
*
_output_shapes

:�
(functional_13_1/lstm_22_1/while/SelectV2SelectV2-functional_13_1/lstm_22_1/while/Tile:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2:z:0-functional_13_1_lstm_22_1_while_placeholder_2*
T0*
_output_shapes

:@�
0functional_13_1/lstm_22_1/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
&functional_13_1/lstm_22_1/while/Tile_1TileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:09functional_13_1/lstm_22_1/while/Tile_1/multiples:output:0*
T0
*
_output_shapes

:�
0functional_13_1/lstm_22_1/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
&functional_13_1/lstm_22_1/while/Tile_2TileLfunctional_13_1/lstm_22_1/while/TensorArrayV2Read_1/TensorListGetItem:item:09functional_13_1/lstm_22_1/while/Tile_2/multiples:output:0*
T0
*
_output_shapes

:�
*functional_13_1/lstm_22_1/while/SelectV2_1SelectV2/functional_13_1/lstm_22_1/while/Tile_1:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/mul_2:z:0-functional_13_1_lstm_22_1_while_placeholder_3*
T0*
_output_shapes

:@�
*functional_13_1/lstm_22_1/while/SelectV2_2SelectV2/functional_13_1/lstm_22_1/while/Tile_2:output:05functional_13_1/lstm_22_1/while/lstm_cell_1/add_2:z:0-functional_13_1_lstm_22_1_while_placeholder_4*
T0*
_output_shapes

:@�
Dfunctional_13_1/lstm_22_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-functional_13_1_lstm_22_1_while_placeholder_1+functional_13_1_lstm_22_1_while_placeholder1functional_13_1/lstm_22_1/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���g
%functional_13_1/lstm_22_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
#functional_13_1/lstm_22_1/while/addAddV2+functional_13_1_lstm_22_1_while_placeholder.functional_13_1/lstm_22_1/while/add/y:output:0*
T0*
_output_shapes
: i
'functional_13_1/lstm_22_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
%functional_13_1/lstm_22_1/while/add_1AddV2Lfunctional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_while_loop_counter0functional_13_1/lstm_22_1/while/add_1/y:output:0*
T0*
_output_shapes
: �
(functional_13_1/lstm_22_1/while/IdentityIdentity)functional_13_1/lstm_22_1/while/add_1:z:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_1Identity=functional_13_1_lstm_22_1_while_functional_13_1_lstm_22_1_max%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_2Identity'functional_13_1/lstm_22_1/while/add:z:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_3IdentityTfunctional_13_1/lstm_22_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes
: �
*functional_13_1/lstm_22_1/while/Identity_4Identity1functional_13_1/lstm_22_1/while/SelectV2:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes

:@�
*functional_13_1/lstm_22_1/while/Identity_5Identity3functional_13_1/lstm_22_1/while/SelectV2_1:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes

:@�
*functional_13_1/lstm_22_1/while/Identity_6Identity3functional_13_1/lstm_22_1/while/SelectV2_2:output:0%^functional_13_1/lstm_22_1/while/NoOp*
T0*
_output_shapes

:@�
$functional_13_1/lstm_22_1/while/NoOpNoOp@^functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOpB^functional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpA^functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp*
_output_shapes
 "a
*functional_13_1_lstm_22_1_while_identity_13functional_13_1/lstm_22_1/while/Identity_1:output:0"a
*functional_13_1_lstm_22_1_while_identity_23functional_13_1/lstm_22_1/while/Identity_2:output:0"a
*functional_13_1_lstm_22_1_while_identity_33functional_13_1/lstm_22_1/while/Identity_3:output:0"a
*functional_13_1_lstm_22_1_while_identity_43functional_13_1/lstm_22_1/while/Identity_4:output:0"a
*functional_13_1_lstm_22_1_while_identity_53functional_13_1/lstm_22_1/while/Identity_5:output:0"a
*functional_13_1_lstm_22_1_while_identity_63functional_13_1/lstm_22_1/while/Identity_6:output:0"]
(functional_13_1_lstm_22_1_while_identity1functional_13_1/lstm_22_1/while/Identity:output:0"�
Ifunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resourceKfunctional_13_1_lstm_22_1_while_lstm_cell_1_add_1_readvariableop_resource_0"�
Jfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resourceLfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_1_readvariableop_resource_0"�
Hfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resourceJfunctional_13_1_lstm_22_1_while_lstm_cell_1_cast_readvariableop_resource_0"�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor�functional_13_1_lstm_22_1_while_tensorarrayv2read_1_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_1_tensorlistfromtensor_0"�
�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor�functional_13_1_lstm_22_1_while_tensorarrayv2read_tensorlistgetitem_functional_13_1_lstm_22_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : :@:@:@: : : : : 2�
?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp?functional_13_1/lstm_22_1/while/lstm_cell_1/Cast/ReadVariableOp2�
Afunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOpAfunctional_13_1/lstm_22_1/while/lstm_cell_1/Cast_1/ReadVariableOp2�
@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp@functional_13_1/lstm_22_1/while/lstm_cell_1/add_1/ReadVariableOp:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:{w

_output_shapes
: 
]
_user_specified_nameECfunctional_13_1/lstm_22_1/TensorArrayUnstack_1/TensorListFromTensor:yu

_output_shapes
: 
[
_user_specified_nameCAfunctional_13_1/lstm_22_1/TensorArrayUnstack/TensorListFromTensor:$ 

_output_shapes

:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@:

_output_shapes
: :

_output_shapes
: :UQ

_output_shapes
: 
7
_user_specified_namefunctional_13_1/lstm_22_1/Max:d `

_output_shapes
: 
F
_user_specified_name.,functional_13_1/lstm_22_1/while/loop_counter"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default
,
x'
serving_default_x:0��3
output_0'
StatefulPartitionedCall:0tensorflow/serving/predict:�0
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
_default_save_signature

signatures"
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
Q
0
2
4
5
6
7
8"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
_velocities"
_generic_user_object
�
trace_02�
"__inference_serving_default_677482�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *#� 
������������ztrace_0
,
serving_default"
signature_map
y
_inbound_nodes
_outbound_nodes
_losses
 	_loss_ids
!_losses_override"
_generic_user_object
y
"_inbound_nodes
#_outbound_nodes
$_losses
%	_loss_ids
&_losses_override"
_generic_user_object
�
'cell
(_inbound_nodes
)_outbound_nodes
*_losses
+	_loss_ids
,_losses_override
-
state_size
._build_shapes_dict"
_generic_user_object
�
/W
0_inbound_nodes
1_outbound_nodes
2_losses
3	_loss_ids
4_losses_override
5_build_shapes_dict"
_generic_user_object
�
6_kernel
7bias
8_inbound_nodes
9_outbound_nodes
:_losses
;	_loss_ids
<_losses_override
=_build_shapes_dict"
_generic_user_object
y
>_inbound_nodes
?_outbound_nodes
@_losses
A	_loss_ids
B_losses_override"
_generic_user_object
�
C_kernel
Dbias
E_inbound_nodes
F_outbound_nodes
G_losses
H	_loss_ids
I_losses_override
J_build_shapes_dict"
_generic_user_object
�
0
1
K2
L3
M4
N5
O6
P7
Q8
R9
S10
T11
U12
V13
W14
X15
Y16
Z17"
trackable_list_wrapper
X
[0
\1
]2
/3
64
75
C6
D7"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
: 2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
"__inference_serving_default_677482inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_677215x"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
jx
kwonlydefaults
 
annotations� *
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
trackable_list_wrapper
�

[kernel
\recurrent_kernel
]bias
^_inbound_nodes
__outbound_nodes
`_losses
a	_loss_ids
b_losses_override
c
state_size
d_build_shapes_dict"
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
trackable_list_wrapper
 "
trackable_dict_wrapper
3:1@2!masked_attention_layer/att_weight
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
!:@@2dense_26/kernel
:@2dense_26/bias
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
!:@2dense_27/kernel
:2dense_27/bias
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
8:6
��2&adam/lstm_22_lstm_cell_kernel_momentum
8:6
��2&adam/lstm_22_lstm_cell_kernel_velocity
A:?	@�20adam/lstm_22_lstm_cell_recurrent_kernel_momentum
A:?	@�20adam/lstm_22_lstm_cell_recurrent_kernel_velocity
1:/�2$adam/lstm_22_lstm_cell_bias_momentum
1:/�2$adam/lstm_22_lstm_cell_bias_velocity
?:=@2/adam/masked_attention_layer_att_weight_momentum
?:=@2/adam/masked_attention_layer_att_weight_velocity
-:+@@2adam/dense_26_kernel_momentum
-:+@@2adam/dense_26_kernel_velocity
':%@2adam/dense_26_bias_momentum
':%@2adam/dense_26_bias_velocity
-:+@2adam/dense_27_kernel_momentum
-:+@2adam/dense_27_kernel_velocity
':%2adam/dense_27_bias_momentum
':%2adam/dense_27_bias_velocity
,:*
��2lstm_22/lstm_cell/kernel
5:3	@�2"lstm_22/lstm_cell/recurrent_kernel
%:#�2lstm_22/lstm_cell/bias
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
trackable_list_wrapper
 "
trackable_dict_wrapper�
"__inference_serving_default_677482d[\]/67CD5�2
+�(
&�#
inputs�����������
� "!�
unknown����������
$__inference_signature_wrapper_677215d[\]/67CD,�)
� 
"�

x�
x��"*�'
%
output_0�
output_0