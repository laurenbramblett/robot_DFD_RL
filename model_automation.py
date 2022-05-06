import numpy as np
import pandas as pd

import sklearn as skl
from sklearn import preprocessing as pp

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from run_vanilla_sim import run_vanilla_sim

# from keras.wrappers.scikit_learn import KerasClassifier #Deprecated
from scikeras.wrappers import KerasClassifier, KerasRegressor


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')



draw=False
train_frac=2/3 #fraction of data to train on

batch_sizes=[32,64,128]
HL_scales=[1,2,4]
optimizers=['SGD','Adam']
learning_rates=[0.1,0.01,0.001]
epochs=100


def createNN(layer_params=[],compile_params=[[],{'loss':'mean_squared_error', 'optimizer':'adam', 'metrics':['mse']}],model=Sequential):
    """Construct an arbitrary Keras layered model
        layer_params and compile params must have form list(iterable(list,dict)),
            ex: [[[layer1_args],{layer1_kwarg1:val,layer1_kwarg2:val}],
                [[layer2_args],{layer2_kwarg1:val,layer2_kwarg2:val}]]
            While this may seem cumbersome, it provides full flexibility in creating the model object
    """
    model=model()
    for i in range(len(layer_params)):
        args=layer_params[i][0]
#         print(args)
        kwargs=layer_params[i][1]
#         print(kwargs)
        model.add(Dense(*args,**kwargs))
    model.compile(*compile_params[0],**compile_params[1])
    return model

def bulk_layers(n,layer_params):
    """Construct appropriate input dicts for bulk of the same layer"""
    return [layer_params for i in range(n)]



def get_driver_layers(HL_scale,input_dim,activation='relu',base=64):
    """Construct layer_params for createNN() in accordance with the NN architecture we want for our controller
        Layer0: base # neurons, input dim set to match features
        Layer1-3: base*HL_scale neurons
        Layer4: 1 output, linear activation. Always used to produce the final angle for the controller
    """
    layers=[[[base], {'input_dim':input_dim, 'activation':activation}],
[[base*HL_scale], {'activation':activation}],
[[base*HL_scale], {'activation':activation}],
[[1], {'activation':'linear'}]]
    return layers

def simulate_model(model,worlds,scaler,batch_size,draw):
    """runs the simulator with a given model, along a set of worlds. returns the total number of succesful runs, along with a world by world Pass/Fail"""
    world_pf={}
    world_c_iters={}
    pf_sum=0
    for w in worlds:
        print("testing on World: %d"%w)
        ctrl_iters,pas=run_vanilla_sim(model,scaler,batch_size,w,draw)
        world_pf[w]=pas
        world_c_iters[w]=ctrl_iters
        pf_sum+=pas

    avg_ctrl_iters=np.array(list(world_c_iters.values())).mean()

    return pf_sum,avg_ctrl_iters,world_pf,world_c_iters


#We ultimately opted for a grid search rather than a random search,but this func facilitates the latter
def get_random_model(layer_sizes,activations,depth,input_dim,output_dim,compile_params=[[],{'loss':'mean_squared_error', 'optimizer':'adam', 'metrics':['mse']}]):
    rng=np.random.default_rng()
    lyrs=rng.choice(layer_sizes,size=depth)
    activ=rng.choice(activations,size=depth+1)
    layers=[[[lyrs[0]],{'input_dim':input_dim,'activation':activ[0]}]] #create input layer
    for i in range(1,depth):
        layers.append([[lyrs[i]],{'activation':activ[i]}])
    layers.append([[1],{'activation':activ[-1]}]) #create output layer

    return layers



raw_data=pd.read_csv('ObsRecordUniformRandom_v6.csv')
# world_data=pd.read_csv('characteristics_by_world.csv')
lidar_list = ["Lidar"+str(x) for x in range(0,32)]
state_list = ["goalDist","goalAng","forceAng","World"]
partial_header = ["goalDist","goalAng","forceAng"]
headers = lidar_list + ["goalDist","goalAng","forceAng","world"]
raw_data.columns = headers


ttr=(1-train_frac)/train_frac #train_test_ratio: 2 training worlds for each test world

world_ids=range(len(raw_data['world'].unique()))
test_world_ids=[w for w in world_ids if w%(ttr+1)==0 ]

h=headers.copy()
h.remove('world')
x = raw_data.values #returns a numpy array
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_data = pd.DataFrame(x_scaled)

normalized_data.columns = headers
normalized_data['world']=raw_data['world']

test_data=normalized_data[normalized_data['world'].isin(test_world_ids)]
train_data=normalized_data[~normalized_data['world'].isin(test_world_ids)]

Xtrn = np.array(train_data.iloc[:,0:-2])
ytrn = np.array(train_data.iloc[:,-2:-1])
Xtst = np.array(test_data.iloc[:,0:-2])
ytst = np.array(test_data.iloc[:,-2:-1])


results={}
score_dict={}


mn=1
for b_sz in batch_sizes:
    for hl_scale in HL_scales:
        for lr in learning_rates:
            for opt in optimizers:
                opti=None
                if opt=='Adam': #This makes indexing easier later,
                    opti=Adam(learning_rate=lr)
                if opt=='SGD':
                    opti=SGD(learning_rate=lr)
                test_params=(b_sz,hl_scale,lr,opt)
                print("\n Model No. %d"%mn)
                print(test_params)
                layers=get_driver_layers(hl_scale,len(headers)-2,'relu')
                compile_params=[[],{'loss':'mean_squared_error', 'optimizer':opti, 'metrics':['mse']}]
                model=createNN(layer_params=layers,compile_params=compile_params)
                model.fit(Xtrn,ytrn,epochs=epochs,batch_size=b_sz)

                _,trn_score=model.evaluate(Xtrn,ytrn)

                _,test_score=model.evaluate(Xtst,ytst)

                pf_sum,avg_ctrl_iters,pf_by_world,ctrl_iters_by_world=simulate_model(model,test_world_ids,min_max_scaler,b_sz,draw)

                results[test_params]={'layers':layers,'model':model,
                                      'training score':trn_score,'test score':test_score,
                                      'pf_score':pf_sum,'pf_by_world':pf_by_world,
                                      'avg_ctrl_iters':avg_ctrl_iters,'ctrl_iters_by_world':ctrl_iters_by_world}
                print(test_score)
                score_dict[test_params]=[trn_score,test_score,pf_sum,avg_ctrl_iters]
                mn+=1

idx=score_dict.keys()
vals=[score_dict[i] for i in idx]
idx
Score_df=pd.DataFrame(vals,index=idx,columns=['Training Score','Testing Score','NN Controller Pass Count','NN AVG Controller Iterations'])
Score_df.index.names=['Batch Size','HL_Scaler','Learning Rate','Optimizer']

import time
tim=str(int(time.time()))
Score_df.to_csv(tim+r'scores.csv')
