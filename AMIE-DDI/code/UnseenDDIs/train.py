# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import sys
from utils import *
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from dataset_processed import Graph_Bert_Dataset_fine_tune
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# %%
from model import *
from model_to import *
# %%
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf:
    print(module.__name__, module.__version__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# %%
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# %%
def input_solver1(sample,sample1,sample2,sample3,sample4,sample5,\
    sample6,sample7,sample8,sample9,sample10,sample11,sample12,sample13,sample14,sample15,sample16):
    return {'molecule_sequence1': sample,'molecule_sequence2': sample1, 'adj_matrix1': sample2,
           'adj_matrix2': sample3,'dist_matrix1': sample4,'dist_matrix2': sample5,
           'atom_features1':sample6,'atom_features2':sample7,'adjoin_matrix1_atom':sample8,
           'adjoin_matrix2_atom':sample9,'dist_matrix1_atom':sample10,'dist_matrix2_atom':sample11,
           'atom_match_matrix1':sample12,'atom_match_matrix2':sample13,'sum_atoms1':sample14,'sum_atoms2':sample15}, sample16

dataFolder = './data/Classification/UnseenDDIs'
tr_dataset = pd.read_csv(dataFolder + '/tr_dataset.csv')
val_dataset = pd.read_csv(dataFolder + '/val_dataset.csv')
tst_dataset = pd.read_csv(dataFolder + '/tst_dataset.csv')

# %%
tokenizer = Mol_Tokenizer('./code/Classification/UnseenDDIs/token_id.json')

# %%
map_dict = np.load('./code/Classification/UnseenDDIs/preprocessed_drug_info.npy',allow_pickle=True).item()

# %%
train_dataset_,validation_dataset, test_dataset_ = Graph_Bert_Dataset_fine_tune(tr_dataset,val_dataset,tst_dataset,label_field='DDI',tokenizer=tokenizer,map_dict=map_dict,batch_size = 64).get_data()
train_dataset = train_dataset_.map(input_solver1)
val_dataset = validation_dataset.map(input_solver1)
test_dataset = test_dataset_.map(input_solver1)

# %%
param = {'name': 'Small', 'num_layers': 4, 'num_heads': 8, 'd_model': 256}

# %%
arch = param   ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']*2
dff = d_model
input_vocab_size = tokenizer.get_vocab_size
dropout_rate = 0.1
training = False
# %%
## motif_level inputs
motif_input1 = Input(shape=(None,), name = "molecule_sequences1")
motif_input2 = Input(shape=(None,), name = "molecule_sequences2")
motif_adj_input1 = Input(shape=(None,None), name= "adj_matrixs1")
motif_adj_input2 = Input(shape=(None,None), name= "adj_matrixs2")
motif_dist_input1 = Input(shape=(None,None), name= "dist_matrixs1")
motif_dist_input2 = Input(shape=(None,None), name= "dist_matrixs2")
### atom_level inputs
atom_input1 = Input(shape=(None,61), name = "atom_feature1")
atom_input2 = Input(shape=(None,61), name = "atom_feature2")
atom_adj_input1 = Input(shape=(None,None), name= "atom_adj_matrixs1")
atom_adj_input2 = Input(shape=(None,None), name= "atom_adj_matrixs2")
atom_dist_input1 = Input(shape=(None,None), name= "atom_dist_matrixs1")
atom_dist_input2 = Input(shape=(None,None), name= "atom_dist_matrixs2")
atom_match_matrixs1 = Input(shape=(None,None), name= "atom_match_matrixs1")
atom_match_matrixs2 = Input(shape=(None,None), name= "atom_match_matrixs2")
sum_atom1 = Input(shape=(None,None), name= "sum_atom1")
sum_atom2 = Input(shape=(None,None), name= "sum_atom2")

# %%

Outseq1,Outseq2, *_, encoder_padding_mask_atom1, encoder_padding_mask_motif1, encoder_padding_mask_atom2, encoder_padding_mask_motif2 = EncoderModel(
    num_layers=2,
    d_model=arch['d_model'],
    dff=dff,
    num_heads=num_heads,
    input_vocab_size=input_vocab_size
)(
    atom_input1,
    atom_input2,
    motif_input1,
    motif_input2,
    adjoin_matrix_atom1=atom_adj_input1,
    adjoin_matrix_atom2=atom_adj_input2,
    dist_matrix_atom1=atom_dist_input1,
    dist_matrix_atom2=atom_dist_input2,
    atom_match_matrix1=atom_match_matrixs1,
    atom_match_matrix2=atom_match_matrixs2,
    sum_atom1=sum_atom1,
    sum_atom2=sum_atom2,
    adjoin_matrix_motif1=motif_adj_input1,
    adjoin_matrix_motif2=motif_adj_input2,
    dist_matrix_motif1=motif_dist_input1,
    dist_matrix_motif2=motif_dist_input2,
    training=training
)


model_motif = Model(
    inputs=[atom_input1,atom_input2, motif_input1, motif_input2, atom_adj_input1, atom_adj_input2, atom_dist_input1, atom_dist_input2, atom_match_matrixs1, atom_match_matrixs2, sum_atom1, sum_atom2, motif_adj_input1, motif_adj_input2, motif_dist_input1, motif_dist_input2],
    outputs=[Outseq1,Outseq2, encoder_padding_mask_atom1, encoder_padding_mask_motif1, encoder_padding_mask_atom2, encoder_padding_mask_motif2]
)

# %%
### Build dual inputs
### motif-level inputs
motif_inputs1 = Input(shape=(None,), name= "molecule_sequence1")
motif_inputs2 = Input(shape=(None,), name= "molecule_sequence2")
# mask_inputs1 = create_padding_mask(inputs1)
# mask_inputs2 = create_padding_mask(inputs2)
motif_adj_inputs1 = Input(shape=(None,None), name= "adj_matrix1")
motif_adj_inputs2 = Input(shape=(None,None), name= "adj_matrix2")
motif_dist_inputs1 = Input(shape=(None,None), name= "dist_matrix1")
motif_dist_inputs2 = Input(shape=(None,None), name= "dist_matrix2")
### atom level inputs
atom_inputs1 = Input(shape=(None,61), name = "atom_features1")
atom_inputs2 = Input(shape=(None,61), name = "atom_features2")
atom_adj_inputs1 = Input(shape=(None,None), name= "adjoin_matrix1_atom")
atom_adj_inputs2 = Input(shape=(None,None), name= "adjoin_matrix2_atom")
atom_dist_inputs1 = Input(shape=(None,None), name= "dist_matrix1_atom")
atom_dist_inputs2 = Input(shape=(None,None), name= "dist_matrix2_atom")
atom_match_matrix1 = Input(shape=(None,None), name= "atom_match_matrix1")
atom_match_matrix2 = Input(shape=(None,None), name= "atom_match_matrix2")
sum_atoms1 = Input(shape=(None,None), name= "sum_atoms1")
sum_atoms2 = Input(shape=(None,None), name= "sum_atoms2")

# %%
# build weight sharing model
druga_trans,drugb_trans,encoder_padding_mask_atom1, encoder_padding_mask_motif1, encoder_padding_mask_atom2, encoder_padding_mask_motif2 = model_motif([atom_inputs1,atom_inputs2, motif_inputs1, motif_inputs2, atom_adj_inputs1, atom_adj_inputs2, atom_dist_inputs1, atom_dist_inputs2, atom_match_matrix1, atom_match_matrix2, sum_atoms1, sum_atoms2, motif_adj_inputs1, motif_adj_inputs2, motif_dist_inputs1, motif_dist_inputs2])

# %%
# build co-attention layers and Fcls
Co_attention_layers = Co_Attention_Layer(d_model,k = 128,num_heads=8,temperature=1.0,name = 'Co_attention_layer')
#BAN = BANLayer1D(v_dim=d_model, q_dim=d_model, h_dim=d_model, h_out=2)
fc1 = tf.keras.layers.Dense(d_model/2, activation='relu')
dropout1 = tf.keras.layers.Dropout(dropout_rate)
fc2 = tf.keras.layers.Dense(d_model/4, activation='relu')
dropout2 = tf.keras.layers.Dropout(dropout_rate)
fc3 = tf.keras.layers.Dense(4,activation='softmax')
# %%
### To avoid high similarity scores
Wa = tf.keras.layers.Dense(d_model)
Wb = tf.keras.layers.Dense(d_model)

# %%
druga_trans_,drugb_trans_,*_ = Co_attention_layers([Wa(druga_trans),Wb(drugb_trans)])
output1_2 = tf.keras.layers.Concatenate()([druga_trans_,drugb_trans_])
output1_2 = fc1(output1_2)
output1_2 = dropout1(output1_2,training=training)
output1_2 = fc2(output1_2)
output1_2 = dropout2(output1_2,training=training)
output1_2 = fc3(output1_2)

# %%
models = Model(inputs=[atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
    ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1,
    atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
    ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2],outputs =[output1_2])

# %%
models.summary()

# %%
opt = Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
models.compile(loss=loss,
            optimizer=opt,
            metrics=['sparse_categorical_accuracy'])
# %%
## Callbacks setting

filepath = './code/Classification/UnseenDDIs/Saved_weights/model.h5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                 patience=5,
                                                 min_delta=5e-4,
                                                 mode='max')

class TemperatureAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, co_attention_layer, initial_temperature, final_temperature, total_epochs):
        self.co_attention_layer = co_attention_layer
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Linear annealing of temperature
        '''
        new_temperature = self.initial_temperature - (
            (self.initial_temperature - self.final_temperature) * (epoch / self.total_epochs)
        )
        # Update the temperature parameter of the Co-Attention Layer
        self.co_attention_layer.temperature = new_temperature
        '''
        new_temperature = self.initial_temperature * ((self.final_temperature / self.initial_temperature) ** (epoch / self.total_epochs))
        # Update the temperature parameter of the Co-Attention Layer
        self.co_attention_layer.temperature = new_temperature

        #temperature_history.append(new_temperature)
        print(f"\nEpoch {epoch + 1}: Updated temperature to {new_temperature:.4f}")

initial_temperature = 1  # Start with a higher temperature
final_temperature = 0.1    # End with a lower temperature
total_epochs = 50          # Total number of epochs for training

temperature_annealing_callback = TemperatureAnnealingCallback(Co_attention_layers, initial_temperature, final_temperature, total_epochs)

callbacks = [checkpoint, earlystopping, temperature_annealing_callback]
# %%
tf.config.run_functions_eagerly(True)
# %%
history = models.fit(train_dataset,
         epochs = 50,callbacks = callbacks,
         validation_data = val_dataset)

final_temperature_stage1 = Co_attention_layers.temperature

opt = Adam(learning_rate = 0.00001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

models.compile(loss = loss,
            optimizer=opt,
            metrics=['sparse_categorical_accuracy'])

filepath = './code/Classification/UnseenDDIs/Saved_weights/model.h5'

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                 patience=5,
                                                 min_delta=5e-4,
                                                 mode='max')

temperature_annealing_callback_stage2 = TemperatureAnnealingCallback(
    Co_attention_layers,
    initial_temperature=final_temperature_stage1,
    final_temperature=0.1,
    total_epochs=50
)

history = models.fit(train_dataset,
         epochs = 50,callbacks = [checkpoint,earlystopping, temperature_annealing_callback_stage2],
         validation_data = val_dataset)