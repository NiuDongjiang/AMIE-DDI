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

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
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
tokenizer = Mol_Tokenizer('token_id.json')
map_dict = np.load('preprocessed_drug_info.npy',allow_pickle=True).item()

# %%
train_dataset_,validation_dataset, test_dataset_ = Graph_Bert_Dataset_fine_tune(tr_dataset,val_dataset,tst_dataset,label_field='DDI',tokenizer=tokenizer,map_dict=map_dict,batch_size = 128).get_data()
train_dataset = train_dataset_.map(input_solver1)
val_dataset_ = validation_dataset.map(input_solver1)
test_dataset = test_dataset_.map(input_solver1)

# %%
param = {'name': 'Small', 'num_layers': 4, 'num_heads': 8, 'd_model': 256}

# %%
arch = param
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

# %%
model_motif = Model(
    inputs=[atom_input1,atom_input2, motif_input1, motif_input2, atom_adj_input1, atom_adj_input2, atom_dist_input1, atom_dist_input2, atom_match_matrixs1, atom_match_matrixs2, sum_atom1, sum_atom2, motif_adj_input1, motif_adj_input2, motif_dist_input1, motif_dist_input2],
    outputs=[Outseq1,Outseq2, encoder_padding_mask_atom1, encoder_padding_mask_motif1, encoder_padding_mask_atom2, encoder_padding_mask_motif2]
)

# %%
motif_inputs1 = Input(shape=(None,), name= "molecule_sequence1")
motif_inputs2 = Input(shape=(None,), name= "molecule_sequence2")
motif_adj_inputs1 = Input(shape=(None,None), name= "adj_matrix1")
motif_adj_inputs2 = Input(shape=(None,None), name= "adj_matrix2")
motif_dist_inputs1 = Input(shape=(None,None), name= "dist_matrix1")
motif_dist_inputs2 = Input(shape=(None,None), name= "dist_matrix2")
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

druga_trans,drugb_trans,encoder_padding_mask_atom1, encoder_padding_mask_motif1, encoder_padding_mask_atom2, encoder_padding_mask_motif2 = model_motif([atom_inputs1,atom_inputs2, motif_inputs1, motif_inputs2, atom_adj_inputs1, atom_adj_inputs2, atom_dist_inputs1, atom_dist_inputs2, atom_match_matrix1, atom_match_matrix2, sum_atoms1, sum_atoms2, motif_adj_inputs1, motif_adj_inputs2, motif_dist_inputs1, motif_dist_inputs2])

Co_attention_layers = Co_Attention_Layer(d_model,k = 128,num_heads=8,temperature=0.1042,name = 'Co_attention_layer')
fc1 = tf.keras.layers.Dense(d_model/2, activation='relu')
dropout1 = tf.keras.layers.Dropout(dropout_rate)
fc2 = tf.keras.layers.Dense(d_model/4, activation='relu')
dropout2 = tf.keras.layers.Dropout(dropout_rate)
fc3 = tf.keras.layers.Dense(4,activation='softmax')
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

models = Model(inputs=[atom_inputs1,atom_adj_inputs1,atom_dist_inputs1\
    ,atom_match_matrix1,sum_atoms1,motif_inputs1,motif_adj_inputs1,motif_dist_inputs1,
    atom_inputs2,atom_adj_inputs2,atom_dist_inputs2\
    ,atom_match_matrix2,sum_atoms2,motif_inputs2,motif_adj_inputs2,motif_dist_inputs2],outputs =[output1_2])

models.load_weights('AMIE_DDI_time_.h5')
# %%
from sklearn import metrics
import numpy as np

def evaluation(preds, truths, average='micro', ndigits=4):
    # 预测类别
    pred_res_to_labels = np.argmax(preds, axis=1)
    label_to_onehot = np.eye(4)[truths]

    acc = round(metrics.accuracy_score(truths, pred_res_to_labels), 4)
    auc = round(metrics.roc_auc_score(label_to_onehot, preds, average=average), ndigits)
    aupr = round(metrics.average_precision_score(label_to_onehot, preds, average=average), ndigits)
    precision = round(metrics.precision_score(truths, pred_res_to_labels, average='macro'), ndigits)
    recall = round(metrics.recall_score(truths, pred_res_to_labels, average='macro'), ndigits)
    f1 = round(metrics.f1_score(truths, pred_res_to_labels, average='macro'), ndigits)

    print(f'ACC: {acc}, AUROC: {auc}, AUPR: {aupr}, '
          f'Precision: {precision}, Recall: {recall}, F1: {f1}')

# %%
pred_res = models.predict(val_dataset_,verbose=False)
labels = val_dataset['DDI'].tolist()

# %%
print('Evaluation results for unseen DDIs:')
evaluation(pred_res,labels)