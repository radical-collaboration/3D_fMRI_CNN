import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import numpy as np
np.random.seed(12)
import theano
import keras
from keras.models import Sequential,model_from_json
from keras.layers import TimeDistributed, Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2, activity_l2
import read_fmri_util as rf
from tqdm import tqdm
import pickle
import argparse
parser = argparse.ArgumentParser(description='Keras LSTM cross validation study')
parser.add_argument('savefolderpath', metavar='p', help='path to folder to save models')
args = parser.parse_args()
savefolder = args.savefolderpath
T = 137
def print_accuracy(model,data,label,batch_Size,timesteps,type_,seed=123):#data format (N,T,V)
	loss = 0
	accuracy = 0
	count = 0
	ratio_ = 0
	for x,_,l in rf.lstm_batch_gen_randomized(data,label,batch_Size,timesteps,seed):
		result = model.evaluate(x, l, batch_size=batch_Size, verbose=0,sample_weight=None)
		count += 1
		loss+=result[0]
		accuracy+=result[1]
		ratio_+=float(np.sum(l))/l.shape[0]
	ratio_/=count
	accuracy/=float(count)
	print ('\t'+type_+' : Loss(mse):%4.4f  Accuracy:%4.4f Schizophrenic/Total ratio:%4.4f  Total:%d'%(loss/float(count),accuracy,ratio_,count))
	return accuracy,loss/float(count)
def print_accuracy_avg(model,data,label,p_id,batch_Size,timesteps,type_,seed=123,avg_type='TIME'):#data format (N,T,V) 
	count = 0
	count_correct = 0
	ratio_ = 0
	if avg_type == 'TIME':
		generator = rf.averaged_predict_batch_gen_randomized(data,label,batch_Size,timesteps,seed,no_of_timepts_per_subject=20)
	elif avg_type == 'RUNS':
		generator = rf.averaged_predict_runs_batch_gen_randomized(data,label,p_id,batch_Size,timesteps,seed,no_of_timepts_per_subject=20)

	for x,_,l in generator:
		L = np.unique(l)
		assert (len(L.shape)==1)
		if L[0]==1:
			ratio_+=1
		result = model.evaluate(x, l, batch_size=batch_Size, verbose=0,sample_weight=None)
		count += 1
		if result[1]>0.5:
			count_correct+=1
	accuracy = count_correct/float(count)
	print ('\t'+type_+' : Averaged Accuracy:%4.4f Schizophrenic/Total ratio:%4.4f  Total:%d'%(accuracy,ratio_/float(count),count))
	return accuracy
def print_accuracy_all(model,data,label,p_id,batch_Size,timesteps,type_,seed=123):#data format (N,T,V) 
	count = 0
	count_correct_subsample = 0
	count_correct_sample = 0
	count_correct_subject = 0
	ratio_ = 0
	accuracy_per_subsample=0
	accuracy_per_sample=0
	accuracy_per_subject=0
	generator = rf.averaged_predict_runs_batch_gen_randomized(data,label,p_id,batch_Size,timesteps,seed,no_of_timepts_per_subject=T-timesteps)

	for x,_,l in generator:
		L = np.unique(l)
		assert (len(L.shape)==1)
		assert(x.shape[0]==4*(T-timesteps))
		if L[0]==1:
			ratio_+=1
		result = model.predict_on_batch(x)
		
		result = np.squeeze(result)
		is_correct = np.equal(np.asarray([1 if x>0.5 else 0 for x in result]).astype('int'),l.astype('int'))
		count_correct_subsample+=np.sum(is_correct.astype('float'))
		# count_correct_sample += np.sum(np.mean(np.reshape(is_correct.astype('float'),(4,is_correct.shape[0]/4)),axis=1)>0.5)
		# Z = 1 if np.mean(is_correct)>0.5 else 0
		# count_correct_subject+= Z

		count_correct_sample += np.sum(np.asarray(np.mean(np.reshape(result,(4,is_correct.shape[0]/4)),axis=1)>0.5).astype('int')==int(L[0]))
		Z = 1 if np.mean(result)>0.5 else 0
		count_correct_subject+= int(Z==L[0])
		count += 1
		
	ratio_ = ratio_/float(count)
	accuracy_per_subsample = count_correct_subsample/(float(count)*4*(T-timesteps))
	accuracy_per_sample = count_correct_sample/(float(count)*4)
	accuracy_per_subject = count_correct_subject/float(count)

	print ('\t'+type_+' : Acc_subsample:%4.4f  Acc_sample:%4.4f Acc_subject:%4.4f Schizophrenic/Total ratio:%4.4f No of subjects:%d'%(accuracy_per_subsample,accuracy_per_sample,accuracy_per_subject,ratio_,count))
	return accuracy_per_subsample,accuracy_per_sample,accuracy_per_subject	



#######################################################Data prep
centres_ =[3,6,9,10,18]
runs_ = [1,2,3,4]
if not os.path.isfile('./lstm_data.pickle'):
	print('Generating data....')
	A,L,p_id,run,centre = rf.load_fmri_2d_data_masked(centres=[3,6,9,10,18],runs=[1,2,3,4])#A format:(N,T,V)
	with open('lstm_data.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
		pickle.dump([A,L,p_id,run,centre], file_)
	print('Pickled lstm_data.pickle !')
else:
	print('Loading existing data....')
	with open('lstm_data.pickle') as fil:  # Python 3: open(..., 'rb')
		A,L,p_id,run,centre = pickle.load(fil) 
	print('Loaded lstm_data.pickle.')

print('Standardizing data....')
A = A - np.mean(A,axis=(1),keepdims=True)
A = rf.standardize(A,axes=(1,2),eps=1e-5)
#A = rf.standardize(A,axes=(0,1),eps=1e-5)
A = rf.standardize_centrewise(A,L,centre)
print('Now subsetting it for given runs and centres....')
ind_centre = [True if x in centres_ else False for x in centre]
ind_run =  [True if x in runs_ else False for x in run]
ind = np.logical_and(ind_centre,ind_run)
A = A[ind]
L = L[ind]
p_id = p_id[ind]
run = run[ind]
centre = centre[ind]
fold_count = 0
print('Running our random split for test and val dataset separation....Care taken to ensure that multiple runs of single subject lie only on one side of split')
for data_train,label_train,pid_train,data_val,label_val,pid_val in rf.rand_train_val_split_cvgen(A,L,p_id,ratio=0.8,seed=142,crossvalsplits=10):
	
	#######################################################Model prep
	V=569

	data_dim = V
	timesteps = 64
	nb_classes = 1
	nb_epoch = 100
	batch_size = 64
	model = None
	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	#model.add(TimeDistributed(Dense(40,activation='relu',W_regularizer=l1(0.05)),input_shape=(timesteps, data_dim)))
	model.add(LSTM(32, return_sequences=True,input_shape=(timesteps,data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(Dropout(0.3))
	model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32
	# model.add(Dropout(0.3))
	# model.add(LSTM(32, return_sequences=False))  # returns a sequence of vectors of dimension 32

	# # model.add(LSTM(32))  # return a single vector of dimension 32
	model.add(Dropout(0.3))
	model.add(Dense(1, W_regularizer=l1(0.01),activation='sigmoid'))
	# model.add(Dropout(0.2))
	# model.add(Dense(1))
	# model.add(Activation('sigmoid'))
	adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9, nesterov=True)
	model.compile(loss='mse',optimizer=sgd,metrics=['accuracy'])


	print(model.summary())
	print('Number of parameters:%s'%model.count_params())
	




	print(data_train.shape,data_val.shape)
	if not os.path.isdir(savefolder):
		os.makedirs(savefolder)
	
	#######################################################Training and Validation
	stop_train = False
	breakpt = 80
	acc1 = []
	acc2 = []
	acc3 = []
	acc4 = []
	acc5 = []
	acc6 = []

	acc_max = -1.0
	is_halved = False
	for e in range(nb_epoch):
		if stop_train:
			break
		print("EPOCH : %d" % e)
		count = 0
		for x,_,l in rf.lstm_batch_gen_randomized(data_train,label_train,batch_size,timesteps,seed=123):
			if stop_train:
				break
			result = model.train_on_batch(x, l, class_weight=None,sample_weight=None)
			if(count%1==0 and count>25) or count==0:
				lrr = model.optimizer.lr.get_value()
				print('Fold no :%d Batch_%d   LR : %4.4f'%(fold_count+1,count+1,lrr))
				acc_train_subsample,acc_train_sample,acc_train_subject = print_accuracy_all(model,data_train,label_train,pid_train,batch_size,timesteps,'TRAIN',seed=123)
				acc_val_subsample,acc_val_sample,acc_val_subject = print_accuracy_all(model,data_val,label_val,pid_val,batch_size,timesteps,'VALIDATION',seed=123)
				acc1.append(acc_train_subsample)
				acc2.append(acc_train_sample)
				acc3.append(acc_train_subject)
				acc4.append(acc_val_subsample)
				acc5.append(acc_val_sample)
				acc6.append(acc_val_subject)
				# a1,l1 = print_accuracy(model,data_train,label_train,batch_size,timesteps,'TRAIN',seed=123)
				# a2 = print_accuracy_avg(model,data_train,label_train,pid_train,batch_size,timesteps,'TRAIN',seed=123,avg_type='RUNS')
				# a3,l2 = print_accuracy(model,data_val,label_val,batch_size,timesteps,'VALIDATION',seed=123)
				# a4 = print_accuracy_avg(model,data_val,label_val,pid_val,batch_size,timesteps,'VALIDATION',seed=123,avg_type='RUNS')
				# a5 = print_accuracy_avg(model,data_train,label_train,pid_train,batch_size,timesteps,'TRAIN',seed=123,avg_type='TIME')
				# a6 = print_accuracy_avg(model,data_val,label_val,pid_val,batch_size,timesteps,'VALIDATION',seed=123,avg_type='TIME')
				# acc1.append(a1)
				# acc2.append(a2)
				# acc3.append(a3)
				# acc4.append(a4)
				# acc5.append(a5)
				# acc6.append(a6)
			if acc_max<acc_val_sample:
				model_json = model.to_json()
				with open(savefolder+'/model_'+str(fold_count+1)+'.json', 'w') as json_file:
					json_file.write(model_json)
				# serialize weights to HDF5
				model.save_weights(savefolder+'/model_'+str(fold_count+1)+'.h5',overwrite=True)
				print("Saved model to disk")
				acc_max = acc_val_sample
			if count>30 and acc_val_subsample>0.60 and (not is_halved):
				model.optimizer.lr.set_value(lrr/2.0)
				is_halved = True
			if count==breakpt:
				break
			count += 1
		if count==breakpt:
			break	
	with open(savefolder+'/cross_val_fold_'+str(fold_count+1)+'_accuracy_plot.pkl','w') as f:
		pickle.dump([acc1,acc2,acc3,acc4,acc5,acc6,None],f)
	fold_count +=1
	

