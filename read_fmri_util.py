import os
import numpy as np
import scipy.io
import scipy.signal
import scipy.cluster.vq
import pickle
from sklearn.decomposition import PCA
import itertools
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing,cross_validation
from keras.preprocessing import sequence
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
#N = 380#total no of samples acros all centres,runs

home_ = os.getenv("HOME")
#home_ = os.path.join(home_,'SIDDHARTH_MUTHUKUMAR_intern')
sampletype_ = 'subsampled/'# either 'full/' or 'subsampled/'
masktype_ = 'masked/'# either 'unmasked/' or 'masked/' each datapoint obeys::if #MASKED DATA shape:(V,T), if #UNMASKED : (X,Y,Z,T)
if sampletype_ is 'full/':
	maskfile = 'Mask_nz_AO_denoised_MNImasked_NN.mat'
	V = 26949
	shape_3d_unmasked = [53,64,37]
	T = 137 
else:
	maskfile = 'Mask_nz_AO_denoised_13_16_12_MNImasked_NN.mat'
	V = 569
	shape_3d_unmasked = [13,16,12]
	T = 137
if home_ in '/home/siddhu/':
	default_pathx = os.path.join(home_,'/FBIRN/original_res/' , sampletype_, masktype_)
	default_pathy = os.path.join(home_,'/FBIRN/original_res/', sampletype_)
elif home_ in '/home/siddhu95':
	default_pathx = os.path.join(home_ ,sampletype_, masktype_)
	default_pathy = os.path.join(home_ ,sampletype_)
elif home_ in '/home/sid95':
	default_pathx = os.path.join(home_,sampletype_,masktype_)
	default_pathy = os.path.join(home_,sampletype_)
maskfilepath = os.path.join(default_pathy,maskfile)
def demaskify_1d_to_3d(data_1d,maskfilepath_=maskfilepath):#data_1d expected format : (N,V) eg. rbm's principal spatial maps 
	mask_3d = scipy.io.loadmat(maskfilepath_)
	mask_1d = np.reshape(mask_3d.astype('bool'),(-1),order='F')

	data_3d = []
	for i in range(data_1d.shape[0]):
		D = np.zeros(mask_1d.shape)
		D[mask_1d] = data_1d[i]
		D = np.reshape(D,mask_3d.shape,order='F') 
		data_3d.append(D)
	return np.asarray(data_3d)

def parse_id(name):
	name = os.path.splitext(os.path.basename(name))[0]
	return str(name.split('_')[0])
def read_fmri_2d_data_masked(pathx=default_pathx,pathy=default_pathy,centres=[3,6,9,10,18]):
	#returned data format : (N,T,V)
	centre_map = {'0003':3,'0006':6,'0009':9,'0010':10,'0018':18}

	# label = scipy.io.loadmat(os.path.join(pathy,'class_label.mat'))
	# label = label['class_label']
	with open(os.path.join(home_,'ADHD_bhaskar_data','SubjectID2label_map.pkl')) as file_:
		SubjectID2label_map = pickle.load(file_)

	#label = np.array((1+label[0])/2,dtype='int')
	
	p_id = []
	l = []
	onlymatfiles = [f for f in os.listdir(pathx) if (os.path.isfile(os.path.join(pathx, f)) and f.endswith('.mat'))]
	onlymatfiles.sort()

	A = []
	run = []
	centre = []
	for i,g in enumerate(onlymatfiles):
		if(centre_map[g[0:4]] in centres):
			D = scipy.io.loadmat(os.path.join(pathx,g))
			#A = np.asarray(A,np.reshape(D['ROI_time_series'],(1,V,T),order='F'),axis=0)
			gu = g.strip().split('_')
			run_no = int(gu[3].split('.')[0])
			A.append(np.transpose(D['t']))#appending in format:(T,V)
			p_id.append(parse_id(g))
			l.append(SubjectID2label_map[parse_id(g)])
			run.append(run_no)
			centre.append(centre_map[g[0:4]])
	A = np.asarray(A)#format now becomes:(N,T,V)
	l = np.asarray(l)
	run = np.asarray(run)
	p_id = np.asarray(p_id)
	centre = np.asarray(centre)
	return (A,l,p_id,run,centre)

def load_fmri_2d_data_masked(pathx=default_pathx,pathy=default_pathy,centres=[3,18],runs=[1,2]):
	#returned data format : (N,T,V)
	if not os.path.isfile('fmri_2d_data_masked.pickle'):
		A,l,p_id,run,centre = read_fmri_2d_data_masked(pathx,pathy,centres=[3,6,9,10,18])
		with open('fmri_2d_data_masked.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
			pickle.dump([A,l,p_id,run,centre], file_)
		print('Pickled fmri_2d_data_masked!')
	else:
		with open('fmri_2d_data_masked.pickle') as fil:  # Python 3: open(..., 'rb')
			A,l,p_id,run,centre = pickle.load(fil) 
		print('Loaded fmri_2d_data_masked. Now subsetting it for given runs and centres....')
	ind_centre = [True if x in centres else False for x in centre]
	ind_run =  [True if x in runs else False for x in run]
	ind = np.logical_and(ind_centre,ind_run)
	A = A[ind]
	l = l[ind]
	p_id = p_id[ind]
	run = run[ind]
	centre = centre[ind]
	return (A,l,p_id,run,centre)

def lstm_iterator(data,label,num_steps=32,stride=5):
	#data format expected : (N,T,V)
	for j in range(data.shape[0]):
		for i in range(T-num_steps):
			x = data[j,i:i+num_steps,:]
			y = data[j,(i+1):(i+1+num_steps),:]
			yield(x,y,label[j])

def lstm_datagen(data,label,num_steps=32,shuffle=False):
	X = []
	Y = []
	L = []
	count = 0
	for (x,y,l) in lstm_iterator(data,label,num_steps):
		X.append(x)
		Y.append(y)
		L.append(l)
		count = count+1
	X = np.asarray(X)
	Y = np.asarray(Y)
	L = np.asarray(L)
	if shuffle:
		rand_ind = np.random.permutation(X.shape[0])
		X = X[rand_ind]
		Y = Y[rand_ind]
		L = L[rand_ind]
	print ('Dataset size:%d'%(count))
	return X,Y,L

def lstm_batch_gen_randomized(data,label,batch_size=32,num_steps=32,seed=123,T=137):
	#data format expected : (N,T,V)
	np.random.seed(seed)
	rand_ind_subject = np.random.permutation(data.shape[0])
	rand_ind_timept = np.random.permutation(T-num_steps)
	X=[]
	Y=[]
	L=[]
	# print(rand_ind_subject.shape[0])
	# print(rand_ind_timept.shape[0])
	# raw_input()
	batch_count = 0
	for i in rand_ind_timept:
		for j in rand_ind_subject:

			x = data[j,i:i+num_steps,:]
			y = data[j,(i+1):(i+1+num_steps),:]
			X.append(x)
			Y.append(y)
			L.append(label[j])
			batch_count += 1
			if batch_count==batch_size:
				batch_count = 0
				# X_ = np.asarray(X)
				# Y_ = np.asarray(Y)
				# L_ = np.asarray(L)
				# X = []
				# Y = []
				# Z = []
				# yield(X_,Y_,L_)
				yield(np.asarray(X),np.asarray(Y),np.asarray(L))
				X = []
				Y = []
				L = []
def averaged_predict_batch_gen_randomized(data,label,batch_size=32,num_steps=32,seed=123,no_of_timepts_per_subject=20,T=137):
	#data format expected : (N,T,V)
	np.random.seed(seed)
	no_of_timepts_per_subject = min(no_of_timepts_per_subject,T-num_steps)
	
	for j in range(data.shape[0]):
		X=[]
		Y=[]
		L=[]
		for i in np.random.choice(T-num_steps,no_of_timepts_per_subject):
			x = data[j,i:i+num_steps,:]
			y = data[j,(i+1):(i+1+num_steps),:]
			X.append(x)
			Y.append(y)
			L.append(label[j])	
		yield(np.asarray(X),np.asarray(Y),np.asarray(L))
def averaged_predict_runs_batch_gen_randomized(data,label,p_id,batch_Size,num_steps=32,seed=123,no_of_timepts_per_subject=20,T=137):
	#data format expected : (N,T,V)
	np.random.seed(seed)
	map_id2index = {}

	for i,id_ in enumerate(list(p_id)):
		if id_ in map_id2index:
			map_id2index[id_].append(i)
		else:
			map_id2index[id_] = [i]

	uniq_ids = map_id2index.keys()
	no_of_timepts_per_subject = min(no_of_timepts_per_subject,T-num_steps)
	
	for PID in map_id2index.keys():
		X=[]
		Y=[]
		L=[]
		for j in map_id2index[PID]: 
			for i in np.random.choice(T-num_steps,no_of_timepts_per_subject):
				x = data[j,i:i+num_steps,:]
				y = data[j,(i+1):(i+1+num_steps),:]
				X.append(x)
				Y.append(y)
				L.append(label[j])	
		assert (len(np.unique(np.asarray(L)).shape)==1)
		yield(np.asarray(X),np.asarray(Y),np.asarray(L))				
def stateful_batch_gen(data,label,num_steps=32,seed=123):
	no_of_batch_per_run = T//num_steps
	for i in range(no_of_batch_per_run):
		X = []
		L = []
		for j in range(data.shape[0]):
			X.append(data[j,i*num_steps:(i+1)*num_steps,:])
			L.append(label[j])
		print(np.asarray(X).shape[0])
		yield(np.asarray(X),np.asarray(L))
	X = []
	L = []
	for j in range(data.shape[0]):
		X.append(sequence.pad_sequences(data[j,(no_of_batch_per_run*num_steps-T):,:], maxlen=num_steps))
		L.append(label[j])
	yield (np.asarray(X),np.asarray(L))

def rand_train_val_split(A,l,p_id,ratio=0.85,seed=123):
	np.random.seed(seed)
	map_id2index = {}
	map_id2label = {}
	count = 0
	for i,id_ in enumerate(list(p_id)):
		if id_ in map_id2index:
			if id_ in map_id2label:
				count+=1
			map_id2index[id_].append(i)
		else:
			map_id2label[id_] = l[i]
			map_id2index[id_] = [i]
	print('Error count:%d'%count)
	uniq_ids = map_id2index.keys()
	corresponding_labels = [map_id2label[x] for x in uniq_ids]
	# splitpt = int(ratio*len(uniq_ids))
	# rand_ind = np.random.permutation(len(uniq_ids))
	# uniq_ids = np.asarray(uniq_ids)

	sss = StratifiedShuffleSplit(corresponding_labels, 1, test_size=1.0-ratio, random_state=seed)
	uniq_ids = np.asarray(uniq_ids)
	
	for train_index, val_index in sss:
		train_ids = uniq_ids[train_index]
		val_ids = uniq_ids[val_index]

	train_ind = []
	val_ind = []

	for id_ in train_ids:
		train_ind.extend(map_id2index[id_])
	for id_ in val_ids:
		val_ind.extend(map_id2index[id_])

	return (A[train_ind],l[train_ind],p_id[train_ind],A[val_ind],l[val_ind],p_id[val_ind])

def rand_train_val_split_cvgen(A,l,p_id,ratio=0.8,seed=123,crossvalsplits=1,leaveoneout=False):
	np.random.seed(seed)
	map_id2index = {}
	map_id2label = {}
	count = 0
	for i,id_ in enumerate(list(p_id)):
		if id_ in map_id2index:
			if id_ in map_id2label:
				count+=1
			map_id2index[id_].append(i)
		else:
			map_id2label[id_] = l[i]
			map_id2index[id_] = [i]
	print('Error count:%d'%count)
	uniq_ids = map_id2index.keys()
	corresponding_labels = [map_id2label[x] for x in uniq_ids]
	# splitpt = int(ratio*len(uniq_ids))
	# rand_ind = np.random.permutation(len(uniq_ids))
	# uniq_ids = np.asarray(uniq_ids)

	#sss = StratifiedShuffleSplit(corresponding_labels, crossvalsplits, test_size=1.0-ratio, random_state=seed)
	sss = cross_validation.ShuffleSplit(len(corresponding_labels), n_iter=crossvalsplits,test_size=1.0-ratio, random_state=seed)
	if leaveoneout:
		sss = LeaveOneOut(len(corresponding_labels))
	uniq_ids = np.asarray(uniq_ids)

	for train_index, val_index in sss:
		train_ids = uniq_ids[train_index]
		val_ids = uniq_ids[val_index]

		train_ind = []
		val_ind = []

		for id_ in train_ids:
			train_ind.extend(map_id2index[id_])
		for id_ in val_ids:
			val_ind.extend(map_id2index[id_])
		yield (A[train_ind],l[train_ind],p_id[train_ind],A[val_ind],l[val_ind],p_id[val_ind])
def rand_train_val_split_cv_leave_centre_out_gen(A,l,p_id,centre):#A format : (N,T,V)
	for c in [3,6,9,10,18]:
		C = [x for x in [3,6,9,10,18] if x!=c]
		val_ind = centre==c
		train_ind = np.logical_not(val_ind)
		
		if not (len(train_ind)==0 or len(val_ind)==0):
			print('Train Val Split created ====>  Train centres:',C,'         Val centres :',[c])
			yield (A[train_ind],l[train_ind],p_id[train_ind],A[val_ind],l[val_ind],p_id[val_ind])

def get_fmri_4d_spectrum_data(data=None,nobins=16):#converts fmri 4d data to 3d volumes with as many channels as 
#there are bins in thebinned frequency spectrum per voxel
#assume data in in shape : (N,X,Y,Z,T) ie a list of fmri data each one being  a 4d tensor
	data = standardize(data,axes=(4))
	#print(data.shape)
	freq_scale,data = scipy.signal.periodogram(data,fs=0.5,nfft=256, detrend='constant',\
return_onesided=True, scaling='spectrum', axis=-1)
	#returns one-sided fft of length 129 = 256/2 + 1
	print(data.shape)
	data_binned_spectrum = []
	#window=scipy.signal.get_window('blackman',9)
	size_ = (data.shape[1],data.shape[2],data.shape[3],nobins)
	fs = 0.5

	print('Creating fmri_4d_spectrum_data...')
	
	for i in range(data.shape[0]):
		print i
		D = data[i]
		Df = np.empty(size_)
		print Df.shape
		for x, y, z in itertools.product(*map(xrange, (D.shape[0], D.shape[1], D.shape[2]))):			
			#freq_scale = np.linspace(0.0, fs/2, num=128)
			Df[x,y,z],_,_ = scipy.stats.binned_statistic(freq_scale,D[x,y,z],statistic='sum',bins=nobins)
		data_binned_spectrum.append(Df)
	return np.asarray(data_binned_spectrum)

def load_fmri_4d_spectrum_data(pathx=default_pathx,pathy=default_pathy,centres=[3,18],runs=[1,2,3,4],no_bins=16):
	filename = 'fmri_4d_spectrum_data_'+str(no_bins)
	if not os.path.isfile(filename+'.pickle'):
		A,l,p_id,run,centre = read_fmri_4d_data_unmasked(pathx,pathy,[3,6,9,10,18])
		A = get_fmri_4d_spectrum_data(A,no_bins)
		print(A.shape)
		print('Processing of ' + filename + ' complete!Now we will pickle it for further use...')
		with open(filename+'.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
			pickle.dump([A,l,p_id,run,centre], file_)
		print('Pickled '+filename+' !')
	else:
		with open(filename+'.pickle') as fil:  # Python 3: open(..., 'rb')
			A,l,p_id,run,centre = pickle.load(fil) 
		print('Loaded '+filename+'. Now subsetting it for given runs and centres....')
	ind_centre = [True if x in centres else False for x in centre]
	ind_run =  [True if x in runs else False for x in run]
	ind = np.logical_and(ind_centre,ind_run)
	A = A[ind]
	l = l[ind]
	p_id = p_id[ind]
	run = run[ind]
	centre = centre[ind]
	return (A,l,p_id,run,centre)
def mask_normalize(data,mask,eps=1e-5):
	#dat format : (N,X,Y,Z,T)
	n=int(np.sum(mask.flatten()))
	mean = np.sum(data,axis=(1,2,3,4),keepdims=True)/float(n*137)
	data = data-mean
	sd = np.sqrt(np.sum(np.multiply(data,data),axis=(1,2,3,4),keepdims=True)/float(n*137))
	return np.divide(data,sd+eps)
def repelem(a,num=T):
	G = []
	for i in list(a):
		G.extend([i]*num)
	return np.asarray(G)		
def load_fmri_4d_data_as_volumes(pathx=default_pathx,pathy=default_pathy,centres=[3,18],runs=[1,2]):
	if not os.path.isfile('fmri_4d_data_norm.pickle'):
		A,l,p_id,run,centre = read_fmri_4d_data_unmasked(pathx,pathy,centres=[3,6,9,10,18])
		mask_ = scipy.io.loadmat(maskfilepath)
		mask_ = mask_['Mask_universal']
		print('Reshaping mask into a volume....')
		mask_ = np.expand_dims(np.reshape(mask_,A[0].shape,order='F').astype('float'),0)
		print('Multiplying mask_ with data....')
		A = np.multiply(A,mask_)
		print('Normalizing data....')
		A = mask_normalize(A,mask_)
		d1,d2,d3,d4,d5 = A.shape#Here A is in format (N,X,Y,Z,T)
		A = np.reshape(np.moveaxis(A,-1,1),(d1*d5,d2,d3,d4))
		#Now we have a batch of volumes with all volumes of a single run, together in the batch
		l = repelem(l,T)
		p_id = repelem(p_id,T)
		run = repelem(run,T)
		centre = repelem(centre,T)
		assert(A.shape[0]==len(list(l)))
		with open('fmri_4d_data_norm.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
			pickle.dump([A,l,p_id,run,centre], file_)
		print('Pickled fmri_4d_data_norm!')
	else:
		with open('fmri_4d_data_norm.pickle') as fil:  # Python 3: open(..., 'rb')
			A,l,p_id,run,centre = pickle.load(fil) 
		print('Loaded fmri_4d_data_norm.')
	print('Now subsetting it for given runs and centres....')
	ind_centre = [True if x in centres else False for x in centre]
	ind_run =  [True if x in runs else False for x in run]
	ind = np.logical_and(ind_centre,ind_run)
	A = A[ind]
	l = l[ind]
	p_id = p_id[ind]
	run = run[ind]
	centre = centre[ind]		
	return (A,l,p_id,run,centre)

# def read_fmri_4d_data_as_unmasked_volumes(pathx=default_pathx,pathy=default_pathy,centres=[3,6,9,10,18]):#returns data in shape(N*T,X,Y,Z)
# 	# np.random.seed(1)
# 	centre_map = {'0003':3,'0006':6,'0009':9,'0010':10,'0018':18}

# 	label = scipy.io.loadmat(os.path.join(pathy,'class_label.mat'))
# 	label = label['class_label']
# 	label = np.array((1+label[0])/2,dtype='int')
	
# 	p_id = []
# 	l = []
# 	onlymatfiles = [f for f in os.listdir(pathx) if (os.path.isfile(os.path.join(pathx, f)) and f.endswith('.mat'))]
# 	onlymatfiles.sort()

# 	A = []
# 	run = []
# 	centre = []
# 	for i,g in enumerate(onlymatfiles):
# 		if(centre_map[g[0:4]] in centres):
# 			D = scipy.io.loadmat(os.path.join(pathx,g))
# 			#A = np.asarray(A,np.reshape(D['ROI_time_series'],(1,V,T),order='F'),axis=0)
# 			gu = g.strip().split('_')
# 			run_no = int(gu[3].split('.')[0])
# 			A.extend(np.split(D['img'],D['img'].shape[3],axis=3))#3 is the time axis D[img].shape = (X,Y,Z,T)
# 			p_id.append(parse_id(g))
# 			l.append(label[i])
# 			run.append(run_no)
# 			centre.append(centre_map[g[0:4]])
# 	A = np.asarray(A)
# 	l = np.asarray(l)
# 	run = np.asarray(run)
# 	p_id = np.asarray(p_id)
# 	centre = np.asarray(centre)
# 	return (A,l,p_id,run,centre)
# def load_fmri_4d_data_as_unmasked_volumes(pathx=default_pathx,pathy=default_pathy,centres=[3,18],runs=[1,2]):
# 	if not os.path.isfile('fmri_4d_data_as_volumes.pickle'):
# 		A,l,p_id,run,centre = read_fmri_4d_data_as_unmasked_volumes(pathx,pathy,centres=[3,6,9,10,18])
# 		with open('fmri_4d_data_as_volumes.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
# 			pickle.dump([A,l,p_id,run,centre], file_)
# 		print('Pickled fmri_4d_data_as_volumes.pickle!')
# 	else:
# 		with open('fmri_4d_data_as_volumes.pickle') as fil:  # Python 3: open(..., 'rb')
# 			A,l,p_id,run,centre = pickle.load(fil) 
# 		print('Loaded fmri_4d_data_as_volumes.pickle. Now subsetting it for given runs and centres....')
# 	ind_centre = [True if x in centres else False for x in centre]
# 	ind_run =  [True if x in runs else False for x in run]
# 	ind = np.logical_and(ind_centre,ind_run)
# 	A = A[ind]
# 	l = l[ind]
# 	p_id = p_id[ind]
# 	run = run[ind]
# 	centre = centre[ind]
# 	return (A,l,p_id,run,centre)
def read_data(pathx,pathy,centres,expand=True):#returns data in shape(N,T,V)
	# np.random.seed(1)
	centre_map = {'0003':3,'0006':6,'0009':9,'0010':10,'0018':18}

	label = scipy.io.loadmat(os.path.join(pathy,'class_label.mat'))
	label = label['class_label']
	label = np.array((1+label[0])/2,dtype='int')
	
	p_id = []
	l = []
	onlymatfiles = [f for f in os.listdir(pathx) if (os.path.isfile(os.path.join(pathx, f)) and f.endswith('.mat'))]
	onlymatfiles.sort()

	A = []
	for i,g in enumerate(onlymatfiles):
		if(centre_map[g[0:4]] in centres):
			D = scipy.io.loadmat(os.path.join(pathx,g))
			#A = np.asarray(A,np.reshape(D['ROI_time_series'],(1,V,T),order='F'),axis=0)
			A.append(np.transpose(D['t']))
			p_id.append(parse_id(g))
			l.append(label[i])
	A = np.asarray(A)
	if expand:    
		A = np.expand_dims(A,3)
	return (A,l,p_id)



def demean(data,axes=(2),return_ = False,params=None):
	if params is not None:
		mean = params
	else:
		mean = np.mean(data,axis=axes,keepdims=1)
	data = data - mean
	print('Data Demeaning Done!')
	if return_:
		return data,mean
	else:
		return data


def standardize(data,axes=(1),eps=1e-4,return_ = False,params=None):
	if params is not None:
		mean = params[0]
		sigma = params[1]
	else:
		mean = np.mean(data,axis=axes,keepdims=1)
		sigma = np.std(data,axis=axes,keepdims=1)
	data = np.divide(data-mean,sigma+eps)
	print('Data Standardizing Done!')
	if return_:
		return data,mean,sigma
	else:
		return data
def standardize_centrewise(data,label,centre,return_ = False,params=None):
	d = [data[centre==x] for x in [3,6,9,10,18]]
	l = [label[centre==x] for x in [3,6,9,10,18]]
	d_list = []
	if params is not None:
		mean_list = params[0]
		sigma_list = params[1]
		for i,d_elem in enumerate(d):
			d_elem = standardize(d_elem,axes=(0,1),eps=1e-6,params=(mean_list[i],sigma_list[i]))
			d_list.append(d_elem)
	else:
		mean_list = []
		sigma_list = []
		for d_elem in d: #d_elem format (N,T,V)
			d_elem,mean_elem,sigma_elem = standardize(d_elem,axes=(0,1),eps=1e-6,return_= True)
			mean_list.append(mean_elem)
			sigma_list.append(sigma_elem)
			d_list.append(d_elem)
	data = np.concatenate(d_list,axis=0)
	if return_:
		return data,(mean_list,sigma_list)
	else:
		return data
def normalize_centrewise(data,label,centre,return_ = False,params=None):
	d = [data[centre==x] for x in [3,6,9,10,18]]
	l = [label[centre==x] for x in [3,6,9,10,18]]
	d_list = []
	if params is not None:
		mean_list = params[0]
		sigma_list = params[1]
		for i,d_elem in enumerate(d):
			d_elem = normalize_using_std(d_elem,axes=(0,1),eps=1e-6,params=(mean_list[i],sigma_list[i]))
			d_list.append(d_elem)
	else:
		mean_list = []
		sigma_list = []
		for d_elem in d: #d_elem format (N,T,V)
			d_elem,mean_elem,sigma_elem = normalize_using_std(d_elem,axes=(0,1),eps=1e-6,return_= True)
			mean_list.append(mean_elem)
			sigma_list.append(sigma_elem)
			d_list.append(d_elem)
	data = np.concatenate(d_list,axis=0)
	if return_:
		return data,(mean_list,sigma_list)
	else:
		return data
def normalize(data,axes=(1),eps=1e-5,return_ = False,params=None):
	if params is not None:
		min_ = params[0]
		max_ = params[1]
	else:
		min_ = np.array(np.min(data,axis=axes,keepdims=1),dtype='float')
		max_ = np.array(np.max(data,axis=axes,keepdims=1),dtype='float')
	data = np.divide(data-min_,max_-min_+eps)
	print('Data Normalizing Done!')
	if return_:
		return data,min_,max_
	else:
		return data

def normalize_using_std(data,axes=(1),eps=1e-5,return_ = False,params=None):
	if params is not None:
		mean = params[0]
		sigma = params[1]
	else:
		mean = np.mean(data,axis=axes,keepdims=1)
		sigma = np.std(data,axis=axes,keepdims=1)
	data = np.divide(data-mean,sigma+eps)
	data = (data + 1)*0.4 + 0.1
	print('Data Standardizing Done!')
	if return_:
		return data,mean,sigma
	else:
		return data

def process(data,norm=True):
	mean = np.array(np.mean(data,axis=(2),keepdims=1),dtype='float32')
	data = np.divide(data-mean,mean+1e-4)
	if norm:
		return normalize(data)
	else:
		return data

def whiten_(data):
	d = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
	d = np.reshape(scipy.cluster.vq.whiten(d),(data.shape[0],data.shape[1],data.shape[2]))
	print('Data Whitening Done!')
	return d

def random_split(data,label,ratio):
	np.random.seed(12)
	data = np.array(data)
	N = data.shape[0]//4
	no_train = max(1,int(ratio*N))
	ind = np.multiply(np.random.permutation(N),4)
	train_ind = ind[:no_train]
	test_ind = ind[no_train:]


	train_ind = np.array(np.hstack([train_ind,np.add(train_ind,1),np.add(train_ind,2),np.add(train_ind,3)]),dtype='int')
	test_ind = np.array(np.hstack([test_ind,np.add(test_ind,1),np.add(test_ind,2),np.add(test_ind,3)]),dtype='int')
	
	X_train = data[train_ind]
	X_test = data[test_ind]

	label = np.array(label)
	Y_train = label[train_ind]
	Y_test = label[test_ind]
	return (X_train,X_test,Y_train,Y_test)

def random_split2(data,label,ratio,num,seed_):
	np.random.seed(seed_)
	data = np.array(data)
	N = data.shape[0]//4
	no_train = max(1,int(ratio*N))
	ind = np.multiply(np.random.permutation(N),4)
	train_ind_base = ind[:no_train]
	test_ind_base = ind[no_train:]
	train_ind = []
	test_ind =[]

	for i in num:
		train_ind = np.array(np.hstack([train_ind,np.add(train_ind_base,i)]),dtype='int')
		test_ind = np.array(np.hstack([test_ind,np.add(test_ind_base,i)]),dtype='int')
	
	X_train = data[train_ind]
	X_test = data[test_ind]

	label = np.array(label)
	Y_train = label[train_ind]
	Y_test = label[test_ind]
	return (X_train,X_test,Y_train,Y_test)

def PCA_comps(data):# data format (N,T,V)
	# D = np.moveaxis(data, -1, 0)
	# print(D.shape)
	# D = np.reshape(D,(data.shape[0]*T,V),order='F')
	D = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
	pca = PCA(n_components=0.95)
	pca.fit(D)
	print('PCA : ',pca.n_components)
	C = np.transpose(pca.components_)
	#column vectors of C are the principal comps
	return np.reshape(C,(V,1,1,C.shape[1]),order='F')
	
def median_filter(data):
	return scipy.signal.medfilt(data,[1,1,3])


def wiener_filter(data):
	return scipy.signal.wiener(data ,mysize=[1,1,3], noise=None)
	#return scipy.signal.medfilt(data,[1,1,3])

def spectrumify(data,load=False):
	#converts time series of activations to time series of their ffts ie sprectrogram
	if load:
		with open('spectrumify_data.pickle') as fil:  # Python 3: open(..., 'rb')
			f,t,c = pickle.load(fil) 
		print('Loaded spectrumify data! Shape of Data: (no of freq bins in last axis as no_channels)',c.shape)
	else:
		f,t,c = scipy.signal.spectrogram(data,nperseg=50,noverlap=49,scaling='spectrum')
		print('Calculated spectrumify data!',c.shape)
		c = np.moveaxis(c, -2, -1)
		with open('spectrumify_data.pickle', 'w') as file_:  # Python 3: open(..., 'wb')
			pickle.dump([f, t, c], file_)
		print('Pickled Spectrumify data!')
	return c



def biased_split(data,label,ratio):
	#just for sanity checking if this improves accuracy herein different of runs of sam patient can be split across test 
	#and train sets, hence we cheat and it will show a false accuracy.This is to sanity check your model
	N = data.shape[0]
	no_train = max(1,int(ratio*N))
	ind=np.random.permutation(N)
	train_ind = ind[:no_train]
	test_ind = ind[no_train:]

	X_train = data[train_ind]
	X_test = data[test_ind]

	label = np.array(label)
	Y_train = label[train_ind]
	Y_test = label[test_ind]
	return (X_train,X_test,Y_train,Y_test)
	
