from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
	def __init__(self,dbPath,batchSize,preprocessors=None,aug=None,binarize=True,classes=2):
		self.db=h5py.File(dbPath)			 #path to HDF5 dataset that store images and class label ....opens file pointer to hdf5 file
		self.batchSize=batchSize 			 #size of mini-batches to yield wwhen training our network
		self.preprocessors=preprocessors			 #list of image preprocessor we are going to apply MeanPreprocessor ImagetoArrayPreprocessor 
		self.aug=aug					 #data augmentation to apply data augmentation directly inside our HDF5datasetgenerator 
		self.binarize=binarize #important****				 #store class labels as single integers innside HDF5 dataset if we use binary or categorical_roossentropy we need one hot encoded vector this switch indicates whteher or not binarization takes placce
		self.classes=classes				#no of unique classesit is required to construct our one-hot encoded vector during binarization phase
		self.numImages=self.db["labels"].shape[0] 	#used to access total nnumber of datapoints in dataset
		
	#it is responsible for yielding batches of images and labels to keras.fit_generator when training network
	def generator(self,passes=np.inf): #optional argument passes which can be considered as total no of epochs
		epochs=0
		
		while epochs<passes: # in  one iteraction of epoch we extract batch wise data  in images and labels .... this loop will run indefinitely until keras reaches termination criteria or ctrl+c
			for i in np.arange(0,self.numImages,self.batchSize):
				images=self.db["images"][i:i+self.batchSize]
				labels=self.db["labels"][i:i+self.batchSize]
				
				if self.binarize: #check to see if labels should be binarized
					labels=np_utils.to_categorical(labels,self.classes)
					
				if self.preprocessors is not None: #if preprocessors is not none then for each image peprocessor loop is travelled....there is chaining of preprocessor inside data generator
					procImages=[]
					
					for image in images:
						for p in self.preprocessors:
							image=p.preprocess(image)
							
						procImages.append(image)
						
					images=np.array(procImages)
				
				if self.aug is not None:
					(images,labels)=next(self.aug.flow(images,labels,batch_size=self.batchSize)) 	#output is 2-tuple of batch of images and labels 
					
				yield(images,labels)
			epochs+=1
			
	def close(self): #closing a pointer HDF5 dataset
		self.db.close()
