import h5py
import os

#it just the class store the data in hdf5 format

class HDF5DatasetWriter:
	def __init__(self,dims,outputPath,dataKey="images",bufSize=1000): #dataKey="images" when we are storing raw pixel values and ="features" when  we are storing exracted feeature from VGG16
		if os.path.exists(outputPath):
			raise ValueError("the supplied output path already exists and cannot be overwritten. Manually delete file before continuing",outputPath) #check whether output path exists
			
		#dims=dimension or shape of data that we are storing in dataset. (N,25088) 25088=512x7x7 is feature for each image after finaal pooling layer of VGG16 acrhitecture
		#outputPath : whereour hdf5 file will be stored
		#bufsize: coontrols the in memory buffer which we default to 1000 feature vectors/image
		self.db=h5py.File(outputPath,"w") #open hdf5 for writing
		
		#create dataset with dataKey name and supplied dims where our raw images/extraccted features will be stored will be stored
		self.data=self.db.create_dataset(dataKey,dims,dtype="float")# we can consider it as memory allocation
		#create a second dataset, this one to store the (integer) class labels for each record in the dataset
		self.labels=self.db.create_dataset("labels",(dims[0],),dtype="int")
		
		#iitialize our buffer
		self.bufSize=bufSize
		self.buffer={"data":[],"labels":[]}
		self.idx=0
		
		
	def add(self,rows,labels): 
		#add rows and their corresponding 	labels
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)
			
		#if buffer is full then fsh it to disk
		if len(self.buffer["data"])>=self.bufSize:
			self.flush()
				
	def flush(self):
		i=self.idx + len(self.buffer["data"]) #keeps the track ie index ie i in for loop
		self.data[self.idx:i]=self.buffer["data"] #stores the index in loop in pytn terms slicing is done to store data and labels
		self.labels[self.idx:i]=self.buffer["labels"]
		self.idx=i
		self.buffer={"data":[],"labels":[]}
		
	def storeClassLabels(self,classLabels): #will store raw string name of class labels in separate dataset
		dt=h5py.special_dtype(vlen=str)
		labelSet=self.db.create_dataset("label_names",(len(classLabels),),dtype=dt)
		labelSet[:]=classLabels
			
			
	def close(self):
		if len(self.buffer["data"])>0:
			self.flush()
				
		self.db.close()
