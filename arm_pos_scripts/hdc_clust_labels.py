import numpy as np
import random

# basic bipolar hypervector class
class Vector:
    # init random HDC vector
    def __init__(self, dim, value=np.empty(0)):
        self.dim = dim # dimension of hypervector
        # either use predetermined value or a random value is none given
        if value.any():
            self.value = value
        else:
            self.value = np.random.choice([-1.0, 1.0], size=dim)
    
    # print vector
    def __repr__(self):
        return np.array2string(self.value)
    
    # print vector
    def __str__(self):
        return np.array2string(self.value)
    
    # Read-only operations (for creating other vectors) with basic symbols
    # addition
    def __add__(self, a):
        if isinstance(a, self.__class__):
            if a.dim == self.dim:
                b = Vector(self.dim)
                b.value = a.value + self.value
                return b
            else:
                raise TypeError("Vector dimensions do not agree")
        else:
            raise TypeError("Unsupported type for vector addition")
    
    # multiplication
    def __mul__(self, a):
        if isinstance(a, self.__class__):
            if a.dim == self.dim:
                b = Vector(self.dim)
                b.value = a.value * self.value
                return b
            else:
                raise TypeError("Vector dimensions do not agree")
        elif isinstance(a, (float, int)):
            b = Vector(self.dim)
            b.value = a * self.value
            return b
        else:
            raise TypeError("Unsupported type for vector multiplication")
            
    # permutation
    def __rshift__(self, a):
        b = Vector(self.dim)
        b.value = np.roll(self.value, a)
        return b
    
    __lshift__ = __rshift__
    
    # cosine similarity
    def __mod__(self, a):
        if isinstance(a, self.__class__):
            if (a.dim == self.dim):
                return np.dot(self.value, a.value)/(np.linalg.norm(self.value) * np.linalg.norm(a.value))
            else:
                raise TypeError("Vector dimensions do not agree")
        else:
            raise TypeError("Unsupported type for vector cosine similarity")
            
    # bipolarization
    def __abs__(self):
        b = Vector(self.dim)
        b.value = self.value
        z = b.value
        z[z > 0] = 1.0
        z[z < 0] = -1.0
        z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
        b.value = z
        return b
    
    # Write-access functions
    # accumulator new vector
    def accumulate(self, a):
        if isinstance(a, self.__class__):
            if a.dim == self.dim:
                self.value = self.value + a.value
            else:
                raise TypeError("Vector dimensions do not agree")
        else:
            raise TypeError("Unsupported type for vector accumulation")
            
    # bind vector into new vector
    def multiply(self, a):
        if isinstance(a, self.__class__):
            if a.dim == self.dim:
                self.value = self.value * a.value
            else:
                raise TypeError("Vector dimensions do not agree")
        elif isinstance(a, (float, int)):
            self.value = a * self.value
        else:
            raise TypeError("Unsupported type for vector accumulation")
            
    # permute vector
    def permute(self, a):
        self.value = np.roll(self.value, a)
        
    # bipolarize vector
    def bipolarize(self):
        z = self.value
        z[z > 0] = 1.0
        z[z < 0] = -1.0
        z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
        self.value = z
        
    # clear the vector
    def clear(self):
        self.value = np.zeros(self.dim)
        

# vector space associative memory
class Memory:
    # init vector space
    def __init__(self, dim=10000):
        self.dim = dim
        self.vec = np.empty((0,dim))
        self.biVec = np.empty((0,dim))
        self.classes = []
        self.clusters = []
        self.clustLabels = []
        
    # print the space
    def __repr__(self):
        return ''.join("Class %d, Cluster %d (%d): %s\n" % (self.classes[v], self.clustLabels[v], self.clusters[v], str(self.vec[v,:])) for v in range(len(self.classes)))
    
    # write a new vector into space
    def write(self, v, vClass=None, vClust=None):
        # make sure type is correct, and sum along elements if ndarray
        if not isinstance(v, (Vector, np.ndarray)):
            raise TypeError("Unsupported type for vector space")
        else:
            if isinstance(v, Vector):
                if v.dim != self.dim:
                    raise TypeError("Vector dimensions do not agree")
                else:
                    v = v.value
            else:
                if v.ndim > 1:
                    if v.shape[1] != self.dim:
                        raise TypeError("Vector dimensions do not agree")
                    else:
                        v = v.sum(axis=0)
                else:
                    if len(v) != self.dim:
                        raise TypeError("Vector dimensions do not agree")

        # determine where to store
        if vClass == None: # no specified class, so add as a new class and new cluster
            if self.classes:
                vClass = max(self.classes) + 1
            else:
                vClass = 0
            vClust = 0
            self.classes.append(vClass)
            self.clusters.append(vClust)
            self.clustLabels.append(vClust)
            self.vec = np.concatenate((self.vec, [v]))
        elif vClust == None: # class is specified, but not cluster, so add as new cluster in class
            classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            if classIdx:
                vClust = max([self.clusters[idx] for idx in classIdx]) + 1
            else:
                vClust = 0
            self.classes.append(vClass)
            self.clusters.append(vClust)
            self.clustLabels.append(vClust)
            self.vec = np.concatenate((self.vec, [v]))
        else: # class and cluster are specified, so check if it already exists
            if vClass in self.classes:
                # class already exists, need to check for cluster
                classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
                clusterIdx = [idx for idx in classIdx if self.clusters[idx] == vClust]
                if not clusterIdx:
                    self.classes.append(vClass)
                    self.clusters.append(vClust)
                    self.clustLabels.append(vClust)
                    self.vec = np.concatenate((self.vec, [v]))
                else:
                    self.vec[clusterIdx,:] = v
            else:
                self.classes.append(vClass)
                self.clusters.append(vClust)
                self.clustLabels.append(vClust)
                self.vec = np.concatenate((self.vec, [v]))

        return vClass, vClust

    # remove class or clusters from memory
    def remove(self, vClass, vClust=None):
        if vClust == None:
            keepIdx = [idx for idx, val in enumerate(self.classes) if val != vClass]
            self.classes = [self.classes[idx] for idx in keepIdx]
            self.clusters = [self.clusters[idx] for idx in keepIdx]
            self.clustLabels = [self.clustLabels[idx] for idx in keepIdx]
            deleteIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            self.vec = np.delete(self.vec,deleteIdx,axis=0)
        else:
            classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            clusterIdx = [idx for idx in classIdx if self.clusters[idx] == vClust]
            self.vec = np.delete(self.vec,clusterIdx,axis=0)
            keepIdx = [idx for idx in range(len(self.clusters)) if idx not in clusterIdx]
            self.classes = [self.classes[idx] for idx in keepIdx]
            self.clusters = [self.clusters[idx] for idx in keepIdx]
            self.clustLabels = [self.clustLabels[idx] for idx in keepIdx]

    # read class or clusters from memory
    def read(self, vClass=None, vClust=None):
        if vClass == None: # no specified class, return all of memory
            return self.classes, self.clusters, self.clustLabels, self.vec
        elif vClust == None: # specified class, but not cluster, so return all clusters of that class
            classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            classes = [self.classes[idx] for idx in classIdx]
            clusters = [self.clusters[idx] for idx in classIdx]
            clustLabels = [self.clustLabels[idx] for idx in classIdx]
            vec = self.vec[classIdx,:]
            return classes, clusters, clustLabels, vec
        else: # specified both class and cluster
            classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            clusterIdx = [idx for idx in classIdx if self.clusters[idx] == vClust]
            classes = [self.classes[idx] for idx in clusterIdx]
            clusters = [self.clusters[idx] for idx in clusterIdx]
            clustLabels = [self.clustLabels[idx] for idx in clusterIdx]
            vec = self.vec[clusterIdx,:]
            return classes, clusters, clustLabels, vec

    # train by accumulating a new vector into space
    def train(self, v, vClass=None, vClust=None):
        # make sure type is correct, and sum along elements if ndarray
        if not isinstance(v, (Vector, np.ndarray)):
            raise TypeError("Unsupported type for vector space")
        else:
            if isinstance(v, Vector):
                if v.dim != self.dim:
                    raise TypeError("Vector dimensions do not agree")
                else:
                    v = v.value
            else:
                if v.ndim > 1:
                    if v.shape[1] != self.dim:
                        raise TypeError("Vector dimensions do not agree")
                    else:
                        v = v.sum(axis=0)
                else:
                    if len(v) != self.dim:
                        raise TypeError("Vector dimensions do not agree")

        # determine where to store
        if vClass == None: # no specified class, so add as a new class and new cluster
            if self.classes:
                vClass = max(self.classes) + 1
            else:
                vClass = 0
            vClust = 0
            self.classes.append(vClass)
            self.clusters.append(vClust)
            self.clustLabels.append(vClust)
            self.vec = np.concatenate((self.vec, [v]))
        elif vClust == None: # class is specified, but not cluster, so add as new cluster in class
            classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            if classIdx:
                vClust = max([self.clusters[idx] for idx in classIdx]) + 1
            else:
                vClust = 0
            self.classes.append(vClass)
            self.clusters.append(vClust)
            self.clustLabels.append(vClust)
            self.vec = np.concatenate((self.vec, [v]))
        else: # class and cluster are specified, so check if it already exists
            if vClass in self.classes:
                # class already exists, need to check for cluster
                classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
                clusterIdx = [idx for idx in classIdx if self.clusters[idx] == vClust]
                if not clusterIdx:
                    self.classes.append(vClass)
                    self.clusters.append(vClust)
                    self.clustLabels.append(vClust)
                    self.vec = np.concatenate((self.vec, [v]))
                else:
                    self.vec[clusterIdx,:] =  self.vec[clusterIdx,:] + v
            else:
                self.classes.append(vClass)
                self.clusters.append(vClust)
                self.clustLabels.append(vClust)
                self.vec = np.concatenate((self.vec, [v]))

        return vClass, vClust

    # search for nearest neighbor in the space
    def match(self, v, bipolar=False):
        # make sure datatype is correct
        if not isinstance(v, (Vector, np.ndarray)):
            raise TypeError("Unsupported type for vector space")
        else:
            if isinstance(v, Vector):
                if v.dim != self.dim:
                    raise TypeError("Vector dimensions do not agree")
                else:
                    v = v.value
            else:
                if v.ndim > 1:
                    if v.shape[1] != self.dim:
                        raise TypeError("Vector dimensions do not agree")
                else:
                    if len(v) != self.dim:
                        raise TypeError("Vector dimensions do not agree")

        am = np.copy(self.vec)
        if bipolar:
            z = am
            z[z > 0] = 1.0
            z[z < 0] = -1.0
            z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
            am = z

        x = v @ am.T
        y = np.outer(np.linalg.norm(v.T,axis=0), np.linalg.norm(am.T,axis=0))
        sim = x/y
        maxIdx = np.argmax(sim,axis=1)
        predClass = [self.classes[i] for i in maxIdx]
        predClust = [self.clusters[i] for i in maxIdx]
        predClustLabel = [self.clustLabels[i] for i in maxIdx]
        return predClass, predClust, predClustLabel, sim


    # train by adaptively writing new examples as new prototypes
    def train_sub_cluster(self, v, vClass, vClustLabel=0, threshold=0.55):
        # make sure type is correct
        if not isinstance(v, (Vector, np.ndarray)):
            raise TypeError("Unsupported type for vector space")
        else:
            if isinstance(v, Vector):
                if v.dim != self.dim:
                    raise TypeError("Vector dimensions do not agree")
                else:
                    v = v.value
            else:
                if v.ndim > 1:
                    if v.shape[1] != self.dim:
                        raise TypeError("Vector dimensions do not agree")
                else:
                    if len(v) != self.dim:
                        raise TypeError("Vector dimensions do not agree")

        classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
        if classIdx:
            if v.ndim > 1:
                for i in range(v.shape[0]):
                    label, clustDummy, sim = self.match(v[i,:])
                    sim = sim[0,classIdx]
                    bestMatch = np.argmax(sim)
                    if sim[bestMatch] > threshold:
                        self.vec[classIdx[bestMatch],:] = self.vec[classIdx[bestMatch],:] + v[i,:]
                    else:
                        self.classes.append(vClass)
                        vClust = max([self.clusters[idx] for idx in classIdx]) + 1
                        self.clusters.append(vClust)
                        self.clustLabels.append(vClustLabel)
                        self.vec = np.concatenate((self.vec, [v[i,:]]))
                    classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
            else:
                label, clustDummy, sim = self.match(v)
                sim = sim[0,classIdx]
                bestMatch = np.argmax(sim)
                if sim[bestMatch] > threshold:
                    self.vec[classIdx[bestMatch],:] = self.vec[classIdx[bestMatch],:] + v
                else:
                    self.classes.append(vClass)
                    vClust = max([self.clusters[idx] for idx in classIdx]) + 1
                    self.clusters.append(vClust)
                    self.clustLabels.append(vClustLabel)
                    self.vec = np.concatenate((self.vec, [v]))
        else:
            if v.ndim > 1:
                self.classes.append(vClass)
                self.clusters.append(0)
                self.clustLabels.append(vClustLabel)
                self.vec = np.concatenate((self.vec, [v[0,:]]))
                for i in range(1,v.shape[0]):
                    classIdx = [idx for idx, val in enumerate(self.classes) if val == vClass]
                    label, clustDummy, sim = self.match(v[i,:])
                    sim = sim[0,classIdx]
                    bestMatch = np.argmax(sim)
                    if sim[bestMatch] > threshold:
                        self.vec[classIdx[bestMatch],:] = self.vec[classIdx[bestMatch],:] + v[i,:]
                    else:
                        self.classes.append(vClass)
                        vClust = max([self.clusters[idx] for idx in classIdx]) + 1
                        self.clusters.append(vClust)
                        self.clustLabels.append(vClustLabel)
                        self.vec = np.concatenate((self.vec, [v[i,:]]))
            else:
                self.classes.append(vClass)
                self.clusters.append(0)
                self.clustLabels.append(vClustLabel)
                self.vec = np.concatenate((self.vec, [v]))
            
            
    # prune unuseful vectors
    def prune(self, threshold=101, min=1):
        uniqClass = np.unique(self.classes)
        for c in uniqClass:
            classIdx = [idx for idx, val in enumerate(self.classes) if val == c]
            if len(classIdx) > min:
                norms = np.linalg.norm(self.vec[classIdx,:], axis=1)
                deleteIdx = [classIdx[idx] for idx, val in enumerate(norms) if val < threshold]
                while len(classIdx) - len(deleteIdx) < min:
                    randIdx = np.random.randint(len(deleteIdx))
                    deleteIdx.pop(randIdx)
                self.vec = np.delete(self.vec,deleteIdx,axis=0)
                self.classes = [self.classes[idx] for idx in range(len(self.classes)) if idx not in deleteIdx]
                self.clusters = [self.clusters[idx] for idx in range(len(self.clusters)) if idx not in deleteIdx] 
                self.clustLabels = [self.clustLabels[idx] for idx in range(len(self.clustLabels)) if idx not in deleteIdx] 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
