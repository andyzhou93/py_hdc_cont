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
        self.vectors = {}
        self.names = []
        self.array = np.empty((0,dim))
        
    # make sure array matches dict
    def _reset_array(self):
        self.array = np.empty((0,self.dim))
        self.names = []
        for v in self.vectors:
            self.array = np.concatenate((self.array,[self.vectors[v].value]))
            self.names.append(v)
        
    # create random name for vectors
    def _random_name(self):
        tempName = ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))
        while tempName in self.vectors:
            tempName = ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))
        return tempName
    
    # print the space
    def __repr__(self):
        return ''.join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)
    
    # add in a new, random vector
    def add(self, name=None):
        if name == None:
            name = self._random_name()
        v = Vector(self.dim)
        self.vectors[name] = v
        self._reset_array()
        return v
    
    # add in an already created vector
    def insert(self, v, name=None):
        if isinstance(v, Vector):
            if v.dim == self.dim:
                if name == None:
                    name = self._random_name()
                newV = Vector(self.dim)
                newV.value = v.value
                self.vectors[name] = newV
                self._reset_array()
                return name
            else:
                raise TypeError("Vector dimensions do not agree")
        else:
            raise TypeError("Unsupported type for vector space")
            
    # train a prototype vector
    def train(self, v, name=None):
        if isinstance(v, Vector):
            value = v.value
        elif isinstance(v, np.ndarray):
            value = v
        else:
            raise TypeError("Unsupported type for training")
        newVec = Vector(self.dim)
        newVec.value = value
        if name == None:
            name = self._random_name()
        elif name in self.vectors:
            self.vectors[name].accumulate(newVec)
        else:
            self.vectors[name] = newVec
        self._reset_array()
        

    # search for nearest neighbor in the space
    def find(self, v):
        if isinstance(v, Vector):
            if v.dim == self.dim:
                a = v.value
            else:
                raise TypeError("Vector dimensions do not agree")
        elif isinstance(v, Memory):
            if v.dim == self.dim:
                a = v.array
            else:
                raise TypeError("Vector dimensions do not agree")
        elif isinstance(v, np.ndarray):
            if v.shape[1] == self.dim:
                a = v
            else:
                raise TypeError("Vector dimensions do not agree")
        else:
            raise TypeError("Unsupported type for vector comparison")
        b = self.array
        x = a @ b.T
        y = np.outer(np.linalg.norm(a.T,axis=0), np.linalg.norm(b.T,axis=0))
        sim = x/y
#         print(np.argmax(sim,axis=1))
        maxIdx = np.argmax(sim,axis=1)
        label = [self.names[i] for i in maxIdx]
        return sim, label
    
    def bipolarize(self):
        for v in self.vectors:
            self.vectors[v].bipolarize()

            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            