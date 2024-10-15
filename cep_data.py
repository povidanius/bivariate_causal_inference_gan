import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk

class CEP(object):
    def __init__(self,folder):
        self.scaler = sk.MinMaxScaler(feature_range=(0, 1))
        self.folder = folder
        self.x = {}
        self.meta = np.loadtxt(self.folder + '/' + 'y.txt')
        self.enabled = np.loadtxt(self.folder + '/' + 'z.txt')
        self.n = len(self.meta)

        for i in range(self.n):
            fname = f'pair{i+1:04d}.txt'
            #print(fname)
            self.x[i] = self.load_pair(fname)
            print(len(self.x[i]))            
            if self.meta[i] == 0:
                    z = self.x[i] 
                    z[:,[0,1]] = z[:,[1,0]]
                    self.x[i] = z
            self.x[i] = self.normalize01(self.x[i])


    def load_pair(self,file_name):
       return np.loadtxt(self.folder + '/' + file_name)
        
    def get_sample(self,x,k):
       return np.array(random.choices(x, k=k))



    def normalize01(self,data):
        #print(data.min())
        #data = data - data.min(axis=0)
        #data = data / data.max()
        #return data
        scaler = self.scaler.fit(data)
        return scaler.transform(data)
        


if __name__ == "__main__":
        pairs = CEP('./pairs')       
        num_plots = 83
        fix, axis = plt.subplots(nrows=11,ncols=8)         
        j = 0
        for i in range(pairs.n):
            if pairs.enabled[i]:                                    
                d = pairs.get_sample(pairs.x[i], 400)   
                #print(d.shape)         
                ii = j // 8
                jj = j % 8
                print("{} {} {}".format(j, ii,jj))
                axis[ii,jj].plot(d[:,0], d[:,1],'r.', label=str(i+1))
                axis[ii,jj].legend()
                j = j + 1

                

        plt.show()

