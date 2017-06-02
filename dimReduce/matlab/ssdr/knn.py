from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
import csv

def get_data(dataset):
    data = load_svmlight_file(dataset)
    return data[0],data[1]

result=[]
ratiol = [0.01,0.05,0.1,0.2,0.4,0.8]
alphal = [0.1,1,10,100]
betal = [0.2,2,20,200]
for ratio in ratiol:
    print ratio
    filepath = '../testdata/gis/' + str(ratio) + '/'
    for i in range(1,5):
        for j in range(1,5):
            traind = filepath + 'newtf'
            vtraind = filepath + 'vtf'
            ted = filepath + 'tef'
            backname = '_' + str(i) + '_'+str(j)
            tf, tl = get_data(traind+backname)
            vtf, vtl = get_data(vtraind+backname)
            tef, tel = get_data(ted+backname)
            nnl= [1,2,3,4];
            for nn in nnl: 
                neigh = KNeighborsClassifier(n_neighbors=nn)
                neigh.fit(tf.todense(),tl)
                vres = neigh.predict(vtf.todense())
                tres = neigh.predict(tef.todense())
                vcor = 0
                tcor = 0
                for t in range(vtl.shape[0]):
                    if vtl[t] == vres[t]:
                        vcor+=1
                for t in range(tel.shape[0]):
                    if tel[t] == tres[t]:
                        tcor+=1

                acc = 1.0*vcor/vtl.shape[0]
                teacc = 1.0*tcor/tel.shape[0]
                csvrow = (alphal[i-1],betal[j-1],nn,teacc,acc)
                result.append(csvrow)

with open('./gisssdr.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerow(['alpha','beta','nn','test_acc','vali_acc'])
    writer.writerows(result)

f.close()
