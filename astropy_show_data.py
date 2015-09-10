__author__ = 'maxi'
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
import os
from sklearn.externals.six import StringIO

def export_data():
    """
    Read the Data of the File
    :return: Data, Header in HDULIST Format
    """

    hdulist = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    tbdata = hdulist[1].data
    header = hdulist[0].header
    return tbdata, header

#def split_data(data, ):

def generate_matrix(data, Targets):
    hdulist = fits.open('/home/maxi/data/y1a1_stripe82_train_subset.fits')
    for x in range(len(astro_data[0])):
        if isinstance(data[1].columns[x], (int, float, complex)) is True
            temp = np.array([astro_data.field(x)])
            data_float = np.hstack((data, temp))



if __name__ == '__main__':
    """Export the Data"""
    astro_data, astro_header = export_data()

    """Plot the Data"""
    #plt.plot(astro_data['Z_SPEC'], astro_data['MAG_AUTO_Z'], '.')
    #plt.show()

    """Get a small Sample for testing"""
    size_data = astro_data['Z_SPEC'].size

    sample_size = round(size_data/100)
    target = np.empty(sample_size)
    value = np.empty([sample_size,1])

    print(repr(astro_data['MAG_AUTO_Y']))
    #plt.plot(target, value, '.')
    #plt.show()
    test = np.array([astro_data.field(0), astro_data['MAG_AUTO_Y']]).T
    #test = np.expand_dims(test, axis=0)
    print(repr(test))
    print(astro_data.field(0))
    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf.fit(np.expand_dims(astro_data, axis=1), astro_data['Z_SPEC'])


    X_TEST = (np.arange(12.0, 24, 12/size_data))
    Y_TEST = (np.arange(12.0, 24, 12/size_data))
    #input = np.array([astro_data, Y_TEST]).T
    print(X_TEST.shape)
    y_predicted = clf.predict(np.expand_dims(astro_data, axis=1))

    #plt.figure()
    #plt.plot(astro_data['MAG_AUTO_Z'], astro_data['Z_SPEC'], '.')
    #plt.plot(astro_data['MAG_AUTO_Z'],y_predicted, '*', c='g')
    #plt.show()


    with open("/home/maxi/data/iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    os.system('dot -Tpdf /home/maxi/data/iris.dot -o /home/maxi/data/iris.pdf')
    os.unlink('/home/maxi/data/iris.dot')

