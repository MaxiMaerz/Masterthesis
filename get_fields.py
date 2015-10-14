from astropy.io import fits

if __name__ == '__main__':
    '''We want all file opening stuff here o avoid confusion'''
    Path = '/home/maxi/data/small_set/'

    '''Opening data and config files'''
    hdulist_test = fits.open(Path + 'PHAT1_TESTSET_MNPLz.fit')
    hdulist_valid = fits.open(Path + 'PHAT1_TRAINING.fits')

    tbfields = hdulist_valid[1].data
    cols = hdulist_valid[1].columns
    x = cols.names
    x[:] = [i + '\n' for i in x]
    with open(Path+ 'fields2.conf', 'w') as f:
        f.write('########################################################################\
                \r#							Config File								   #\
                \r#	mark Targets by \'!\'		                                           #\
                \r#	exclude Features by \'#\'                                            #\
                \r#        															   #\
                \r########################################################################\n')
        for s in x:
            f.write(s)
