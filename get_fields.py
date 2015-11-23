from astropy.io import fits

if __name__ == '__main__':
    '''We want all file opening stuff here o avoid confusion'''
    Path = '/home/maxi/data/'

    '''Opening data and config files'''
    hdulist_test = fits.open(Path + 'specPhotoDR12v3_hoyleb_extcorr_clean1e5.fit')
    hdulist_valid = fits.open(Path + 'specPhotoDR12v3_hoyleb_extcorr_clean1e5.fit')

    tbfields = hdulist_valid[1].data
    cols = hdulist_valid[1].columns
    x = cols.names
    x[:] = [i + '\n' for i in x]
    with open(Path+ 'fields_SDSS.conf', 'w') as f:
        f.write('########################################################################\
                \r#							Config File								   #\
                \r#	mark Targets by \'!\'		                                           #\
                \r#	exclude Features by \'#\'                                            #\
                \r#        															   #\
                \r########################################################################\n')
        for s in x:
            f.write(s)
