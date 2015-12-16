def read_hdu(hdu, feature_conf):
    astro_data = hdu[1].data
    for i in range(len(feature_conf)):
        if feature_conf[i][0] != '#' and feature_conf[i][0] != '!':
            try:
                features
                features = np.vstack((features, astro_data[feature_conf[i]]))
            except NameError:
                features = astro_data[feature_conf[i]]
        if feature_conf[i][0] == '!':
            targets = astro_data[feature_conf[i][1:]]
    return features.T, targets.T

def generate_features(data):
    N = len(data[0])
    for i in range(N):
        for j in range(N):
            if(i != j):
                new = data[:,i] - data[:,j]
                data = np.hstack((data, np.expand_dims(new, axis=1)))
    return data

def clean_data(data, x):
    average = np.empty((len(data[0])))
    for i in range(0, len(data[0])):
        average[i] = (np.sum(data[:,i][abs(data[:,i]) != x]) /
                      len(data[:,i][abs(data[:,i]) != x]))

    for i in range(0, len(data[0])):
        data[:,i][abs(data[:,i]) == x] = average[i]
    return data