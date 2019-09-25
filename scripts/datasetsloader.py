from config import *
from utils.dataloader import InputData


def load_iemocap(output=True):
    input_data = InputData(CORPUS+"iemocap.nc")
    N = 5
    map_dict = {"ang": 0, "hap": 1, "exc": 1, "neu": 2, "sad": 3}
    X = []   # input features
    y = []   # labels (category)
    for i in range(1, N+1):
        session = input_data.get_session("Ses0"+str(i), output)
        X.append(input_data.get_features(session))
        y.append(input_data.get_category(session).map(map_dict))
    n_ang = len(pd.concat(X)[pd.concat(y)==0])
    n_hap_exc = len(pd.concat(X)[pd.concat(y)==1])
    n_neu = len(pd.concat(X)[pd.concat(y)==2])
    n_sad = len(pd.concat(X)[pd.concat(y)==3])
    if output:
        print("Number of IEMOCAP samples: angry ({}), happy+excited ({}), "
        "neutral ({}), sad ({})\n".format(n_ang, n_hap_exc, n_neu, n_sad))
    return X, y


def load_tedlium(output=True):
    data_path = "/mount/arbeitsdaten34/projekte/slu/michael/data/" + \
    "TEDLIUM_release2_wav/tedlium_emobase2010.nc"
    input_data_u = InputData(data_path)
    X_u = input_data_u.get_features()
    if output:
        print("Number of TEDLIUM samples: {}\n".format(len(X_u)))
    return X_u


def load_synthetic_data(lambda_cls, output=True):
    ang = pd.read_hdf(RESULTS+'train/syn_samples/syn_total_5_'+ \
                      str(lambda_cls)+'.h5', key='ang')
    hap = pd.read_hdf(RESULTS+'train/syn_samples/syn_total_5_'+ \
                      str(lambda_cls)+'.h5', key='hap')
    neu = pd.read_hdf(RESULTS+'train/syn_samples/syn_total_5_'+ \
                      str(lambda_cls)+'.h5', key='neu')
    sad = pd.read_hdf(RESULTS+'train/syn_samples/syn_total_5_'+ \
                      str(lambda_cls)+'.h5', key='sad')
    X_syn = pd.concat([ang, hap, neu, sad])
    y_syn = pd.Series(np.concatenate([[0]*len(ang), 
                                      [1]*len(hap), 
                                      [2]*len(neu),
                                      [3]*len(sad)]))
    y_syn.index = X_syn.index.values
    if output:
        print("Number of syn_{}: angry ({}), happy+excited ({}), "
        "neutral ({}), sad ({})\n".format(lambda_cls, len(ang), 
                                         len(hap), len(neu), len(sad)))
    return X_syn, y_syn


def load_msp(output=True):
    map_dict = {"ang": 0, "hap": 1, "exc": 1, "neu": 2, "sad": 3}
    input_data_msp = InputData(CORPUS+"msp.nc")
    X_msp = input_data_msp.get_features()
    y_msp = input_data_msp.get_category().map(map_dict)
    if output:
        print("Number of msp samples: angry ({}), happy ({}), "
             "neutral ({}), sad ({})\n".format(len(X_msp[y_msp==0]),
                                               len(X_msp[y_msp==1]),
                                               len(X_msp[y_msp==2]),
                                               len(X_msp[y_msp==3])))
    return X_msp, y_msp

