import pathlib
import re

for path in ['KROK_2 LSTM SRNN.ipynb', 'KROK_3 XGBoost.ipynb']:
    txt = pathlib.Path(path).read_text(encoding='utf-8')
    print('\n===', path, '===')
    for pat in [
        'from sklearn.cross_validation import train_test_split',
        "data\\KROK_1\\BTC_data.csv",
        "objective':'reg:linear'",
        'xgb.DMatrix(x_train,y_train)',
        'trainY_sc = scaler.inverse_transform([trainY])',
        'testY_sc = scaler.inverse_transform([testY])'
    ]:
        count = txt.count(pat)
        print(f'PATTERN {pat!r}: {count}')
        if count:
            idx = txt.find(pat)
            print('...ctx:', txt[max(0, idx-100):idx+len(pat)+100])
