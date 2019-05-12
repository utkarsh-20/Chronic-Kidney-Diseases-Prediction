import flask
from flask import Flask, render_template, request
from lib.preprocessing import *

val_dict = {'age': {'NaN': 55.0},
            'bp': {'NaN': 80.0},
            'sg': {'NaN': 0, '1.005': 1, '1.010': 2, '1.015': 3, '1.020': 4, '1.025': 5},
            'al': {'NaN': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6},
            'su': {'NaN': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6},
            'rbc': {'NaN': 0, 'abnormal': 1, 'normal': 2},
            'pc': {'NaN': 0, 'abnormal': 1, 'normal': 2},
            'pcc': {'NaN': 0, 'notpresent': 1, 'present': 2},
            'ba': {'NaN': 0, 'notpresent': 1, 'present': 2},
            'bgr': {'NaN': 121.0},
            'bu': {'NaN': 42.0},
            'sc': {'NaN': 1.3},
            'sod': {'NaN': 138.0},
            'pot': {'NaN': 4.4},
            'hemo': {'NaN': 12.649999999999999},
            'pcv': {'NaN': 40.0},
            'wc': {'NaN': 8000.0},
            'rc': {'NaN': 4.8},
            'htn': {'NaN': 0, 'no': 1, 'yes': 2},
            'dm': {'NaN': 0, 'no': 1, 'yes': 2},
            'cad': {'NaN': 0, 'no': 1, 'yes': 2},
            'appet': {'NaN': 0, 'poor': 1, 'good': 2},
            'pe': {'NaN': 0, 'no': 1, 'yes': 2},
            'ane': {'NaN': 0, 'no': 1, 'yes': 2},
            }

app = Flask(__name__)


@app.route('/')
def entry():
    return render_template('form.html', the_title='CKD prediction', form_title='Input Data for Prediction')


@app.route('/analyse', methods=['GET', 'POST'])
def proc():
    test_val = create_df()
    for feature in test_val.columns:
        if request.form[feature]:
            test_val[feature] = request.form[feature]
        else:
            test_val[feature] = 'NaN'
    fix_missing(test_val, val_dict)
    test_val = test_val.astype(float)
    scaler = load_scaler()
    test_val = scaler.transform(test_val)
    resp = {}
    model = load_model()
    pred = model.predict(test_val)
    resp['prediction'] = str(pred).strip('[]')
    if pred == 1:
        resp['inference'] = 'Patient has Chronic Kidney Disease'
    else:
        resp['inference'] = 'Patient does not has Chronic Kidney Disease'
    return render_template('result.html', pred=resp['inference'])


if __name__ == '__main__':
    app.run(debug=True)

