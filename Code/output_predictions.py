import xgboost as xgb
import pandas as pd
import settings
import operator


def average_models_and_save(modelfiles,testX,ycoeff,xcoeff,INDIR):
    for k, modelfile in enumerate(modelfiles):
        gbm = xgb.Booster({'nthread':4})
        gbm.load_model(modelfile)
        ypred = gbm.predict(xgb.DMatrix((testX/xcoeff)))
        ypredout = (ypred*np.float64(ycoeff)).reshape(1796,98,order='F')
        sampsub = None
        if sampsub is None:
            sampsub = pd.read_csv(os.getcwd()+'/sampleSubmission.csv')
            for i,cols in enumerate(sampsub.columns.values[1:]):
                sampsub[cols] = ypredout[:,i]
        else:
            for i,cols in enumerate(sampsub.columns.values[1:]):
                sampsub[cols] = sampsub[cols]+ypredout[:,i]
    sampsub.to_csv(INDIR+'submit.csv',index=False)


def output_model(gbm, testX, subname, ycoeff, featnames):

    # Predict the Y data
    ypred = gbm.predict(xgb.DMatrix(testX, feature_names=featnames))

    # Reverse scale the Y data
    ypred = ypred * ycoeff
    ypred_out = ypred.reshape(1796, 98, order='F')

    sampsub = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV)
    for i, cols in enumerate(sampsub.columns.values[1:]):
        sampsub[cols] = ypred_out[:, i]

    sampsub.to_csv(subname, index=False)
