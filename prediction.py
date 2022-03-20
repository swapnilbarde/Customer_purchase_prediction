import pickle

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

def purchase_predict(gender,age,salary):
    return loaded_model.predict([[gender,age,salary]])[0]
#purchase_predict(gender,age,salary)