def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.

    """
    paramas = request.get_json()
    showdata = {"success":False}
    import pandas as pd
    import numpy as np
    
    from google.cloud import storage
    client=storage.Client()
    bucket=client.get_bucket('qwiklabs-gcp-02-81b9a19561ac.appspot.com')
    blob=bucket.get_blob(file['name'])
    contents = blob.download_as_string()
    print(contents)
    from io import StringIO
    contents=str(contents,"utf-8")
    contents=StringIO(contents)
    data=pd.read_csv(contents)
    data=np.array(data)
    print(data);
    data=data['Open']
    data=np.array(data)
    data=np.reshape(data,(1235,1))
    scaler=MinMaxScaler()
    data=scaler.fit_transform(data)
    data=np.reshape(data,(1235))
    data_training =data[100:]
    data_test = data[:100]
    data_test=data_test[::-1] 
    data_training=data_training[::-1]
    X_train = []
    y_train = []
    for i in range(60, data_training.shape[0]):
        X_train.append(data_training[i-60:i])
        y_train.append(data_training[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape)
    print(y_train.shape)
    print(X_train.shape[1])
    X_train=np.reshape(X_train,(1075,60,1))
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    regressior = Sequential()
    regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    regressior.add(Dropout(0.2))
    regressior.add(LSTM(units = 120, activation = 'relu'))
    regressior.add(Dropout(0.2))
    regressior.add(Dense(units = 1))
    regressior.compile(optimizer='adam', loss = 'mean_squared_error',metrics=['accuracy'])
    hist=regressior.fit(X_train, y_train, epochs=8, batch_size=32)
    preddata=data[:59]
    preddata=np.reshape(preddata,(-1,1))
    preddata=scaler.inverse_transform(preddata)
    preddata=preddata[::-1]
    xdata=[]
    xdata.append(paramas['Open'])
    preddata=np.append(preddata,xdata)
    preddata=np.reshape(preddata,(-1,1))
    preddata=scaler.transform(preddata)
    preddata=np.reshape(preddata,(1,60,1))
    predvalue=regressior.predict(preddata)
    tranpred=predvalue
    print(tranpred)
    tranpred=scaler.inverse_transform(tranpred)
    print(tranpred)
