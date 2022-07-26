
def Label_Encoder_only(data,columns_name,after_name):
    from sklearn.preprocessing import LabelEncoder
    df=data
    df = df.dropna(subset=[columns_name])
    le = LabelEncoder()
    encoded = le.fit_transform(df[columns_name].values)
    decoded = le.inverse_transform(encoded)
    df[after_name] = encoded
    df_out=df.drop([columns_name], axis=1)
    return df_out
