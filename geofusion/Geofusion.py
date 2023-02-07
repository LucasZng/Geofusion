import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
import umap.umap_ as umap

class Geofusion(object):
    def __init__(self):
        self.home_path = ''
        self.popAte9_scaler =  pickle.load(open(self.home_path + 'parameter/popAte9_scaler_scaler.pkl', 'rb') )
        self.popDe10a14_scaler = pickle.load(open(self.home_path + 'parameter/popDe10a14_scaler.pkl', 'rb'))
        self.popDe15a19_scaler =              pickle.load(open(self.home_path + 'parameter/popDe15a19_scaler.pkl', 'rb'))
        self.popDe20a24_scaler =                  pickle.load(open(self.home_path + 'parameter/popDe20a24_scaler.pkl', 'rb'))
        self.popDe25a34_scaler =            pickle.load(open(self.home_path + 'parameter/popDe25a34_scaler.pkl', 'rb'))
        self.popDe35a49_scaler =            pickle.load(open(self.home_path + 'parameter/popDe35a49_scaler.pkl', 'rb'))
        self.popDe50a59_scaler =            pickle.load(open(self.home_path + 'parameter/popDe50a59_scaler.pkl', 'rb'))
        self.popMaisDe60_scaler =            pickle.load(open(self.home_path + 'parameter/popMaisDe60_scaler.pkl', 'rb'))
        self.domiciliosA1_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosA1_scaler.pkl', 'rb'))
        self.domiciliosA2_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosA2_scaler.pkl', 'rb'))
        self.domiciliosB1_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosB1_scaler.pkl', 'rb'))
        self.domiciliosB2_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosB2_scaler.pkl', 'rb'))
        self.domiciliosC1_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosC1_scaler.pkl', 'rb'))
        self.domiciliosC2_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosC2_scaler.pkl', 'rb'))
        self.domiciliosD_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosD_scaler.pkl', 'rb'))
        self.domiciliosE_scaler =            pickle.load(open(self.home_path + 'parameter/domiciliosE_scaler.pkl', 'rb'))
        self.rendaMedia_scaler =            pickle.load(open(self.home_path + 'parameter/rendaMedia_scaler.pkl', 'rb'))
        self.popDe25a50_scaler =            pickle.load(open(self.home_path + 'parameter/popDe25a50_scaler.pkl', 'rb'))
        self.A_scaler =            pickle.load(open(self.home_path + 'parameter/A_scaler.pkl', 'rb'))
        self.B_scaler =            pickle.load(open(self.home_path + 'parameter/B_scaler.pkl', 'rb'))
        self.C_scaler =            pickle.load(open(self.home_path + 'parameter/C_scaler.pkl', 'rb'))
        
        
    def data_cleaning(self, df1):
        
        df1 = df1[df1['estado'] == 'SP']

        df1['rendaMedia'] = df1['rendaMedia'].astype(float64)

        ### 1.4. Fillout NA

        imputer = KNNImputer(n_neighbors=5)
        df_aux = df1.copy()
        df_aux = df_aux.drop(['nome', 'cidade', 'estado', 'população', 'popAte9', 'popDe10a14', 'popDe15a19', 
                           'popDe20a24', 'popDe25a34', 'popDe35a49', 'popDe50a59', 'popMaisDe60', 'faturamento', 'potencial'], axis = 1)
        df_aux = pd.DataFrame(imputer.fit_transform(df_aux), columns = df_aux.columns)

        df1 = df1.drop('rendaMedia', axis = 1)

        df1 = pd.merge(df1, df_aux[['rendaMedia', 'codigo']], how='left', on='codigo')
        df1 = df1[df1['população'] > 0]

        return df1
    
    def data_preparation(self, df3):
        
        # popAte9
        df3['popAte9'] = self.popAte9_scaler.transform(df3[['popAte9']].values)

        # popDe10a14
        df3['popDe10a14'] = self.popDe10a14_scaler.transform(df3[['popDe10a14']].values)

        # popDe15a19
        df3['popDe15a19'] = self.popDe15a19_scaler.transform(df3[['popDe15a19']].values)

        # popDe20a24
        df3['popDe20a24'] = self.popDe20a24_scaler.transform(df3[['popDe20a24']].values)

        # popDe25a34
        df3['popDe25a34'] = self.popDe25a34_scaler.transform(df3[['popDe25a34']].values)

        # popDe35a49
        df3['popDe35a49'] = self.popDe35a49_scaler.transform(df3[['popDe35a49']].values)

        # popDe50a59
        df3['popDe50a59'] = self.popDe50a59_scaler.transform(df3[['popDe50a59']].values)

        # popMaisDe60
        df3['popMaisDe60'] = self.popMaisDe60_scaler.transform(df3[['popMaisDe60']].values)

        # domiciliosA1
        df3['domiciliosA1'] = self.domiciliosA1_scaler.transform(df3[['domiciliosA1']].values)

        # domiciliosA2
        df3['domiciliosA2'] = self.domiciliosA2_scaler.transform(df3[['domiciliosA2']].values)

        # domiciliosB1
        df3['domiciliosB1'] = self.domiciliosB1_scaler.transform(df3[['domiciliosB1']].values)

        # domiciliosB2
        df3['domiciliosB2'] = self.domiciliosB2_scaler.transform(df3[['domiciliosB2']].values)
        
        # domiciliosC1
        df3['domiciliosC1'] = self.domiciliosC1_scaler.transform(df3[['domiciliosC1']].values)

        # domiciliosC2
        df3['domiciliosC2'] = self.domiciliosC2_scaler.transform(df3[['domiciliosC2']].values)

        # domiciliosD
        df3['domiciliosD'] = self.domiciliosD_scaler.transform(df3[['domiciliosD']].values)

        # domiciliosE
        df3['domiciliosE'] = self.domiciliosE_scaler.transform(df3[['domiciliosE']].values)

        # rendaMedia
        df3['rendaMedia'] = self.rendaMedia_scaler.transform(df3[['rendaMedia']].values)

        ### 3.2. Encoding
        
        cols_selected = ['popAte9', 'popDe10a14', 'popDe15a19', 'popDe20a24', 'popDe25a34', 'popDe35a49', 'popDe50a59', 'popMaisDe60',
                         'domiciliosA1', 'domiciliosA2', 'domiciliosB1', 'domiciliosB2', 'domiciliosC1', 'domiciliosC2', 'domiciliosD', 'domiciliosE', 'rendaMedia']
        
        return df3[cols_selected]

    def get_prediction_faturamento(self, model, original_data, test_data):
        
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['pred_faturamento'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')
    
    def get_prediction_potencial(self, model, original_data, test_data):
        
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['pred_potencial'] = pred
        original_data['pred_potencial'] = original_data['pred_potencial'].apply(lambda x: 'Baixo' if x == 0 else 'Médio' if x == 1 else 'Alto')
        return original_data.to_json(orient='records', date_format='iso')
    
    def get_tree(self, model, test_data):
        
        test_data['popDe25a50'] = test_data['popDe25a34'] + test_data['popDe35a49']
        
        # popDe50a59
        test_data['popDe25a50'] = self.popDe25a50_scaler.transform (test_data[['popDe25a50']].values)
        
        X_etaria = test_data.drop('popDe25a50', axis = 1)
        
        df_leaf = pd.DataFrame(model.apply(X_etaria))
        
        # Reduzer dimensionality
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(df_leaf)

        # embedding
        df_tree = pd.DataFrame()
        df_tree['embedding_x'] = embedding[:, 0]
        df_tree['embedding_y'] = embedding[:, 1]
        
        return df_tree
    
    def get_umap (self, test_data):
        
        test_data['A'] = test_data['domiciliosA1'] + test_data['domiciliosA2']
        test_data['B'] = test_data['domiciliosB1'] + test_data['domiciliosB2']
        test_data['C'] = test_data['domiciliosC1'] + test_data['domiciliosC2']

        test_data = test_data.drop(['domiciliosA1', 'domiciliosA2', 'domiciliosB1', 'domiciliosB2', 'domiciliosC1', 'domiciliosC2'], axis = 1)
        
        # domiciliosA
        test_data['A'] = self.A_scaler.transform (test_data[['A']].values)

        # domiciliosB
        test_data['B'] = self.B_scaler.transform (test_data[['B']].values)

        # domiciliosC
        test_data['C'] = self.C_scaler.transform (test_data[['C']].values)

        # domiciliosD
        test_data['domiciliosD'] = self.domiciliosD_scaler.transform (test_data[['domiciliosD']].values)

        # domiciliosE
        test_data['domiciliosE'] = self.domiciliosE_scaler.transform (test_data[['domiciliosE']].values)
        
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(test_data)

        # embedding
        df_umap = pd.DataFrame()
        df_umap['embedding_x'] = embedding[:, 0]
        df_umap['embedding_y'] = embedding[:, 1]
        
        return df_umap
    
    def get_cluster(self, model, original_data, test_data, cluster, pred_cluster):
        
        # model predict
        labels = model.predict(test_data)
        
        # join pred into the original data
        original_data[pred_cluster] = labels
        
        return original_data.to_json(orient='records', date_format='iso')