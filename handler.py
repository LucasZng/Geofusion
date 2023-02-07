import pickle
import pandas as pd
from flask             import Flask, request, Response
from geofusion.Geofusion import Geofusion
#import umap.umap_ as umap

# loading model
model_faturamento = pickle.load(open('model/regression.pkl', 'rb'))
model_potencial = pickle.load(open('model/classifier.pkl', 'rb'))
#model_rf_etaria = pickle.load(open('model/clust_rf_etaria.pkl', 'rb'))
#model_clust_etaria = pickle.load(open('model/clust_etaria.pkl', 'rb'))
#model_clust_classe = pickle.load(open('model/clust_classe.pkl', 'rb'))

#cols_etaria = [['popAte9', 'popDe10a14', 'popDe15a19', 'popDe20a24', 'popDe25a34', 'popDe35a49', 'popDe50a59', 'popMaisDe60']]
#cols_classe = [['domiciliosA1', 'domiciliosA2', 'domiciliosB1', 'domiciliosB2', 'domiciliosC1', 'domiciliosC2', 'domiciliosD', 'domiciliosE']]

# initialize API
app = Flask( __name__ )
@app.route( '/geofusion/predict', methods = ['POST'])

def geofusion_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        
        else: # multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # Instantiate Rossmann class
        pipeline = Geofusion()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # data preparation
        df3 = pipeline.data_preparation(df1)
        
        # faturamento
        df_faturamento = pipeline.get_prediction_faturamento(model_faturamento, test_raw, df3)
        
        # potencial
        df_potencial = pipeline.get_prediction_potencial(model_potencial, test_raw, df3)
        
        ## df_tre
        #df_tree = pipeline.get_tree(model_rf_etaria, df3[cols_etaria])
        #
        ## df_umap
        #df_umap = pipeline.get_umap(df3[cols_classe])
        #
        ## etaria
        #df_etaria = pipeline.get_cluster(model_clust_etaria, df_tree, test_raw, 3, 'etaria')
        #
        ## classe
        #df_classe = pipeline.get_cluster(model_clust_classe, df_umap, test_raw, 5, 'classe')
        
        df_response = df_faturamento.copy()
        df_responde['pred_potencial'] = df_potencial['pred_potencial']
        #df_responde['etaria'] = df_etaria['etaria']
        #df_responde['classe'] = df_classe['classe']
        
        return df_response
    
    else:
        return Reponse( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    app.run('0.0.0.0')