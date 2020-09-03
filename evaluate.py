import numpy as np
import json

from tensorflow.keras.models import load_model
from model import MyModel

# vocab 불러오기
token_to_idx_filename = 'vocab/token_to_idx.json'
ner_to_index_filename = 'vocab/ner_to_index.json'

# config 저장 디렉토리
config_filename = 'config.json'

# checkpoint path
checkpoint_path = './tf_ckpts/ckpt-11'

# model path
saved_model_filename = 'saved_model/model'

with open(token_to_idx_filename, 'rb') as f:
    token_to_idx = json.load(f) 
    idx_to_token = {v: k for k, v in token_to_idx.items()}

with open(ner_to_index_filename, 'rb') as f:
    ner_to_index = json.load(f)  
    index_to_ner = {v: k for k, v in ner_to_index.items()}



'''
테스트 함수
num_of_case: test 데이터셋의 몇번째 인덱스 문장을 테스트 해볼건지
'''
def test_one(model, num_of_case, X_test, y_test):
    predictions = model(X_test[num_of_case:num_of_case+1], training=False)    
    
    print('문장 토큰|실제ner|예측ner')
    print('-------------------------')
    for i, j, t in zip(X_test[num_of_case], y_test[num_of_case], predictions[0].numpy()):
        print(idx_to_token[i]+'\t'+index_to_ner[np.argmax(j, axis=-1)]+'\t'+index_to_ner[np.argmax(t, axis=-1)])



if __name__ == "__main__":

    ########################################################################
    ## config
    ########################################################################

    # config 로딩
    print('config를 읽어옵니다...')

    with open(config_filename, 'rb') as f:
        config = json.load(f)   

    print('읽기 완료!\n')




    ########################################################################
    ## load evaluate data
    ########################################################################

    print('eval 데이터를 읽어옵니다...')
    X_test = np.load('train_val_eval_data/X_test.npy')
    y_test = np.load('train_val_eval_data/y_test.npy')
    print('읽기 완료!\n')




    ########################################################################
    ## load the model
    ########################################################################
    print('\n\nmodel을 읽어옵니다...')
    # # Recreate the exact same model, including its weights and the optimizer
    model = load_model(saved_model_filename)
    print('읽기 완료!\n')




    ########################################################################
    ## Evaluate
    ########################################################################

    num_of_case = 15
    print(f'*** {num_of_case}번째 문장 테스트')
    test_one(model, num_of_case, X_test, y_test)

    