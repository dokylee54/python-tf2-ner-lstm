import numpy as np
import json
from sklearn.metrics import f1_score

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
전체 테스트 함수
f1-score 계산
'''
def test_all(model, X_test, y_test):
    predictions = model(X_test, training=False)   

    y_pred = np.array([np.argmax(prediction.numpy(), axis=-1) for prediction in predictions])

    y_test = np.array([np.argmax(y, axis=-1) for y in y_test])

    f1 = f1_score(y_test.flatten(), y_pred.flatten(), average='micro')
    print(f'f1-score = {f1}')
    print('\n\n')



'''
일부 테스트 함수
start_of_case: test 데이터셋의 몇번째 인덱스 문장을 시작으로 테스트 해볼건지
end_of_case: 몇번째 인덱스 문장을 끝으로 테스트
'''
def test_one(model, start_of_case, end_of_case, X_test, y_test):
    predictions = model(X_test[start_of_case:end_of_case+1], training=False)   

    for idx, prediction in enumerate(predictions):
        print(f'>> #{idx+1}번째 문장 테스트')
        print('문장 토큰\t|실제ner\t|예측ner')
        print('-'*45)
        for i, j, t in zip(X_test[start_of_case+idx], y_test[start_of_case+idx], prediction.numpy()):
            print(idx_to_token[i]+'\t\t'+index_to_ner[np.argmax(j, axis=-1)]+'\t\t'+index_to_ner[np.argmax(t, axis=-1)])
        print('='*45)



if __name__ == "__main__":

    ########################################################################
    ## config
    ########################################################################

    # config 로딩
    print('>> config를 읽어옵니다...')

    with open(config_filename, 'rb') as f:
        config = json.load(f)   

    print('>> 읽기 완료!\n')




    ########################################################################
    ## load evaluate data
    ########################################################################

    print('>> eval 데이터를 읽어옵니다...')
    X_test = np.load('train_val_eval_data/X_test.npy')
    y_test = np.load('train_val_eval_data/y_test.npy')
    print('>> 읽기 완료!\n')




    ########################################################################
    ## load the model
    ########################################################################
    print('\n\n>> model을 읽어옵니다...')
    # # Recreate the exact same model, including its weights and the optimizer
    model = load_model(saved_model_filename)
    print('>> 읽기 완료!\n')




    ########################################################################
    ## Evaluate All
    ########################################################################

    print(f'*** 전체 문장 테스트: 총 {len(X_test)}개')
    test_all(model, X_test, y_test)


    ########################################################################
    ## Evaluate Part
    ########################################################################

    start_of_case = 0
    end_of_case = 1
    print(f'*** {start_of_case} ~ {end_of_case}번째 문장 테스트')
    test_one(model, start_of_case, end_of_case, X_test, y_test)
    