from pathlib import Path
import codecs
import os
from tqdm import tqdm
from konlpy.tag import Okt
from collections import Counter
import re
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 훈련시킬 데이터 경로 정의
train_data_dir = Path('data')

# 전처리 후 데이터 경로 정의
train_val_eval_data_dir = Path('train_val_eval_data')

# config 저장 디렉토리
config_filename = 'config.json'

# vocab 디렉토리
vocab_dir = Path('vocab')
token_to_idx_filename = 'vocab/token_to_idx.json'
ner_to_index_filename = 'vocab/ner_to_index.json'


# 스페셜 토큰 정의
PAD = "[PAD]"
START_TOKEN = "<S>"
END_TOKEN = "<T>"
UNK = "[UNK]"
CLS = "[CLS]"
MASK = "[MASK]"
SEP = "[SEP]"
SEG_A = "[SEG_A]"
SEG_B = "[SEG_B]"
NUM = "<num>"

padding_token = PAD
cls_token = CLS
sep_token = SEP

special_tokens = [PAD, START_TOKEN, END_TOKEN, UNK, CLS, MASK, SEP, SEG_A, SEG_B, NUM]  

# 토큰화 시킬 라이브러리 함수 설정
split_fn = Okt().morphs

# 라벨 데이터 원핫인코딩 위한 config
ner_tag_size = 0

# 불러온 데이터 전처리 결과 리스트 변수
list_of_total_source_no, list_of_total_source_str, list_of_total_target_str = [], [], []


# 전처리 리스트 변수
list_of_X_tokens = []
list_of_padded_X_token_ids_with_cls_sep = []
list_of_padded_ner_ids_with_cls_sep = []

# padding congif
maxlen = 0



'''
파일 한개에서 데이터 읽기
'''
def load_data_from_one_txt(full_file_name):
    
    with codecs.open(full_file_name, 'r', 'utf-8') as io:
        lines = io.readlines()

        chunked_lines = []

        for line in lines:

            # '/n' 제거
            line = line.rstrip()

            if line[:3] == '## ':
                chunked_lines.append(line[3:])

    list_of_source_no, list_of_source_str, list_of_target_str = chunked_lines[0::3], chunked_lines[1::3], chunked_lines[2::3]
    
    return list_of_source_no, list_of_source_str, list_of_target_str
    


'''
파일 전체에서 데이터 읽기
'''
def load_data_from_all_txt(train_data_dir):
    
    list_of_file_name = [file_name for file_name in sorted(os.listdir(train_data_dir)) if '.txt' in file_name]

    list_of_full_file_path = [train_data_dir / file_name for file_name in list_of_file_name]

    print("불러올 파일 개수: ", len(list_of_full_file_path))

    for i, full_file_name in enumerate(list_of_full_file_path):
        if(i % 100 == 0): print(f'#{i+1}번째 파일: {full_file_name}')
        list_of_source_no, list_of_source_str, list_of_target_str = load_data_from_one_txt(full_file_name=full_file_name)
        
        assert list_of_source_no[0] != 0
   
        list_of_total_source_no.extend([int(num)+14*i for num in list_of_source_no])
        list_of_total_source_str.extend(list_of_source_str)
        list_of_total_target_str.extend(list_of_target_str)

    assert len(list_of_total_source_str) == len(list_of_total_target_str)
    print("전체 파일 읽기 완료!")
    
    return list_of_total_source_no, list_of_total_source_str, list_of_total_target_str



'''
(token -> index) vocab : token_to_idx
(index -> token) vocab : idx_to_token
생성
'''
def create_token_vocab():
    token_to_idx = {}
    idx_to_token = {}

    # dict 없으면
    if not os.path.exists(token_to_idx_filename):
        print()
        print('token_vocab 사전이 존재하지 않습니다.')

        # 스페셜 토큰을 vocab에 추가 
        for special_token in special_tokens:
            token_to_idx[special_token] = len(token_to_idx)

        # 불러온 파일들 토큰화
        counter = Counter()
        print(f'sentence 개수: {len(list_of_total_source_str)}')

        for text in tqdm(list_of_total_source_str):
            tokens_ko = split_fn(text)
            counter.update(tokens_ko)

        tokens = [token for token, cnt in counter.items()]

        # 토큰들을 vocab에 추가
        for token in tokens:
            if not token in token_to_idx:
                token = str(token)
                token_to_idx[token] = len(token_to_idx)


        # token vocab 저장
        with open(token_to_idx_filename, 'w', encoding='utf-8') as io:
            json.dump(token_to_idx, io, ensure_ascii=False, indent=4)

        print('token_vocab 사전 저장 완료...')
        
    
    # dict 있으면
    else:
        print('token_vocab 사전이 존재합니다.')
        
        # token vocab 불러오기
        with open(token_to_idx_filename, 'rb') as f:
            token_to_idx = json.load(f)   

    # 반대 vocab 생성
    idx_to_token = {v: k for k, v in token_to_idx.items()}

    return token_to_idx, idx_to_token




'''
token <-> index 변환 함수 작성
'''
def token2idx_fn(token):
    
    try:
        return token_to_idx[token]
    
    except:
        print("key error: " + str(token))
        token = UNK
        return token_to_idx[token]
    

def idx2token_fn(idx):
    
    try:
        return idx_to_token[idx]
    
    except:
        print("key error: " + str(idx))
        idx = token_to_idx[UNK]
        return idx_to_token[idx]


'''
ner <-> index 변환 함수 작성
'''
def create_ner_vocab():
    ner_to_index = {}
    index_to_ner = {}

    # dict 없으면
    if not os.path.exists(ner_to_index_filename):
        print('ner_vocab 사전이 존재하지 않습니다.')
        regex_ner = re.compile('<(.+?):[A-Z]{3}>') # 객체명 태그가 세자리라서 {3}
        list_of_ner_tag = []
        

        for label_text in list_of_total_target_str:
            regex_filter_res = regex_ner.finditer(label_text)

            for match_item in regex_filter_res:
                ner_tag = match_item[0][-4:-1] # 객체명 태그

                if ner_tag not in list_of_ner_tag:
                    list_of_ner_tag.append(ner_tag)

        # vocab에 스페셜 토큰 추가
        for special_token in special_tokens:
            ner_to_index[special_token] = len(ner_to_index)
        
        # vocab에'O' 태그 추가
        ner_to_index['O'] = len(ner_to_index)

        for ner_tag in list_of_ner_tag:
            ner_to_index['B-'+ner_tag] = len(ner_to_index)
            ner_to_index['I-'+ner_tag] = len(ner_to_index)

        # ner vocab 저장
        with open(ner_to_index_filename, 'w', encoding='utf-8') as io:
            json.dump(ner_to_index, io, ensure_ascii=False, indent=4)
        
    
    # dict 있으면
    else:
        print('ner_vocab 사전이 존재합니다.')
        
        with open(ner_to_index_filename, 'rb') as f:
            ner_to_index = json.load(f)    


    # 반대 vocab 생성
    index_to_ner = {v: k for k, v in ner_to_index.items()}

    return ner_to_index, index_to_ner



'''
START, END token 붙이는 함수
'''
def add_start_token(X_token_batch):
    dec_input_token_batch = [[START_TOKEN] + X_token for X_token in X_token_batch]
    return dec_input_token_batch
    
def add_end_token(X_token_batch):
    dec_output_token_batch = [X_token + [END_TOKEN] for X_token in X_token_batch]
    return dec_output_token_batch



'''
tokens 리스트를 vocab을 이용하여 token id의 리스트로 변환시켜주는 함수
'''
def tokens_list_to_token_ids_list(X_tokens_batch):
    X_token_ids_batch = []
    for X_tokens in X_tokens_batch:
        X_token_ids_batch.append([token2idx_fn(X_token) for X_token in X_tokens])
        
    return X_token_ids_batch



'''
EDA: padding에서 사용할 maxlen 정하기
'''
def eda_of_X_tokens_for_padding_and_return_avg(target_list, title, xlabel, ylabel):
    
    import matplotlib.pyplot as plt

    print('\n\nEDA: padding에서 사용할 maxlen 정해봅니다...')
    print(f'* 타겟 리스트의 총 개수: {len(target_list)}')

    # 그래프에 대한 이미지 사이즈 선언
    # figsize: (가로, 세로) 형태의 튜플로 입력
    plt.figure(figsize=(12, 5))

    # 히스토그램 선언
    # bins: 히스토그램 값들에 대한 버켓 범위
    # range: x축 값의 범위
    # alpha: 그래프 색상 투명도
    # color: 그래프 색상
    # label: 그래프에 대한 라벨
    plt.hist(target_list, bins=10, alpha=0.5, color= 'r', label='word')

    # plt.yscale('1', nonposy='clip')

    # 그래프 제목
    plt.title(title)

    # 그래프 x 축 라벨 = 중복 개수
    plt.xlabel(xlabel)

    # 그래프 y 축 라벨 = 동일한 중복 횟수를 가진 질문의 개수
    plt.ylabel(ylabel)
    
    # EDA
    print(f'길이 최소값: {np.min(target_list)}')
    print(f'길이 최대값: {np.max(target_list)}')
    print(f'길이 평균값: {np.mean(target_list):.2f}')
    print(f'길이 표준편차: {np.std(target_list):.2f}')
    print(f'길이 중간값: {np.median(target_list)}')
    print(f'길이 제 1 사분위: {np.percentile(target_list, 25)}')
    print(f'길이 제 3 사분위: {np.percentile(target_list, 75)}')
    
    return int(np.mean(target_list))



'''
padding 함수
'''
def padding(target_list, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=0.0):
    
    padded_list = pad_sequences(
    target_list, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating, value=value
    )
    
    return padded_list



'''
Transform source sentence
'''
def transform_source_fn(source_text):
    # sentence를 tokens로 변환
    X_tokens = split_fn(source_text)

    # 1. tokens에 cls와 sep도 달아주기 / 2. tokens를 token ids로 변환
    X_tokens_with_cls_sep = [cls_token] + X_tokens + [sep_token]
    X_token_ids_with_cls_sep = [token2idx_fn(X_token) for X_token in X_tokens_with_cls_sep]

    # padding
    padded_X_token_ids_with_cls_sep = padding([X_token_ids_with_cls_sep], value=token2idx_fn(PAD), maxlen=maxlen)
    
    # save token sequence length for matching entity label to sequence label
    prefix_sum_of_token_start_index = []
    sum = 0
    for i, token in enumerate(X_tokens):

        if i == 0:
            prefix_sum_of_token_start_index.append(0)
        else:
            prefix_sum_of_token_start_index.append(sum)

        sum += len(token)
        
    return padded_X_token_ids_with_cls_sep, X_tokens, prefix_sum_of_token_start_index



'''
Transform target sentence
'''
def transform_target_fn(source_text, target_text, X_tokens, prefix_sum_of_token_start_index):
    
    mod_source_text = source_text.replace(" ", "")
    mod_target_text = target_text.replace(" ", "")

    regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
    regex_filter_res = regex_ner.finditer(mod_target_text)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []


    for i, match_item in enumerate(regex_filter_res):
        ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
        ner_text = match_item[1]  # <4일간:DUR> -> 4일간
        start_index = match_item.start()
        end_index = match_item.end()
        m_start_index = match_item.start() - 6 * i  # delete previous '<, :, 3 words tag name, >'
        m_end_index = match_item.end() - 6 - 6 * i
#         print(ner_tag, '\t', ner_text, '\t', start_index, '\t', end_index, '\t', m_start_index, '\t', m_end_index, '\t', i)

        list_of_ner_tag.append(ner_tag)
        list_of_ner_text.append(ner_text)
        list_of_tuple_ner_start_end.append((m_start_index, m_end_index))
        
        
    list_of_ner_label = []
    entity_index = 0
    is_entity_still_B = True

    for tup in zip(X_tokens, prefix_sum_of_token_start_index):
        token, index = tup
        start, end = list_of_tuple_ner_start_end[entity_index]

        if entity_index < len(list_of_tuple_ner_start_end):
            start, end = list_of_tuple_ner_start_end[entity_index]

            if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                is_entity_still_B = True
                entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                start, end = list_of_tuple_ner_start_end[entity_index]

            if start <= index and index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                entity_tag = list_of_ner_tag[entity_index]
                if is_entity_still_B is True:
                    entity_tag = 'B-' + entity_tag
                    list_of_ner_label.append(entity_tag)
                    is_entity_still_B = False
                else:
                    entity_tag = 'I-' + entity_tag
                    list_of_ner_label.append(entity_tag)
            else:
                is_entity_still_B = True
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)

        else:
            entity_tag = 'O'
            list_of_ner_label.append(entity_tag)
    
    with open(ner_to_index_filename, 'rb') as f:
        ner_to_index = json.load(f)
        
    # cls + ner_tag + sep
    padded_ner_ids_with_cls_sep = [ner_to_index['[CLS]']] + [ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [ner_to_index['[SEP]']]
    
    # padding
    padded_ner_ids_with_cls_sep = padding([padded_ner_ids_with_cls_sep], value=token2idx_fn(PAD), maxlen=maxlen)[0]
    
    return padded_ner_ids_with_cls_sep, list_of_ner_label




if __name__ == "__main__":


    ########################################################################
    ## load data
    ########################################################################
    # 데이터 로딩 후 리스트로 소스와 타겟 분리
    list_of_total_source_no, list_of_total_source_str, list_of_total_target_str = load_data_from_all_txt(train_data_dir=train_data_dir)




    ########################################################################
    ## check FileExists
    ########################################################################

    # vocab이 저장될 디렉토리 확인 및 생성
    try:
        vocab_dir.mkdir(parents=True, exist_ok=False)

    except FileExistsError:
        print()
        print(f'vocab_dir directory: {vocab_dir} is already there')


    # 전처리 후 train, val, eval 데이터가 저장될 디렉토리 확인 및 생성
    try:
        train_val_eval_data_dir.mkdir(parents=True, exist_ok=False)

    except FileExistsError:
        print()
        print(f'train_val_eval_data_dir directory: {train_val_eval_data_dir} is already there')
    




    ########################################################################
    ## create vocab
    ########################################################################

    # vocab 생성
    token_to_idx, idx_to_token = create_token_vocab()
    ner_to_index, index_to_ner = create_ner_vocab()





    ########################################################################
    ## preprocessing
    ########################################################################

    # 불러온 데이터 토큰화
    X_tokens_batch = [split_fn(source_str) for source_str in tqdm(list_of_total_source_str)]

    # tokens 리스트를 vocab을 이용하여 token id의 리스트로 변환
    X_tokens_ids_batch = tokens_list_to_token_ids_list(X_tokens_batch)


    # 토큰이 인덱스로 잘 변환됐는지 첫문장으로 확인
    # i: 변환된 token_id
    print('\n\n*** <눈으로 확인하기> 전처리가 잘 됐는지 테스트합니다... 두 행의 값이 같게 나오면 성공적인 것입니다~')
    for i, token_id in enumerate(X_tokens_ids_batch[0]):
        print(idx2token_fn(token_id), '\t', X_tokens_batch[0][i])
    print()


    # padding에서 사용할 maxlen 정하기 -> 보통 문장들의 토큰 개수의 평균값
    length_of_X_tokens = np.array([len(X_tokens) for X_tokens in X_tokens_batch])
    title = 'Log-Histogram of length of sentence'
    xlabel = 'Length of sentence'
    ylabel = 'Counts'
    maxlen = eda_of_X_tokens_for_padding_and_return_avg(length_of_X_tokens, title, xlabel, ylabel)


    # padding
    padded_list = padding(X_tokens_ids_batch, maxlen=maxlen)


    # 소스, 타겟 문장 리스트 전처리    
    print('전처리를 본격적으로 시작합니다...')
    for source_str, target_str in tqdm(zip(list_of_total_source_str, list_of_total_target_str)):

        # 불러온 소스 데이터 한개를 token id로 변환하고 ner 태그 떼어내기 위해 prefix_sum_of_token_start_index 생성
        padded_X_token_ids_with_cls_sep, X_tokens, prefix_sum_of_token_start_index = transform_source_fn(source_str)
        
        # 리스트에 추가
        list_of_X_tokens.append(X_tokens)
        list_of_padded_X_token_ids_with_cls_sep.append(padded_X_token_ids_with_cls_sep[0])
        
        # 불러온 타겟 데이터 한개를 token id로 변환
        padded_ner_ids_with_cls_sep, list_of_ner_label = transform_target_fn(source_str, target_str, X_tokens, prefix_sum_of_token_start_index)
        
        # 리스트에 추가
        list_of_padded_ner_ids_with_cls_sep.append(padded_ner_ids_with_cls_sep)


    # 전처리 잘됐나 테스트
    print('\n\n*** <눈으로 확인하기> 전처리가 잘 됐는지 테스트합니다... ')
    print('[ 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 32, 33, 34]')
    print(list_of_padded_X_token_ids_with_cls_sep[0])
    print('\n[ 4, 11, 12, 12, 12, 13, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 12, 10, 15, 10, 10, 10, 10, 10]')
    print(list_of_padded_ner_ids_with_cls_sep[0])
    print()




    ########################################################################
    ## split and save 'train, val, eval' data
    ########################################################################

    # train, val, eval 데이터로 분리 후 저장
    ner_tag_size = len(ner_to_index) # y라벨 원핫인코딩 위한 config
    X, y = np.array(list_of_padded_X_token_ids_with_cls_sep), np.array(list_of_padded_ner_ids_with_cls_sep)
    y2 = to_categorical(y, num_classes=ner_tag_size, dtype='float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, shuffle=True, random_state=1004)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, shuffle=True, random_state=1004)

    np.save('train_val_eval_data/X_train', X_train)
    np.save('train_val_eval_data/y_train', y_train)
    np.save('train_val_eval_data/X_val', X_val)
    np.save('train_val_eval_data/y_val', y_val)
    np.save('train_val_eval_data/X_test', X_test)
    np.save('train_val_eval_data/y_test', y_test)





    
    ########################################################################
    ## save configuration
    ########################################################################

    # config 저장
    print('\n*** config를 파일로 저장합니다...')
    config = {}
    config['epochs'] = 10
    config['embedding_dim'] = 32
    config['vocab_size'] = len(token_to_idx)
    config['dense_feature_dim'] = 512
    config['dropout_rate'] = 0.2
    config['ner_tag_size'] = len(ner_to_index)
    config['maxlen'] = maxlen

    with open(config_filename, 'w', encoding='utf-8') as io:
        json.dump(config, io, ensure_ascii=False, indent=4)

    print('*** config 파일 저장 완료!')





    ########################################################################
    ## END
    ########################################################################
    print('\n\n\n~~~모든 전처리 과정이 끝났습니다!!~~~')