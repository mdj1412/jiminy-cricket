import argparse
import os
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print (f"device: {device}")

class NearestNeighborSelector():
    """
        Retrieval-based prompt selection approach

        [ Sentence Embeddings for Retrieval ]

        Used model (Sentence Encoder, retrieval module):
        1. princeton-nlp/unsup-simcse-bert-large-uncased
        2. princeton-nlp/unsup-simcse-roberta-large
        3. princeton-nlp/sup-simcse-bert-large-uncased
        4. princeton-nlp/sup-simcse-roberta-large

        Permutation Case:
        1. default order ( d(x_i, x) < d(x_j, x) if i < j )
        2. reverse order ( d(x_i, x) > d(x_j, x) if i < j )
        3. Bootstrapping via Permutation
            3.1. Compute the majority voting of all (permutations) predictions.
            3.2. Compute the average of all permutations of sentence embeddings 
                and predict the sentence embedding that is closest to this average. 
                (Consider k!)

        Metric :
        1. euclidean
        2. cosine
    """
    def __init__(self, encoder_name, train_dataset, cluster_name, batch_size=128, device='cuda'):
        assert encoder_name in ["princeton-nlp/unsup-simcse-bert-large-uncased", "princeton-nlp/unsup-simcse-roberta-large",
                                "princeton-nlp/sup-simcse-bert-large-uncased", "princeton-nlp/sup-simcse-roberta-large"]
        assert cluster_name in ["DBSCAN", "HDBSCAN"]

        if 'cuda' not in device:
            raise ValueError(f"Target device should be a gpu. Device {device} is not supported")
    
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = AutoModel.from_pretrained(encoder_name)
        self.encoder_name = encoder_name
        self.cluster_name = cluster_name
        self.batch_size = batch_size

        self.model.to(device=device)# Use GPU Memory

        self.train_dataset = train_dataset
        # train_dataloader = DataLoader([qa['question'] for qa in train_dataset], batch_size=batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        print ("Convert the questions in the train dataset to sentence embeddings. "); start = time.perf_counter()
        self.train_q_embeddings = []
        for train_i, batch in enumerate(tqdm(train_dataloader)):
            # Convert train (batch) questions to embeddings
            train_q_embeddings = self.convert_embedding(
                batch
            )
            self.train_q_embeddings.append(train_q_embeddings)
        self.train_q_embeddings = torch.cat(self.train_q_embeddings, dim=0)
        print (f"Finish convert the train sentence embeddings : {time.perf_counter()-start} sec")

        print ("start load dimension reduction model "); start = time.perf_counter()
        if self.cluster_name == "DBSCAN":
            # epsilon, 최소 샘플 개수 설정
            cluster_model = DBSCAN(eps=0.5, min_samples=2)
        elif self.cluster_name == "HDBSCAN":
            cluster_model = HDBSCAN(eps=0.5, min_samples=2)
        else: 
            raise NotImplementedError()
        
        # 군집화 모델 학습 및 클러스터 예측 결과 반환
        cluster_model.fit(self.train_q_embeddings)
        # df_scale['cluster'] = model.fit_predict(df_scale)
        self.cluster_model = cluster_model
        print (f"Finish preparing dimension reduction model : {time.perf_counter()-start} sec")

    def convert_embedding(self, inputs):
        # Tokenize input texts
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in inputs.items()}# "input_ids", "token_type_ids", "attention_mask"

        # Get the embeddings
        with torch.no_grad():
            # Get [CLS] token vector (classification token)
            input_embeddings = self.model(**input_data, output_hidden_states=True, return_dict=True).pooler_output

        return input_embeddings.detach().cpu()
    
    def predict_clustering(self, save_file_path, min_samples):
        df_scale = pd.DataFrame({})

        # 축소한 차원의 수를 정합니다.
        n_components = 2
        # TSNE 모델의 인스턴스를 만듭니다.
        reduction_model = TSNE(n_components=n_components)
        # data를 가지고 TSNE 모델을 훈련(적용) 합니다.
        X_embedded = reduction_model.fit_transform(self.train_q_embeddings)

        X_embedded = X_embedded.tolist()
        df_scale['X'] = [cordinate[0] for cordinate in X_embedded]
        df_scale['Y'] = [cordinate[1] for cordinate in X_embedded]
        
        # 다중 플롯 동시 시각화
        f, ax = plt.subplots(2, 2)
        f.set_size_inches((12, 12))

        for i in range(4):
            # epsilon을 증가시키면서 반복
            eps = 0.4 * (i + 1)

            # 군집화 및 시각화 과정 자동화
            model = DBSCAN(eps=eps, min_samples=min_samples)

            model.fit(self.train_q_embeddings)
            df_scale['cluster'] = model.fit_predict(self.train_q_embeddings)
            print (f"{eps} {min_samples} : {df_scale['cluster'].values}")

            for j in range(-1, df_scale['cluster'].max() + 1):
                ax[i // 2, i % 2].scatter(
                    df_scale.loc[df_scale['cluster'] == j, 'X'], 
                    df_scale.loc[df_scale['cluster'] == j, 'Y'], 
                    label = 'cluster ' + str(j)
                )

            ax[i // 2, i % 2].legend()
            ax[i // 2, i % 2].set_title('eps = %.1f, min_samples = %d'%(eps, min_samples), size = 15)
            ax[i // 2, i % 2].set_xlabel('x', size = 12)
            ax[i // 2, i % 2].set_ylabel('y', size = 12)

        # 그림을 저장 (파일명과 경로를 지정)
        f.savefig(f"madrag-figures/{save_file_path}-min_samples({min_samples}).png", dpi=300, bbox_inches='tight')
        plt.close(f)  # 리소스 해제 및 시각화 중단  

def main(min_samples=12):
    path = "/MIR_NAS/jerry0110_1/madrag/madrag/processed_data/"
    print (os.listdir(path))

    text_anomal_image_normal_path = os.path.join(path, "text_anomal_image_normal")
    text_normal_image_anomal_path = os.path.join(path, "text_normal_image_anomal")
    print (os.listdir(text_anomal_image_normal_path))
    print (os.listdir(text_normal_image_anomal_path))

    text_an_image_n_file = os.path.join(text_anomal_image_normal_path, "data-00000-of-00001.arrow")
    text_n_image_an_file = os.path.join(text_normal_image_anomal_path, "data-00000-of-00001.arrow")

    # Load the .arrow file into a Dataset
    an_n_dataset = Dataset.from_file(text_an_image_n_file)
    n_an_dataset = Dataset.from_file(text_n_image_an_file)

    an_n_list = [item['caption'] for item in an_n_dataset]
    an_n_nns = NearestNeighborSelector(
        "princeton-nlp/unsup-simcse-bert-large-uncased", 
        an_n_list, 
        cluster_name="DBSCAN", 
        device='cuda'
    )
    an_n_embeddings = an_n_nns.predict_clustering(
        save_file_path="text_anomal_image_normal",
        min_samples=min_samples
    )

    n_an_list = [item['caption'] for item in n_an_dataset]
    n_an_nns = NearestNeighborSelector(
        "princeton-nlp/unsup-simcse-bert-large-uncased", 
        n_an_list, 
        cluster_name="DBSCAN", 
        device='cuda'
    )
    n_an_embeddings = an_n_nns.predict_clustering(
        save_file_path="text_nomal_image_anormal",
        min_samples=min_samples
    )

    print ("finish !")
    exit()

if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser()

    # 2. add arguments to parser
    parser.add_argument('--min_samples',
                        type=int,
                        default=12,
                        help="~")
    
    # 3. parse arguments
    args = parser.parse_args()

    # 4. use arguments
    print (args)
    main(args.min_samples)