### 3.Data pipeline (Major: Haorong Liang)

#### Persistent Storage

We use two types of storage: 
1. Two object storage space to store the raw datasheets, model weights, tokenizers and offline evaluation code.
2. One block storage space to store the service that run on VM instance.

**Object Storage 1**: object-persist-project28-train located at CHI@TACC. VM can bind the bucket by run this script: [script](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/main/mount_object_store_train.sh).The total size is 3.60GB. After mount it with VM-training, the folder structure is shown as follows:

   ```
   /mnt/object/                             # General data storage shared by train and ETL container
   ├── models/                              # Pre-saved Hugging Face hub images
   │   ├── bart_source                      # News summary task
   |   ├── bert_source                      # News classification task
   |   └── xln_source                       # True/Fake News task
   |
   ├── etl_data/                            # Training datasheets and online evaluation datasheets
   │   └── task1_data
   |        └── summary_train.csv
   │   └── task2_data
   |        └── welfake_train.csv
   │   └── task3_data
   |        └── classification_train.csv

   ```


**Object Storage 2**:object-persist-project28-infer located at CHI@TACC. VM can bind the bucket by run this script: [script](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/main/mount_object_store_infer.sh).The total size is 4.85GB. The bucket is used to store the inference model, tokenizer and data to simulate online user request. After mount it with vm-infer, the folder structure is shown as folllows:

   ```
   /mnt/object/                             # General data storage for inference
   ├── models/                              # Pre-saved Hugging Face hub images
   │   ├── BART                             # News summary task
   |   ├── BERT                             # News classification task
   |   └── XLN                              # True/Fake News task
   |
   ├── eval_data/                           # Online data to simulate user requests
   │   ├── summary_eval.jsonl               # News summary task
   |   ├── welfake_eval.jsonl               # News classification task
   |   └── classification_eval.jsonl        # True/Fake News task

   |
   ├── on_tests/                            # Stored tokenizer used for inference task
   │   ├── bart_source
   │   ├── bert_source
   │   └── xln_source

   ```

**Block Storage**: We use one block storage to store the monitor data volumes for Minio, wandb and load testing results.VM can bind this volume by running this script:[script](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/main/mount_block_store.sh).The total size is expanded when running the system. After mount it with vm-ops, the folder structure is shown as follows:

   ```
   /mnt/block/                             # General data storage 
   ├── minio_data/                         # folder to save minio logging files
   |
   ├── wandb_data/                         # folder to save wandb logging files
   |
   ├── on_test/                           
   │   ├── locustfile.py                   # ???改路径
   │   └── sample.jsonl                    # datasheet for load testing


   ```


#### Offline Data

There are three datasheets in out task. 
1. CNN_Dailymail dataset is used for summarision. It contains news articles (from 2007–2015) and their human-written highlights, which are concatenated into summaries. The data was collected from Wayback Machine archives and preprocessed using the PTBTokenizer (lowercasing, punctuation normalization). Each sample simulates how a news service summarizes articles for readers. In production, summaries may not be immediately available, and later feedback (e.g., user clicks) could serve as ground truth for improvement.

2. HuffPost News Category Dataset called News Category Datasheet (2012–2018), containing 45,500 news headlines evenly distributed across 10 categories such as Business, Politics, Wellness, and Entertainment. Each class includes exactly 4,500 samples. The dataset is derived from the original Kaggle dataset and modified to be beginner-friendly by removing noise and balancing class distribution. Each sample represents a real HuffPost article headline and its topic label. In production use (e.g., news apps or recommender systems), ground truth categories may come from editorial tags. User engagement (e.g., click-through rate) could later provide weak supervision signals.

3. WELFake dataset is used in true/fake identification task, which consists of 72,134 labeled news articles: 35,028 real and 37,106 fake. The dataset merges four well-known sources (Kaggle, McIntire, Reuters, BuzzFeed Political) to ensure diversity and avoid overfitting. The data includes article titles and full text, labeled as 0 (fake) or 1 (real). Each sample simulates a user-posted or shared news item. In production, articles may arrive unlabeled and require human verification. Ground truth labels (fake/real) are binary and often only available after manual fact-checking or crowdsourcing.

The sample input and output is shown as follows:

| Outside Materials|Name | Sampel Input 1 | Sampel Input 2 | Sample Output |
|------------------|-----|--------------|------------|---------------|
| Data set 1   | CNN_Dailymail      | LONDON, England(Reuters)-- HarryPotter star Daniel Radcliffe gains accessto a reported ￡20million fortune as heturns 18 on Monday...|NAN|Harry Potter starDaniel Radeliffe gets￡20M fortune as heturns 18 Monday... |
| Data set 2   | News Category Dataset | Resting is part of training. I'veconfirmed what I sortof already knew: I'mnot built for runningstreaks... |NAN| WILLNESS |
| Data set 3   | WELFake Dataset | L4W ENFORCEMENT ONHIGH ALERT FollowingThreats Against Cops4nd whites On 9-11By#BlackLivesMatter And#FYFq11 Terrorists|RCEMENT ON HIGH ALERT Following Threats Against Cops4nd whites On 9-11By#BlackLivesMatter And#FYF911 Terrorists | 1|

#### Data Pipeline

Raw data is downloaded from Kaggle & Hugging face and stored in object storage (object-persist-project28-train and object-persist-project28-eval) for training and online evaluation, while the datasheet for load testing is stored in block sotrage.
The model is split into training & test and evaluation data using the fixed random seed. 

Pre-processing of the raw datasheet includes merging the title and body text into a single input field, drop the NAN value, cleaning invalid characters and saving the processed data for model consumption. The clean version of processed training data is store in object storage and could be read by train docker at the starting stage. [Online-Data-Pipeline](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/Data-pipeline/data_pipeline/etl_app.py)

The online evaluation data is read from MinIO by monitor container's auto trigger or manager's trigger manually[Trigger](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/f49a7bfce85137d94fd3fd56e0fe6fc2301ddf98/data_pipeline/etl_app.py#L223). Then the container would download evaluation.db file. Next is the preprocess of data including dropout NAN and duplicate values. Finally, the data would be append to the existing .csv file and trigger the train container to retrain the model.

#### Data Dashboard

The dashboard is integrated with Prometheus and monitors data quality metrics in real time. It tracks missing values using the data_quality_missing_values gauge, which records the number of NaN entries based on task and field. Duplicate entries are captured by the data_quality_duplicates gauge, while any data processing failures are counted using the data_quality_processing_errors counter, categorized by error type [metrics defination](https://github.com/YunchiZ/ECE-GY-9183-Project/blob/f49a7bfce85137d94fd3fd56e0fe6fc2301ddf98/data_pipeline/etl_app.py#L28C1-L30C127).
These metrics help ensure the reliability and integrity of the dataset during pipeline execution.