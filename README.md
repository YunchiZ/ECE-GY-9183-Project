
## MLops: Info Lens 👓 - Automatic News Classification, Content Summary, True and False Identification

### Value Proposition
A cloud-deployed machine learning operation system that are designed for automatic and quick-response news content extraction. 

The system is aimed at benefiting any user or professional journalist who wants to process news information efficiently through 3 functions:
- [x] **News Content Summary**
- [x] **Identification**: True / Fake 
- [x] **News Category Classification**: Business/Politics/Food & Drink/Travel/Parenting/Style & beauty/Wellness/World news/Sports/Entertainment

The system is **expected** to achieve a total of tens of milliseconds of reasoning time for three functional responses per request, avoiding the high cost of time-consuming and requiring professional processing and verification when compared to traditional non-ML businesses. Besides, the system **will be** host at least dozens of API calls per second, which makes business efficiency significantly higher than traditional services.

The overall system is judged by the following **business** metric: 
- Feedback from users: average satisfaction per batch(for content summary); average accuracy per batch(for identification & classification)
- 

<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be 
used in an existing business or service. (You should not propose a system in which 
a new business or service would be developed around the machine learning system.) 
Describe the value proposition for the machine learning system. What’s the (non-ML) 
status quo used in the business or service? What business metric are you going to be 
judged on? (Note that the “service” does not have to be for general users; you can 
propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |                 |                                    |
| Team member 1                   |                 |                                    |
| Team member 2                   |                 |                                    |
| Team member 3                   |                 |                                    |
| Team member 4 (if there is one) |                 |                                    |



### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->
We aim to build and optimize large-scale models for three natural language processing (NLP) tasks:

+ Text Summarization: Generating summaries for long articles.
+ Text Classification: Categorizing news articles into predefined categories (e.g., politics, sports, health).
+ Fake News Detection: Classifying news articles as "real" or "fake" based on the content.

For each task, we will select appropriate pre-trained models, fine-tune them, and evaluate the impact of various training strategies, particularly for large models, using distributed training to optimize training time and resource efficiency.


1. Model Selection: 
    - Task 1: Summary Generation
        - We plan to use pre-trained BART model provided by Hugging Face. (https://huggingface.co/facebook/bart-large-cnn)

        ```python 
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "facebook/bart-large-cnn" 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        ```
        - Processing of input text: Before entering the text into the BART model, the text needs to be tokenized. A Hugging Face's word tokenizer automatically converts the text into a format the model can understand.

        ```python 
        inputs = tokenizer([article], max_length=1024, truncation=True, return_tensors="pt")
        ```

        - Generating summary: Once the input text is processed and ready, it can be passed to the BART model to generate a summary.

        ```python 
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        ```

        - Parameters setting: 
            - num_beams: Controlling the width of the beam search, higher values can produce higher quality summaries, but the computational cost will increase
            - max_length: Controls the maximum length of the generated summary
            - length_penalty: Length penalties when generating summaries to help avoid generating summaries that are too long or too short
            - temperature: Used to control randomness when generating summaries, lower values generate summaries that are more certain and higher values generate summaries that are more random.
        
        - Evulation: The quality of the model-generated summaries can be evaluated using a common text summary evaluation metrics, ROUGE.

    - Task 2: Classify
        - We plan to use pre-trained DistilBERT model provided by Hugging Face. (https://huggingface.co/distilbert/distilbert-base-uncased)

        ``` Python
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n)
        ```

        - Processing of input text: we can use the tokenized summary generated during task 1, or tokenize it again.

        ``` Python
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ```

        - Processing of labels: create a map of the expected ids to their labels with id2label and label2id

        - Parameters setting:
            - warmup_steps
            - weight_decay
            - per_device_train_batch_size
            - per_device_eval_batch_size
            - logging_dir

    - Task 3: Fake news detection
        - We plan to use pre-trained XLNet model provided by Hugging Face. (https://huggingface.co/xlnet/xlnet-base-cased)

        ```Python
        model_name = "xlnet-base-cased"
        ```

        - Preprocess: 

        ``` Python
        dataset = Dataset.from_dict(data)
        ```

        - Processing of input text: we can use the tokenized summary generated during task 1, or tokenize it again.

        ``` Python
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ```

        - Parameters setting:
            - warmup_steps
            - weight_decay
            - per_device_train_batch_size
            - per_device_eval_batch_size
            - logging_dir
    
2. Training Strategies: For training these large models, especially with large datasets, we will use some strategies to ensure both efficiency and scalability. The strategies will address both model fine-tuning and large-scale training jobs that may not fit on a low-end GPU.
    - Fine-Tuning with Pre-trained Models: We will fine-tune the models using task-specific data, starting from the pre-trained versions available in Hugging Face’s Transformers library. This approach leverages transfer learning to adapt models to the specific tasks while benefiting from pre-trained weights, which reduces the time and computational cost.

    - Batch Size Adjustments: We will explore different batch sizes and evaluate their effect on training time and model performance. We may adjust the batch size based on the GPU memory available.

    - Gradient Accumulation: If the model does not fit on the available GPU memory with a large batch size, we will use gradient accumulation. This allows us to simulate a larger batch size by accumulating gradients over multiple mini-batches before performing an update.

    - Mixed Precision Training: We will use mixed precision training to speed up training and reduce memory usage. By using 16-bit floating point operations instead of 32-bit, we can significantly reduce training time without compromising model accuracy.

3. Distributed Training
    - Data Parallelism: We will use Distributed Data Parallel (DDP) for distributed training. DDP works by splitting the dataset across multiple devices, where each GPU processes a subset of the data, and gradients are averaged across all GPUs at each step.

    - Fully Sharded Data Parallel: FSDP divides model parameters and optimizes them in parallel across multiple GPUs, helping to minimize memory overhead and improve scalability for models that don’t fit into the memory of a single GPU.

4. Training Time Evaluation: We will conduct experiments to evaluate the effect of distributed training strategies (DDP vs. FSDP) and batch size on training time.

    - Experiment Design: 
        - Single GPU vs. Multiple GPUs: We will measure the training time for a given model with one GPU and compare it to training on multiple GPUs (e.g., 1, 2, 4 GPUs).

        - FSDP vs. DDP: We will compare training with DDP and FSDP, looking at training time and model performance.

        - Batch Size and Training Time: We will vary the batch size to identify the optimal batch size for training efficiency.
    
    - Metrics:
        - Total Training Time: The total time it takes to complete model training.

        - Speed-up Factor: The ratio of training time using one GPU to training time using multiple GPUs.

        - Model Performance: Accuracy, F1 score, and loss for each strategy to ensure that the performance is not sacrificed for speed.

5. Experiment Tracking with MLFlow
    - We will deploy MLFlow on a Chameleon node and use its logging capabilities to track metrics such as training loss, validation accuracy, model parameters, and other relevant artifacts. MLFlow will provide the ability to track experiments, compare results, and visualize model performance over time.

    - Integration with Training Code
    ``` Python
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.0001)

    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_accuracy", val_accuracy) 

    mlflow.pytorch.log_model(model, "model")
    ```

6. Scheduling Training Jobs with Ray

    - Ray Cluster Setup: We will provision a Ray cluster on Chameleon, which will consist of several nodes (each with one GPU) to handle parallel training jobs.

    - The Ray cluster will be configured with the necessary resources, such as CPU cores, GPU devices, and storage, for efficient model training.

    - We will configure checkpointing to store model progress in remote storage, enabling recovery in case of failure.

    - Integration with Training Code:
    ``` Python
    import ray
    from ray import train
    from ray.train import Trainer

    trainer = Trainer(
        model=model,
        train_data=train_data,
        eval_data=eval_data,
        config=config
    )
    
    trainer.fit()

    trainer = train.tune.trainable(train_model)
    trainer.fit()
    ```

7. Hyperparameter Tuning with Ray Tune
    - Define search space: We will define the search space for hyperparameters such as learning rate, batch size, optimizer type, etc.

    - Run tuning jobs: Ray Tune will schedule multiple training jobs with different hyperparameter configurations, evaluating each configuration's performance.

    - Stop early: Ray Tune’s early stopping feature will terminate underperforming trials, making the tuning process more efficient.

    - Integration with Training Code:
    ``` Python
    from ray import tune

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "optimizer": tune.choice(["adam", "sgd"]),
    }

    tune.run(
        train_model, 
        config=search_space,
        num_samples=10,
        resources_per_trial={"cpu": 1, "gpu": 1},
        name="hyperparameter_tuning"
    )
    ```

#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


