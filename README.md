
# SQL Generation through Classification Answer Selector by LLM (SCAS)
This repo contains the source code to reporduce result obtained in the [SCAS](https://aclanthology.org/2024.clinicalnlp-1.63.pdf).

## Steps to Reproduce

- Install requirements
- Create .env file with following openai details
    ```
    OPENAI_API_TYPE =    <api_type>
    OPENAI_API_KEY =     <api_key>
    OPENAI_API_BASE =    <api_base>
    OPENAI_API_VERSION = <api_version>
    ```
- Classification Phase :
    - For creating custom tokenizer:
        ```
        python cls_tokenizer.py --output_path <output_path>
        ```
    - For training CodeLLama for sequence classification:
        ```
        torchrun \
        --nproc_per_node 4 cls_code_llama_train.py \
        --model_name "codellama/CodeLlama-7b-Instruct-hf" \
        --gradient_checkpointing \
        --all_linear
        ```
    - For Generating inference from trained CodeLLama sequence Classifier:
        ```
        python cls_code_llama_infer --model_path <model_path>
        ```
    - For Generating inference LLM Answer Selector:
        ```
        python cls_answer_selector.py \
        --code_llama_result_path <codellama result path> \
        --custom_tokenizer_result_path <custom tokenizer result path> \
        --submission_result_path <output path>
        ```

- SQL Generation Phase:
    - For training CodeLLama:
        ```    
        torchrun \
        --nproc_per_node 4 sql_code_llama_train.py \
        --model_name "codellama/CodeLlama-7b-Instruct-hf" \
        --split_model \
        --gradient_checkpointing \
        --all_linear \
        ```
    - Merging the model
        ```
        python merge.py \
        --base_model <base model path>
        --adapter_path <adapter path>
        --output <directory to save merged model>
        ```
    - For inferencing CodeLLama:
        ```
        python sql_code_llama_infer.py \
        --model_path <adapter merged model path> \
        --output_directory <output directory for saving results>
        ```
    - For abstaining:
        ```
        python abstain.py \
        ---sql_generated_data_path <generated sql json file path> \
        --classification_data_path <generated classification json file path>
        --final_result_path <directory to save final result path>
        ```
## Citation
Please cite with below link. Also, If you have any question, contact to the corresponding author. Thank you
```
@inproceedings{jabir-etal-2024-saama,
    title = "Saama Technologies at {EHRSQL} 2024: {SQL} Generation through Classification Answer Selector by {LLM}",
    author = "Jabir, Mohammed  and
      Kanakarajan, Kamal  and
      Sankarasubbu, Malaikannan",
    editor = "Naumann, Tristan  and
      Ben Abacha, Asma  and
      Bethard, Steven  and
      Roberts, Kirk  and
      Bitterman, Danielle",
    booktitle = "Proceedings of the 6th Clinical Natural Language Processing Workshop",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.clinicalnlp-1.63",
    doi = "10.18653/v1/2024.clinicalnlp-1.63",
    pages = "655--671",
    abstract = "The EHRSQL task aims to develop a dependable text-to-SQL model for Electronic Health Records (EHR) databases, which are crucial sources of clinical data that store patients{'} medical histories in hospitals. Large language models (LLM) have been proven to exhibit state-of-the-art performance for text-to-SQL tasks across various domains. To this end, we have developed a framework, SQL Generation through Classification Answer Selector by LLM (SCAS), which comprises two modules. The CAS module determines the answerability of the question, while the SG model generates the SQL query exclusively for answerable questions. Our system ranked 7th on the leaderboard with a Reliability Score of 53.21 on the official test set.",
}
```
