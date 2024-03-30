
# Classifier aggregated SQL Generation

## Steps to Reduce

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
