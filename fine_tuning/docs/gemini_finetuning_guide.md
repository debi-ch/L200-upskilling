# Fine-tuning Gemini Models on Vertex AI

> **IMPORTANT**: For the latest information on Gemini supervised fine-tuning, refer to the [official Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning).

This guide provides a streamlined approach to fine-tuning Gemini models using Vertex AI. The process has been simplified to make it more accessible while maintaining all the necessary functionality.

## Prerequisites

Before you begin, ensure you have:

- A Google Cloud account with Vertex AI API enabled
- The [OpenAssistant Guanaco dataset](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) properly formatted for fine-tuning
- IAM permissions:
  - `roles/aiplatform.user` or `roles/aiplatform.admin`
  - `roles/storage.admin` for working with Cloud Storage buckets
- A Python environment with required packages

## Setting Up Your Environment

1. Install required packages:

```bash
pip install google-cloud-aiplatform>=1.36.0 google-cloud-storage vertexai
```

2. Configure authentication with Google Cloud:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

3. Set required environment variables:

```bash
export PROJECT_ID="your-project-id"
export LOCATION="us-central1"  # Must use a region that supports Gemini fine-tuning
export BUCKET_NAME="${PROJECT_ID}-gemini-tuning"
```

## Dataset Preparation

1. Download the OpenAssistant Guanaco dataset:

```python
# transform_dataset.py
import os
import json
from datasets import load_dataset

# Create directory for data
os.makedirs("data/openassistant-guanaco", exist_ok=True)

# Download dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
train_data = dataset["train"]

# Format dataset for Gemini fine-tuning
output_file = "data/openassistant-guanaco/openassistant_best_replies_train_reformatted.jsonl"
with open(output_file, "w") as f:
    for item in train_data:
        input_text = item["text"].split("Human: ")[1].split("Assistant: ")[0].strip()
        output_text = item["text"].split("Assistant: ")[1].strip()
        
        entry = {
            "input_text": input_text,
            "output_text": output_text
        }
        f.write(json.dumps(entry) + "\n")

print(f"Dataset reformatted and saved to {output_file}")
```

2. Run the transformation script:

```bash
python transform_dataset.py
```

3. Verify the JSONL format is correct:

```bash
head -n 1 data/openassistant-guanaco/openassistant_best_replies_train_reformatted.jsonl
```

Each line should be a JSON object with `input_text` and `output_text` fields.

## Simplified Fine-Tuning Process

The fine-tuning process has been streamlined into a single script (`finetune_gemini_streamlined.py`) that handles:

1. Creating a Google Cloud Storage bucket (if it doesn't exist)
2. Uploading your dataset to the bucket (if it exists locally)
3. Configuring and running the fine-tuning job
4. Testing the fine-tuned model

To run the fine-tuning process:

```bash
python finetune_gemini_streamlined.py
```

The script will automatically:

- Initialize Vertex AI with your project and location
- Create or verify a staging bucket in the correct location
- Configure a fine-tuning job for Gemini 2.0 Flash
- Execute the fine-tuning process
- Log progress and results

## Monitoring Fine-Tuning Progress

Fine-tuning can take several hours depending on your dataset size. You can monitor progress using:

1. **Google Cloud Console**:
   - Navigate to Vertex AI > Training
   - Find your fine-tuning job in the list
   - Check the status and logs

2. **Command Line**:
   ```bash
   gcloud ai custom-jobs describe JOB_ID --region=LOCATION
   ```

3. **Cloud Storage Bucket**:
   - Monitor the output directory in your bucket for checkpoints and artifacts
   - `gsutil ls gs://YOUR_BUCKET_NAME/output/`

4. **Email Notifications**:
   - In the Google Cloud Console, set up alerts for job status changes
   - Navigate to Monitoring > Alerting > Create Policy
   - Configure alerts based on custom job status

## Troubleshooting Common Issues

### Error 1: Invalid Dataset Format

**Symptoms**: Job fails with an error mentioning invalid examples or dataset format.

**Solution**: 
- Ensure your dataset follows the required format with `input_text` and `output_text` fields
- Verify there are no malformed JSON entries in your JSONL file

### Error 2: Insufficient Permissions

**Symptoms**: Permission denied errors when creating buckets or running jobs.

**Solution**:
- Verify you have the required IAM roles
- Run `gcloud auth application-default login` again

### Error 3: Region Compatibility

**Symptoms**: Error that the selected region doesn't support Gemini fine-tuning.

**Solution**:
- Use a supported region like `us-central1`
- Update both your script configuration and environment variables

### Error 4: Model Compatibility

**Symptoms**: Error about using an unsupported model for fine-tuning.

**Solution**:
- Only Gemini 2.0 Flash and Gemini 2.0 Flash-Lite currently support supervised fine-tuning
- Update your script to use one of these models:
  ```python
  BASE_MODEL = "gemini-2.0-flash"  # or "gemini-2.0-flash-lite"
  ```

## Testing Your Fine-Tuned Model

After fine-tuning completes, you can test your model using:

```python
from vertexai.generative_models import GenerativeModel

# Initialize the fine-tuned model
model = GenerativeModel("YOUR_TUNED_MODEL_NAME")

# Test with a prompt
response = model.generate_content("Your test prompt here")
print(response.text)
```

## Important Notes

- Only Gemini 2.0 Flash and Gemini 2.0 Flash-Lite currently support supervised fine-tuning
- Fine-tuning jobs can take several hours depending on dataset size
- Billing occurs while the fine-tuning job is running
- Always refer to the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning) for the current list of supported models and features 