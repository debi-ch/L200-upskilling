# Gemini API Examples

This directory contains examples of how to interact with Gemini models using different API approaches.

## Contents

- `gemini_sdk.py`: Demonstrates using the Vertex AI SDK to interact with Gemini models, including both base and fine-tuned versions. This is the approach used in the main application.

- `run_gemini_sdk.sh`: Shell script to run the SDK example with the correct environment variables.

## Purpose

These examples are provided as a reference for:

1. **Implementation Guidance**: Shows how to implement Gemini model integration using the Vertex AI SDK
2. **Debugging**: Can be used to test API access independently from the main application
3. **Learning**: Demonstrates best practices for Gemini model integration

## Usage

To run the example:

```bash
# From the api_examples directory
chmod +x run_gemini_sdk.sh
./run_gemini_sdk.sh
```

## Additional Resources

Other API approaches (REST API, direct API, API key-based access) were previously available but have been archived for simplicity. If you need examples of these approaches, they can be found in the `archive` subdirectory.

## Main Application Integration

The main application in `app/backend/models/gemini_chat.py` uses the same approach as demonstrated in the `gemini_sdk.py` example. The key components are:

1. **Initialization**: Using `vertexai.init()` to set up the API client
2. **Model Loading**: Creating a `GenerativeModel` instance with the model name
3. **Content Generation**: Using `generate_content()` to get responses
4. **Error Handling**: Implementing fallback mechanisms if the primary model fails 