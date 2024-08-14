# RAG PDF Helper

## Overview

The RAG PDF Helper is an application that enables users to ask questions about a PDF document loaded into the system. Utilizing LangChain for document processing and OpenAI's powerful language model, this tool allows for dynamic interactions with the content of the PDF, providing responses based on the information extracted from the document.

## Features

- Load PDF documents and process their content.
- Retrieve information from the PDF using natural language queries.
- Seamless integration of advanced language models for accurate and context-aware responses.
- Uses embeddings for improved retrieval of relevant information.

## Requirements

To run this application, you need the following:

- Python 3.7 or higher
- Required Python packages:
  - chainlit
  - langchain
  - openai (for embeddings and models)
  - pdfplumber (for PDF loading)
  - chromadb (for vector store functionality)

## Installation

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install the required packages:

    ```bash
    pip install chainlit langchain openai pdfplumber chromadb
    ```

3. (Optional) Set up environment variables for OpenAI API keys:

    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```

## Usage

1. Start the application:

    ```bash
    chainlit run app.py
    ```

2. Upload a PDF file using the provided interface.

3. After the PDF is loaded, you can ask questions related to the content of the document.

## Code Structure

- **app.py**: Main application file that contains the logic for loading PDFs, extracting information, and handling user queries using LangChain's models and tools.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please submit an issue or a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com) for its powerful framework for building applications with LLMs.
- [OpenAI](https://openai.com) for providing the language model.
- Contributors and open-source community for their continuous support and enhancements. 

For any queries or further information, please feel free to reach out via the project's issue tracker.